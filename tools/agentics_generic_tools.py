"""
Tools using agentics framework
"""

from crewai.tools import BaseTool
from typing import Type, Optional, List
from pydantic import BaseModel, Field
from agentics import AG
import pandas as pd
import json
import asyncio
import os
import csv
import sys
import io
import traceback


class TransductionAnswer(BaseModel):
    short_answer: str | None = None
    detailed_answer: str | None = Field(
        None,
        description="A detailed Markdown Document reporting evidence for the above answer and a more detailed explanation"
    )
    explanation: str | None = Field(
        None,
        description="A detailed explanation of the answer"
    )


class TransductionInput(BaseModel):
    """Input schema for TransductionTool."""
    question: str = Field(..., description="The question to answer")
    start_date: str = Field(..., description="The start date of the data to use for the answer (YYYY-MM-DD)")
    end_date: str = Field(..., description="The end date of the data to use for the answer (YYYY-MM-DD)")
    selected_columns: Optional[List[str]] = Field(
        default=None,
        description="List of column names to include in the analysis. If None, all columns are included. Date column is always included."
    )


class UnifiedTransductionTool(BaseTool):
    name: str = "Advanced Transduction Analysis"
    description: str = (
        "Answers complex financial questions that other tools cannot answer using the agentics framework. "
        "This tool performs deep analysis by reducing large datasets into meaningful insights. "
        "Use this tool when you need comprehensive analysis across a date range that requires "
        "synthesizing information from multiple data points that standard tools cannot handle. "
        "The tool analyzes a comprehensive merged dataset containing macroeconomic indicators, "
        "market factors, DJ30 stock prices, company fundamentals, and news data. "
        "\n\nInput format: question (str), start_date (YYYY-MM-DD), end_date (YYYY-MM-DD)\n"
        "Note: For large date ranges (>100 rows), the tool automatically applies uniform temporal sampling "
        "to analyze ~100 representative data points while preserving time-series patterns."
    )
    args_schema: Type[BaseModel] = TransductionInput

    def _run(self, question: str, start_date: str, end_date: str, selected_columns: Optional[List[str]] = None) -> str:
        try:
            print(f"\n{'='*80}")
            print(f"🔍 Starting Transduction Analysis")
            print(f"{'='*80}")
            print(f"Question: '{question}'")
            print(f"Date range: {start_date} to {end_date}")

            # Use the merged dataset (supports both single file and split files)
            date_column = "Date"
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

            # Import the CSV reader utility
            from utils.csv_reader import get_merged_data_file_for_agentics

            try:
                csv_path = get_merged_data_file_for_agentics(data_dir)
            except FileNotFoundError as e:
                print(f"❌ ERROR: {e}")
                return json.dumps({
                    "success": False,
                    "error": str(e)
                }, indent=2)

            # Increase CSV field size limit to handle large text fields (e.g., Headlines, Earningcall)
            # Default limit is 131072 bytes (128KB), increase to handle large text content
            # Set to a large value (10MB) to accommodate very long text fields
            try:
                csv.field_size_limit(10 * 1024 * 1024)  # 10MB
            except OverflowError:
                # If 10MB is too large for the system, use maximum allowed
                csv.field_size_limit(sys.maxsize)

            # MEMORY OPTIMIZATION: Pre-filter CSV using pandas before loading into Agentics
            # This avoids loading the entire 194MB CSV into memory when we only need a date range
            print(f"\n📂 Loading and filtering merged dataset (memory-optimized)")
            print(f"📅 Date column: {date_column}")
            print(f"📅 Target date range: {start_date} to {end_date}")

            # Try to get memory usage info (if psutil is available)
            try:
                import psutil
                import os as os_module
                process = psutil.Process(os_module.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                print(f"💾 Memory before loading: {mem_before:.1f} MB")
            except ImportError:
                mem_before = None
                print("💾 Memory profiling not available (psutil not installed)")

            # MEMORY OPTIMIZATION: Use pandas to filter CSV before loading into Agentics
            # NOTE: This still loads the entire CSV once, but immediately filters and deletes the full dataframe.
            # This reduces peak memory usage compared to loading everything into Agentics first.
            # For even better memory efficiency, chunked reading could be implemented, but that's more complex.
            print(f"🔍 Filtering CSV by date range before loading (memory optimization)...")
            from utils.csv_reader import read_merged_data_csv

            chunk_df_loaded = False  # Initialize flag

            try:
                # Determine which columns to load
                if selected_columns is not None and len(selected_columns) > 0:
                    # Always include Date column
                    columns_to_load = ["Date"] + [col for col in selected_columns if col != "Date"]
                    print(f"📋 Will load {len(columns_to_load)} columns: {', '.join(columns_to_load[:5])}...")
                else:
                    columns_to_load = None  # Load all columns
                    print(f"📋 Will load all columns")

                # Load CSV and filter in one pass
                # NOTE: This still loads the full CSV, but we filter immediately and delete it
                print(f"📖 Loading CSV and filtering to date range...")
                print(f"⚠️  Note: Loading full CSV to filter (this may use significant memory)...")
                full_df = read_merged_data_csv(data_dir)

                # Convert Date column to datetime for proper comparison
                full_df[date_column] = pd.to_datetime(full_df[date_column], errors='coerce')

                # Filter to date range
                mask = (full_df[date_column] >= pd.to_datetime(start_date)) & (full_df[date_column] <= pd.to_datetime(end_date))
                chunk_df = full_df[mask].copy()

                # Free memory by deleting the full dataframe
                del full_df

                if len(chunk_df) == 0:
                    print(f"❌ ERROR: No data found for date range {start_date} to {end_date}")
                    return json.dumps({
                        "success": False,
                        "error": f"No data found for date range {start_date} to {end_date}"
                    }, indent=2)

                print(f"✅ Found {len(chunk_df):,} rows in date range")

                # Select columns if specified
                if columns_to_load:
                    # Only include columns that actually exist in the DataFrame
                    available_cols = [col for col in columns_to_load if col in chunk_df.columns]
                    if "Date" not in available_cols:
                        available_cols = ["Date"] + [col for col in available_cols if col != "Date"]
                    chunk_df = chunk_df[available_cols]
                    print(f"✅ Filtered to {len(available_cols)} columns")

                print(f"✅ Final dataset: {len(chunk_df):,} rows × {len(chunk_df.columns)} columns")

                # Convert to CSV string and load into Agentics
                # This is more memory-efficient than loading the entire file
                csv_buffer = io.StringIO()
                chunk_df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                csv_buffer.close()

                # Free memory
                del chunk_df
                del csv_buffer

                # Load into Agentics from the CSV string
                # AG.from_csv() can accept a string directly
                agentics = AG.from_csv(csv_string)
                print(f"✅ Agentics object created with {len(agentics.states):,} states")

                # Memory check after loading
                try:
                    if mem_before is not None:
                        mem_after = process.memory_info().rss / 1024 / 1024  # MB
                        mem_used = mem_after - mem_before
                        print(f"💾 Memory after loading: {mem_after:.1f} MB (used: {mem_used:.1f} MB)")
                except:
                    pass

                # Mark that we used the optimized path
                chunk_df_loaded = True

            except MemoryError as e:
                # Handle memory errors specifically
                print(f"❌ MEMORY ERROR: {e}")
                print(f"💡 Suggestion: Try selecting fewer columns or a smaller date range")
                return json.dumps({
                    "success": False,
                    "error": f"Memory error: {str(e)}. Try selecting fewer columns or a smaller date range.",
                    "error_type": "MemoryError"
                }, indent=2)
            except Exception as e:
                # Fallback to original method if optimization fails
                print(f"⚠️  Memory optimization failed: {e}")
                print(f"🔄 Falling back to standard loading method...")
                print(f"Error details: {str(e)}")
                print(f"Error type: {type(e).__name__}")

                try:
                    # Load the full dataset (original method)
                    agentics = AG.from_csv(csv_path)
                    print(f"✅ Loaded {len(agentics.states):,} total rows with {len(agentics.atype.model_fields)} columns")
                    chunk_df_loaded = False
                except MemoryError as e2:
                    print(f"❌ MEMORY ERROR during fallback: {e2}")
                    return json.dumps({
                        "success": False,
                        "error": f"Memory error: {str(e2)}. The dataset is too large for the available memory. Try selecting fewer columns or a smaller date range.",
                        "error_type": "MemoryError"
                    }, indent=2)

            # If we used the optimized path, columns are already filtered and date range is already applied
            # If we used the fallback path, we need to filter columns and dates
            if chunk_df_loaded:
                # We used the optimized path - data is already filtered
                print(f"✅ Data already filtered to date range and selected columns")
                filtered_agentics = agentics
                num_rows = len(filtered_agentics.states)
            else:
                # Fallback path - need to filter columns and dates
                # Filter to selected columns if provided (using Agentics __call__ method)
                if selected_columns is not None and len(selected_columns) > 0:
                    # Always include Date column for filtering, and ensure it comes first
                    # Remove Date from selected_columns if present to avoid duplicates
                    selected_without_date = [col for col in selected_columns if col != "Date"]
                    # Put Date first, then the rest of the selected columns
                    columns_to_include = ["Date"] + selected_without_date
                    print(f"\n🔧 Filtering to {len(columns_to_include)} selected columns: {', '.join(columns_to_include[:5])}...")

                    # Use Agentics __call__ method to filter to only selected columns
                    # This creates a new AG with only the specified fields
                    # The order of arguments determines the order of fields in the model
                    agentics = agentics(*columns_to_include)
                    print(f"✅ Filtered dataset to {len(columns_to_include)} columns (Date first)")
                else:
                    print(f"❌ You must select at least one column to analyze")
                    return json.dumps({
                        "success": False,
                        "error": "You must select at least one column to analyze"
                    })

                # Find the indices that correspond to start_date and end_date
                # Agentics creates states where each state has attributes corresponding to CSV columns
                # We need to find which states have Date values in our range
                print(f"\n🔎 Searching for date range indices using column '{date_column}'...")
                start_index = None
                end_index = None

                for i, state in enumerate(agentics.states):
                    state_date = getattr(state, date_column, None)
                    if state_date:
                        # Compare dates as strings (they're in YYYY-MM-DD format)
                        if start_index is None and state_date >= start_date:
                            start_index = i
                        if state_date <= end_date:
                            end_index = i + 1  # +1 because filter_states uses [start:end)

                if start_index is None or end_index is None:
                    print(f"❌ ERROR: No data found for date range {start_date} to {end_date}")
                    return json.dumps({
                        "success": False,
                        "error": f"No data found for date range {start_date} to {end_date}"
                    })

                # Calculate the number of rows in our date range
                num_rows = end_index - start_index
                print(f"📊 Found date range: rows {start_index} to {end_index} ({num_rows:,} total rows)")

                # Filter the dataset to the date range first
                print(f"\n🔽 Filtering dataset to date range...")
                filtered_agentics = agentics.filter_states(start=start_index, end=end_index)
                print(f"✅ Filtered dataset contains {len(filtered_agentics.states):,} states")

            # Apply uniform sampling if we have too many rows
            TARGET_SAMPLE_SIZE = 100
            actual_rows = len(filtered_agentics.states)

            if actual_rows > TARGET_SAMPLE_SIZE:
                print(f"\n🎲 Dataset exceeds {TARGET_SAMPLE_SIZE} rows. Applying uniform sampling...")
                print(f"   Original rows: {actual_rows:,}")
                sampling_interval = actual_rows / TARGET_SAMPLE_SIZE
                print(f"   Sampling interval: ~1 row every {sampling_interval:.1f} rows")

                filtered_agentics = filtered_agentics.get_uniform_sample(TARGET_SAMPLE_SIZE)
                sampled_rows = len(filtered_agentics.states)
                print(f"✅ Sampled down to {sampled_rows:,} rows (uniform temporal distribution)")
            else:
                sampled_rows = actual_rows
                print(f"✅ No sampling needed ({actual_rows} rows <= {TARGET_SAMPLE_SIZE})")

            # Calculate batch size: aim for ~10 batches
            batch_size = max(2, sampled_rows // 10)  # At least 2 rows per batch
            num_batches = (sampled_rows + batch_size - 1) // batch_size  # Ceiling division
            print(f"📦 Batch configuration: {num_batches} batches of ~{batch_size} rows each")

            # Generate intermediate answer type based on the question
            print(f"\n🤖 Generating dynamic Pydantic model for intermediate analysis...")
            intermediate_answer_ag = AG()
            intermediate_answer_ag = asyncio.run(
                intermediate_answer_ag.generate_atype(question)
            )
            pydantic_class = intermediate_answer_ag.atype
            pydantic_code = intermediate_answer_ag.atype_code

            print(f"\n✨ Generated Pydantic class: {pydantic_class.__name__}")
            print("=" * 80)
            print("DYNAMIC PYDANTIC MODEL CODE:")
            print("=" * 80)
            print(f"{pydantic_code}")
            print("=" * 80)

            # Log the model fields
            if hasattr(pydantic_class, 'model_fields'):
                print(f"📋 Model fields: {list(pydantic_class.model_fields.keys())}")

            # Perform reduction on the filtered (and possibly sampled) dataset
            print(f"\n⚡ Starting batch reduction with {num_batches} batches...")

            # Create AG with areduce_batch_size
            reducer_ag = AG(
                atype=pydantic_class,
                transduction_type="areduce",
                areduce_batch_size=batch_size,
            )

            reduced = asyncio.run(reducer_ag << filtered_agentics)

            # Store transduction flow data for visualization
            # This will be accessed by the Streamlit app to display the flow
            try:
                import streamlit as st
                # Store flow data in session state
                flow_data = {
                    "initial_states": {
                        "agentics": filtered_agentics,
                        "atype_name": filtered_agentics.atype.__name__ if filtered_agentics.atype else "Unknown",
                        "num_rows": len(filtered_agentics.states)
                    },
                    "intermediate_batches": {
                        "batches": getattr(reduced, 'areduce_batches', []),
                        "atype_name": pydantic_class.__name__ if pydantic_class else "Unknown",
                        "num_batches": len(getattr(reduced, 'areduce_batches', []))
                    },
                    "final_intermediate": {
                        "agentics": reduced,
                        "atype_name": reduced.atype.__name__ if reduced.atype else "Unknown",
                        "num_rows": len(reduced.states)
                    }
                }
                st.session_state.transduction_flow = flow_data
            except Exception as e:
                # If streamlit is not available (e.g., during testing), just continue
                pass

            # Add question attribute to the reduced result
            reduced = reduced.add_attribute("question", default_value=question)

            # Display intermediate batch results if available
            # Note: areduce_batches contains the intermediate results from each batch
            # These are the results after reducing each batch of batch_size rows
            if hasattr(reduced, 'areduce_batches') and reduced.areduce_batches:
                print(f"\n 🔽 INTERMEDIATE BATCH RESULTS ({len(reduced.areduce_batches)} batches):")
                for i, batch_result in enumerate(reduced.areduce_batches, 1):
                    print(f"\n--- Batch {i} ---")
                    if hasattr(batch_result, 'model_dump_json'):
                        print(batch_result.model_dump_json(indent=2))
                    else:
                        print(str(batch_result))

            print(f"\n 🔽 FINAL REDUCED RESULT:")
            reduced.pretty_print()

            # Generate final answer
            print(f"\n📝 Generating final comprehensive answer...")
            answer = asyncio.run(
                AG(atype=TransductionAnswer, transduction_type="areduce") << reduced
            )
            print(f"✅ Final answer generation complete")

            # Store final answer in flow data
            try:
                import streamlit as st
                if st.session_state.transduction_flow:
                    st.session_state.transduction_flow["final_answer"] = {
                        "agentics": answer,
                        "atype_name": answer.atype.__name__ if answer.atype else "Unknown",
                        "num_rows": len(answer.states)
                    }
            except Exception:
                pass

            # Extract the answer from the AG object
            final_answer = answer[0] if len(answer) > 0 else None

            if final_answer:
                print("\n" + "=" * 80)
                print("🎉 SUCCESS: Analysis complete!")
                print("=" * 80)
                print(f"📌 Short Answer: {final_answer.short_answer}")
                print("=" * 80 + "\n")

                result = {
                    "success": True,
                    "date_range": {
                        "start": start_date,
                        "end": end_date,
                        "total_rows_in_range": num_rows,
                        "rows_analyzed": sampled_rows,
                        "sampling_applied": sampled_rows < num_rows,
                        "sampling_ratio": f"1:{int(num_rows/sampled_rows)}" if sampled_rows < num_rows else "1:1",
                        "batch_size": batch_size,
                        "num_batches": num_batches
                    },
                    "question": question,
                    "short_answer": final_answer.short_answer,
                    "detailed_answer": final_answer.detailed_answer,
                    "explanation": final_answer.explanation
                }

                print(f"✅ Returning result with {len(result)} fields\n")
                return json.dumps(result, indent=2)
            else:
                print("❌ ERROR: Failed to generate answer - empty result from transduction\n")
                return json.dumps({
                    "success": False,
                    "error": "Failed to generate answer"
                })

        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()

            print("\n" + "=" * 80)
            print("💥 EXCEPTION OCCURRED during Transduction Analysis")
            print("=" * 80)
            print(f"❌ Error: {error_msg}")
            print("\nTraceback:")
            print(error_trace)
            print("=" * 80 + "\n")

            return json.dumps({
                "success": False,
                "error": error_msg,
                "traceback": error_trace
            }, indent=2)