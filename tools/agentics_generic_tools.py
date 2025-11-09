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

# Module-level variable to store selected columns from UI
# This is set by the Streamlit app before running analysis
_selected_columns_from_ui: Optional[List[str]] = None


def set_selected_columns(columns: Optional[List[str]]):
    """Set the selected columns from the UI. Called by Streamlit app before running analysis."""
    global _selected_columns_from_ui
    _selected_columns_from_ui = columns


def get_selected_columns() -> Optional[List[str]]:
    """Get the selected columns from the UI. Called by the tool during execution."""
    return _selected_columns_from_ui


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

    def _run(self, question: str, start_date: str, end_date: str) -> str:
        try:
            # Get selected columns from UI (set deterministically by user selection)
            selected_columns = get_selected_columns()

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

            # Load the full dataset first
            print(f"\n📂 Loading merged dataset")
            print(f"📅 Date column: {date_column}")
            agentics = AG.from_csv(csv_path)
            print(f"✅ Loaded {len(agentics.states):,} total rows with {len(agentics.atype.model_fields)} columns")

            # Filter to selected columns if provided (using Agentics __call__ method)
            if selected_columns is not None and len(selected_columns) > 0:
                # Agentics sanitizes column names when creating Pydantic models
                # (removes special characters like ^, -, etc.)
                # We need to sanitize the column names to match what's in the model
                import re
                def sanitize_field_name(name: str) -> str:
                    """Sanitize field name to match Agentics' sanitization."""
                    name = name.strip()
                    # Remove underscores only from the start
                    name = re.sub(r"^_+", "", name)
                    # If the result is alphanumeric, return as-is
                    if re.fullmatch(r"[a-zA-Z0-9_]+", name):
                        return name
                    # Otherwise, remove all non-alphanumeric and non-underscore characters
                    return re.sub(r"[^\w]", "", name)

                # Sanitize all selected columns to match Agentics model field names
                sanitized_selected = [sanitize_field_name(col) for col in selected_columns]

                # Always include Date column for filtering, and ensure it comes first
                # Remove Date from sanitized_selected if present to avoid duplicates
                selected_without_date = [col for col in sanitized_selected if col != "Date"]
                # Put Date first, then the rest of the selected columns
                columns_to_include = ["Date"] + selected_without_date
                print(f"\n🔧 Filtering to {len(columns_to_include)} selected columns: {', '.join(columns_to_include[:5])}...")
                print(f"   Original columns: {selected_columns[:5]}...")
                print(f"   Sanitized columns: {columns_to_include[:5]}...")

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