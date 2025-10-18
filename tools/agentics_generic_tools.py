"""
Tools using agentics framework
"""

from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
from agentics import AG
import pandas as pd
import json
import asyncio
import os
from datetime import datetime


class TransductionAnswer(BaseModel):
    short_answer: str | None = None
    answer_report: str | None = Field(
        None,
        description="""
A detailed Markdown Document reporting evidence for the above answer
""",
    )


class TransductionInput(BaseModel):
    """Input schema for TransductionTool."""
    question: str = Field(..., description="The question to answer")
    dataset: str = Field(
        ...,
        description="Dataset to analyze. Must be one of: 'macro' (macroeconomic indicators), 'market' (market data & indices), 'dj30' (DJ30 stock prices)"
    )
    start_date: str = Field(..., description="The start date of the data to use for the answer (YYYY-MM-DD)")
    end_date: str = Field(..., description="The end date of the data to use for the answer (YYYY-MM-DD)")


class UnifiedTransductionTool(BaseTool):
    name: str = "Advanced Transduction Analysis"
    description: str = (
        "Answers complex financial questions that other tools cannot answer using the agentics framework. "
        "This tool performs deep analysis by reducing large datasets into meaningful insights. "
        "Use this tool when you need comprehensive analysis across a date range that requires "
        "synthesizing information from multiple data points that standard tools cannot handle. "
        "\n\nSupported datasets:\n"
        "- 'macro': Macroeconomic indicators (FEDFUNDS, CPI, UNRATE, etc.)\n"
        "- 'market': Market data & indices (S&P 500, VIX, BTC, Gold, etc.)\n"
        "- 'dj30': DJ30 stock prices (OHLCV data for 30 Dow Jones companies)\n"
        "\n\nInput format: question (str), dataset (str), start_date (YYYY-MM-DD), end_date (YYYY-MM-DD)\n"
        "Note: Limited to 500 rows per analysis."
    )
    args_schema: Type[BaseModel] = TransductionInput

    def _run(self, question: str, dataset: str, start_date: str, end_date: str) -> str:
        try:
            print(f"\n{'='*80}")
            print(f"🔍 Starting Transduction Analysis")
            print(f"{'='*80}")
            print(f"Question: '{question}'")
            print(f"Dataset: {dataset}")
            print(f"Date range: {start_date} to {end_date}")

            # Map dataset names to CSV files and their date columns
            dataset_mapping = {
                "macro": {"file": "macro_factors_new.csv"},
                "market": {"file": "market_factors_new.csv"},
                "dj30": {"file": "dj30_data_full.csv"},
            }

            # Validate dataset parameter
            if dataset not in dataset_mapping:
                print(f"❌ ERROR: Invalid dataset: {dataset}")
                return json.dumps({
                    "success": False,
                    "error": f"Invalid dataset '{dataset}'. Must be one of: {list(dataset_mapping.keys())}"
                }, indent=2)

            # Get the absolute path to the CSV file and date column name
            csv_filename = dataset_mapping[dataset]["file"]
            date_column = "Date"
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", csv_filename)

            if not os.path.exists(csv_path):
                print(f"❌ ERROR: Dataset file not found: {csv_path}")
                return json.dumps({
                    "success": False,
                    "error": f"Dataset file not found: {csv_filename}"
                }, indent=2)

            # Load the full dataset using Agentics
            print(f"\n📂 Loading {dataset} dataset from {csv_filename}")
            print(f"📅 Date column: {date_column}")
            agentics = AG.from_csv(csv_path)
            print(f"✅ Loaded {len(agentics.states):,} total rows from {dataset} dataset")

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

            # Check if number of rows exceeds the limit
            MAX_ROWS_LIMIT = 500
            if num_rows > MAX_ROWS_LIMIT:
                print(f"⚠️  WARNING: Selected number of rows ({num_rows:,}) exceeds limit of {MAX_ROWS_LIMIT:,}")
                return json.dumps({
                    "success": False,
                    "error": f"Selected number of rows ({num_rows}) exceeds the limit for transduction ({MAX_ROWS_LIMIT} rows). Please select a smaller date range.",
                    "rows_requested": num_rows,
                    "max_allowed": MAX_ROWS_LIMIT,
                    "suggestion": f"Try reducing the date range to analyze approximately {MAX_ROWS_LIMIT} days or less."
                }, indent=2)

            # Calculate batch size: aim for ~10 batches
            batch_size = max(2, num_rows // 10)  # At least 2 rows per batch
            num_batches = (num_rows + batch_size - 1) // batch_size  # Ceiling division
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

            # Filter the dataset to the date range
            print(f"\n🔽 Filtering dataset to date range...")
            filtered_agentics = agentics.filter_states(start=start_index, end=end_index)
            print(f"✅ Filtered dataset contains {len(filtered_agentics.states):,} states")

            # Perform reduction on the filtered dataset
            print(f"\n⚡ Starting batch reduction with {num_batches} batches...")
            reduced = asyncio.run(
                AG(
                    atype=pydantic_class,
                    transduction_type="areduce",
                    areduce_batch_size=batch_size,
                )
                << filtered_agentics
            )
            print(f"✅ Batch reduction complete. Generated {len(reduced.states):,} intermediate results")
            reduced = reduced.add_attribute("question", default_value=question)

            # Generate final answer
            print(f"\n📝 Generating final comprehensive answer...")
            answer = asyncio.run(
                AG(atype=TransductionAnswer, transduction_type="areduce") << reduced
            )
            print(f"✅ Final answer generation complete")

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
                    "dataset": dataset,
                    "date_range": {
                        "start": start_date,
                        "end": end_date,
                        "rows_analyzed": num_rows,
                        "batch_size": batch_size,
                        "num_batches": num_batches
                    },
                    "question": question,
                    "short_answer": final_answer.short_answer,
                    "answer_report": final_answer.answer_report
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
            import traceback
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