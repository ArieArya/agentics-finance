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
import logging
from datetime import datetime

# Configure logging for this tool
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
    start_date: str = Field(..., description="The start date of the data to use for the answer (YYYY-MM-DD)")
    end_date: str = Field(..., description="The end date of the data to use for the answer (YYYY-MM-DD)")


class MacroTransductionTool(BaseTool):
    name: str = "Macro Transduction Analysis"
    description: str = (
        "Answers complex macroeconomics questions that other tools cannot answer using the agentics framework. "
        "This tool performs deep analysis by reducing large datasets into meaningful insights. "
        "Use this tool when you need comprehensive analysis across a date range that requires "
        "synthesizing information from multiple data points. "
        "Input should include: question (str), start_date (YYYY-MM-DD), end_date (YYYY-MM-DD)."
    )
    args_schema: Type[BaseModel] = TransductionInput

    def _run(self, question: str, start_date: str, end_date: str) -> str:
        try:
            logger.info(f"Starting Macro Transduction Analysis for question: '{question}'")
            logger.info(f"Date range: {start_date} to {end_date}")

            # Get the absolute path to the CSV file
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "macro_factors_new.csv")

            # Load the full dataset using Agentics
            logger.info(f"Loading macro factors dataset from {csv_path}")
            agentics = AG.from_csv(csv_path)
            logger.info(f"Loaded {len(agentics.states)} total rows from dataset")

            # Find the indices that correspond to start_date and end_date
            # Agentics creates states where each state has attributes corresponding to CSV columns
            # We need to find which states have Date values in our range
            logger.info("Searching for date range indices...")
            start_index = None
            end_index = None

            for i, state in enumerate(agentics.states):
                state_date = getattr(state, 'Date', None)
                if state_date:
                    # Compare dates as strings (they're in YYYY-MM-DD format)
                    if start_index is None and state_date >= start_date:
                        start_index = i
                    if state_date <= end_date:
                        end_index = i + 1  # +1 because filter_states uses [start:end)

            if start_index is None or end_index is None:
                logger.error(f"No data found for date range {start_date} to {end_date}")
                return json.dumps({
                    "success": False,
                    "error": f"No data found for date range {start_date} to {end_date}"
                })

            # Calculate the number of rows in our date range
            num_rows = end_index - start_index
            logger.info(f"Found date range: rows {start_index} to {end_index} ({num_rows} total rows)")

            # Check if number of rows exceeds the limit
            MAX_ROWS_LIMIT = 500
            if num_rows > MAX_ROWS_LIMIT:
                logger.warning(f"Selected number of rows ({num_rows}) exceeds limit of {MAX_ROWS_LIMIT}")
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
            logger.info(f"Batch configuration: {num_batches} batches of ~{batch_size} rows each")

            # Generate intermediate answer type based on the question
            logger.info("Generating dynamic Pydantic model for intermediate analysis...")
            intermediate_answer_ag = AG()
            intermediate_answer_ag = asyncio.run(
                intermediate_answer_ag.generate_atype(question)
            )
            pydantic_class = intermediate_answer_ag.atype
            pydantic_code = intermediate_answer_ag.atype_code

            logger.info(f"Generated Pydantic class: {pydantic_class.__name__}")
            logger.info("=" * 80)
            logger.info("DYNAMIC PYDANTIC MODEL CODE:")
            logger.info("=" * 80)
            logger.info(f"\n{pydantic_code}\n")
            logger.info("=" * 80)

            # Log the model fields
            if hasattr(pydantic_class, 'model_fields'):
                logger.info(f"Model fields: {list(pydantic_class.model_fields.keys())}")

            # Filter the dataset to the date range
            logger.info(f"Filtering dataset to date range...")
            filtered_agentics = agentics.filter_states(start=start_index, end=end_index)
            logger.info(f"Filtered dataset contains {len(filtered_agentics.states)} states")

            # Perform reduction on the filtered dataset
            logger.info(f"Starting batch reduction with {num_batches} batches...")
            reduced = asyncio.run(
                AG(
                    atype=pydantic_class,
                    transduction_type="areduce",
                    areduce_batch_size=batch_size,
                )
                << filtered_agentics
            )
            logger.info(f"Batch reduction complete. Generated {len(reduced.states)} intermediate results")
            reduced = reduced.add_attribute("question", default_value=question)

            # Generate final answer
            logger.info("Generating final comprehensive answer...")
            answer = asyncio.run(
                AG(atype=TransductionAnswer, transduction_type="areduce") << reduced
            )
            logger.info("Final answer generation complete")

            # Extract the answer from the AG object
            final_answer = answer[0] if len(answer) > 0 else None

            if final_answer:
                logger.info("=" * 80)
                logger.info("SUCCESS: Analysis complete!")
                logger.info(f"Short Answer: {final_answer.short_answer}")
                logger.info("=" * 80)

                result = {
                    "success": True,
                    "date_range": {
                        "start": start_date,
                        "end": end_date,
                        "rows_analyzed": num_rows,
                        "batch_size": batch_size,
                        "num_batches": num_batches
                    },
                    "question": question,
                    "generated_model": pydantic_class.__name__,
                    "model_code": pydantic_code,
                    "short_answer": final_answer.short_answer,
                    "answer_report": final_answer.answer_report
                }

                logger.info(f"Returning result with {len(result)} fields")
                return json.dumps(result, indent=2)
            else:
                logger.error("Failed to generate answer - empty result from transduction")
                return json.dumps({
                    "success": False,
                    "error": "Failed to generate answer"
                })

        except Exception as e:
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()

            logger.error("=" * 80)
            logger.error("EXCEPTION OCCURRED during Macro Transduction Analysis")
            logger.error(f"Error: {error_msg}")
            logger.error("Traceback:")
            logger.error(error_trace)
            logger.error("=" * 80)

            return json.dumps({
                "success": False,
                "error": error_msg,
                "traceback": error_trace
            }, indent=2)