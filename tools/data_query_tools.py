"""
Tools for querying financial data.
"""

from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import pandas as pd
import json
from utils.data_loader import load_macro_factors, load_market_factors, get_column_descriptions


class DateRangeQueryInput(BaseModel):
    """Input schema for DateRangeQueryTool."""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    indicators: str = Field(..., description="Comma-separated list of indicator names (e.g., 'FEDFUNDS,^GSPC,^VIX')")
    dataset: str = Field(default="both", description="Dataset to query: 'macro', 'market', or 'both'")


class DateRangeQueryTool(BaseTool):
    name: str = "Query Data by Date Range"
    description: str = (
        "Retrieves financial data for specific indicators within a date range. "
        "Use this tool to fetch historical data for analysis. "
        "Supports both macro factors (FEDFUNDS, CPIAUCSL, UNRATE, etc.) "
        "and market factors (^GSPC, ^VIX, BTC-USD, etc.)."
    )
    args_schema: Type[BaseModel] = DateRangeQueryInput

    def _run(self, start_date: str, end_date: str, indicators: str, dataset: str = "both") -> str:
        try:
            # Parse indicators
            indicator_list = [ind.strip() for ind in indicators.split(',')]

            # Load appropriate dataset(s)
            result_data = {}

            if dataset in ["macro", "both"]:
                macro_df = load_macro_factors()
                macro_indicators = [ind for ind in indicator_list if ind in macro_df.columns]
                if macro_indicators:
                    filtered = macro_df.loc[start_date:end_date, macro_indicators]
                    result_data['macro'] = filtered.to_dict('index')

            if dataset in ["market", "both"]:
                market_df = load_market_factors()
                market_indicators = [ind for ind in indicator_list if ind in market_df.columns]
                if market_indicators:
                    filtered = market_df.loc[start_date:end_date, market_indicators]
                    result_data['market'] = filtered.to_dict('index')

            # Convert dates to strings for JSON serialization
            formatted_result = {}
            for dataset_name, data in result_data.items():
                formatted_result[dataset_name] = {
                    str(date): values for date, values in data.items()
                }

            return json.dumps({
                "success": True,
                "date_range": {"start": start_date, "end": end_date},
                "indicators": indicator_list,
                "data": formatted_result
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class IndicatorStatsInput(BaseModel):
    """Input schema for IndicatorStatsTool."""
    indicator: str = Field(..., description="Indicator name (e.g., 'FEDFUNDS', '^GSPC', '^VIX')")
    start_date: Optional[str] = Field(None, description="Optional start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="Optional end date in YYYY-MM-DD format")


class IndicatorStatsTool(BaseTool):
    name: str = "Get Indicator Statistics"
    description: str = (
        "Calculates statistical summary for a specific indicator including mean, median, "
        "std deviation, min, max, and percentiles. Optionally filter by date range."
    )
    args_schema: Type[BaseModel] = IndicatorStatsInput

    def _run(self, indicator: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        try:
            # Try to find indicator in macro or market dataset
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            if indicator in macro_df.columns:
                df = macro_df
                dataset = "macro_factors"
            elif indicator in market_df.columns:
                df = market_df
                dataset = "market_factors"
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Indicator '{indicator}' not found in any dataset"
                })

            # Filter by date range if provided
            if start_date and end_date:
                series = df.loc[start_date:end_date, indicator]
            elif start_date:
                series = df.loc[start_date:, indicator]
            elif end_date:
                series = df.loc[:end_date, indicator]
            else:
                series = df[indicator]

            # Drop NaN values
            series = series.dropna()

            # Calculate statistics
            stats = {
                "indicator": indicator,
                "dataset": dataset,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75))
            }

            if start_date or end_date:
                stats["date_range"] = {
                    "start": start_date or "beginning",
                    "end": end_date or "end"
                }

            return json.dumps({
                "success": True,
                "statistics": stats
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class AvailableIndicatorsInput(BaseModel):
    """Input schema for AvailableIndicatorsTool."""
    pass


class AvailableIndicatorsTool(BaseTool):
    name: str = "List Available Indicators"
    description: str = (
        "Lists all available indicators in the dataset with their descriptions. "
        "Use this tool to discover what data is available for analysis."
    )
    args_schema: Type[BaseModel] = AvailableIndicatorsInput

    def _run(self) -> str:
        try:
            descriptions = get_column_descriptions()
            return json.dumps({
                "success": True,
                "indicators": descriptions
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })

