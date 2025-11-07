"""
Tools for querying available data and indicators.
"""

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import pandas as pd
import json
import os
from utils.data_loader import load_macro_factors, load_market_factors, get_column_descriptions
from utils.dj30_data_loader import get_available_tickers as get_dj30_tickers
from utils.firm_data_loader import get_available_tickers as get_firm_tickers


class AvailableIndicatorsInput(BaseModel):
    """Input schema for AvailableIndicatorsTool."""
    category: str = Field(
        default="all",
        description="Category to filter by: 'all', 'macro', 'market', 'dj30', 'fundamentals', or 'stocks'"
    )


class AvailableIndicatorsTool(BaseTool):
    name: str = "List Available Indicators"
    description: str = (
        "Lists all available indicators, columns, and data fields in the merged dataset. "
        "Use this tool to discover what indicators are available before using visualization tools. "
        "Returns indicators organized by category: macroeconomic indicators, market factors, "
        "DJ30 stock prices, and company fundamentals. "
        "This is essential when you need to know valid indicator names for tools like "
        "ComparativePerformanceTool, TimeSeriesPlotTool, or MultiIndicatorPlotTool."
    )
    args_schema: Type[BaseModel] = AvailableIndicatorsInput

    def _run(self, category: str = "all") -> str:
        try:
            result = {
                "success": True,
                "category": category,
                "indicators": {}
            }

            # Load data to get actual column names
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            # Get column descriptions
            descriptions = get_column_descriptions()

            # Macroeconomic indicators
            if category in ["all", "macro"]:
                macro_indicators = {
                    "indicators": sorted(macro_df.columns.tolist()),
                    "count": len(macro_df.columns),
                    "descriptions": descriptions.get("macro_factors", {})
                }
                result["indicators"]["macroeconomic"] = macro_indicators

            # Market factors
            if category in ["all", "market"]:
                market_indicators = {
                    "indicators": sorted(market_df.columns.tolist()),
                    "count": len(market_df.columns),
                    "descriptions": descriptions.get("market_factors", {})
                }
                result["indicators"]["market_factors"] = market_indicators

            # DJ30 Stock Tickers
            if category in ["all", "dj30", "stocks"]:
                dj30_tickers = get_dj30_tickers()
                dj30_indicators = {
                    "tickers": sorted(dj30_tickers),
                    "count": len(dj30_tickers),
                    "note": "For DJ30 stocks, use ticker symbols (e.g., 'AAPL', 'MSFT') with price columns like 'open_AAPL', 'close_MSFT', etc."
                }
                result["indicators"]["dj30_stocks"] = dj30_indicators

            # Company Fundamentals
            if category in ["all", "fundamentals"]:
                firm_tickers = get_firm_tickers()
                fundamental_metrics = [
                    "EPS", "DPS", "ROA", "ROE", "NAV", "GRM", "EBS",
                    "BPS", "CPS", "SAL", "NET"
                ]
                fundamentals_info = {
                    "tickers": sorted(firm_tickers),
                    "metrics": fundamental_metrics,
                    "count_tickers": len(firm_tickers),
                    "count_metrics": len(fundamental_metrics),
                    "note": "Fundamental metrics are available per ticker. Format: 'EPS_AAPL_MEDEST', 'ROE_MSFT_ACTUAL', etc."
                }
                result["indicators"]["company_fundamentals"] = fundamentals_info

            # Create a summary message
            summary_parts = []
            if "macroeconomic" in result["indicators"]:
                summary_parts.append(f"{result['indicators']['macroeconomic']['count']} macroeconomic indicators")
            if "market_factors" in result["indicators"]:
                summary_parts.append(f"{result['indicators']['market_factors']['count']} market factors")
            if "dj30_stocks" in result["indicators"]:
                summary_parts.append(f"{result['indicators']['dj30_stocks']['count']} DJ30 stock tickers")
            if "company_fundamentals" in result["indicators"]:
                summary_parts.append(f"{result['indicators']['company_fundamentals']['count_tickers']} companies with fundamentals")

            result["summary"] = f"Found: {', '.join(summary_parts)}"

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

