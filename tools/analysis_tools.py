"""
Tools for financial data analysis (volatility, correlation, etc.).
"""

from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
from utils.data_loader import load_macro_factors, load_market_factors


class VolatilityAnalysisInput(BaseModel):
    """Input schema for VolatilityAnalysisTool."""
    indicator: str = Field(..., description="Indicator name to analyze volatility (e.g., '^GSPC', '^VIX')")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    window: int = Field(default=30, description="Rolling window size for volatility calculation (default: 30 days)")


class VolatilityAnalysisTool(BaseTool):
    name: str = "Analyze Volatility"
    description: str = (
        "Analyzes volatility of a specific indicator over time. "
        "Calculates rolling standard deviation and identifies periods of high volatility. "
        "Returns volatility statistics and dates with largest volatility spikes."
    )
    args_schema: Type[BaseModel] = VolatilityAnalysisInput

    def _run(self, indicator: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None, window: int = 30) -> str:
        try:
            # Find indicator in datasets
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            if indicator in macro_df.columns:
                df = macro_df
            elif indicator in market_df.columns:
                df = market_df
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Indicator '{indicator}' not found"
                })

            # Filter by date range
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

            # Calculate returns (percentage change)
            returns = series.pct_change().dropna()

            # Calculate rolling volatility (standard deviation of returns)
            rolling_vol = returns.rolling(window=window).std()

            # Find top volatility spikes
            top_vol_dates = rolling_vol.nlargest(10)

            # Calculate overall volatility stats
            volatility_stats = {
                "indicator": indicator,
                "window": window,
                "overall_volatility": float(returns.std()),
                "mean_rolling_volatility": float(rolling_vol.mean()),
                "max_rolling_volatility": float(rolling_vol.max()),
                "min_rolling_volatility": float(rolling_vol.min())
            }

            # Format top volatility dates
            top_vol_list = [
                {
                    "date": str(date),
                    "volatility": float(vol),
                    "value": float(series[date])
                }
                for date, vol in top_vol_dates.items()
            ]

            return json.dumps({
                "success": True,
                "statistics": volatility_stats,
                "top_volatility_periods": top_vol_list
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class CorrelationAnalysisInput(BaseModel):
    """Input schema for CorrelationAnalysisTool."""
    indicators: str = Field(..., description="Comma-separated list of indicators to analyze correlation (e.g., '^GSPC,^VIX,FEDFUNDS')")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")


class CorrelationAnalysisTool(BaseTool):
    name: str = "Analyze Correlation"
    description: str = (
        "Calculates correlation coefficients between multiple indicators. "
        "Use this to understand relationships between different economic and market factors."
    )
    args_schema: Type[BaseModel] = CorrelationAnalysisInput

    def _run(self, indicators: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> str:
        try:
            # Parse indicators
            indicator_list = [ind.strip() for ind in indicators.split(',')]

            if len(indicator_list) < 2:
                return json.dumps({
                    "success": False,
                    "error": "Need at least 2 indicators for correlation analysis"
                })

            # Load datasets
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            # Combine both datasets
            combined_df = pd.concat([macro_df, market_df], axis=1)

            # Filter by date range
            if start_date and end_date:
                combined_df = combined_df.loc[start_date:end_date]
            elif start_date:
                combined_df = combined_df.loc[start_date:]
            elif end_date:
                combined_df = combined_df.loc[:end_date]

            # Select only requested indicators
            available_indicators = [ind for ind in indicator_list if ind in combined_df.columns]

            if len(available_indicators) < 2:
                return json.dumps({
                    "success": False,
                    "error": f"Not enough valid indicators. Found: {available_indicators}"
                })

            # Calculate correlation matrix
            correlation_matrix = combined_df[available_indicators].corr()

            # Convert to dict
            corr_dict = correlation_matrix.to_dict()

            # Format as nested structure
            formatted_corr = {
                ind1: {ind2: float(val) for ind2, val in corr_dict[ind1].items()}
                for ind1 in available_indicators
            }

            # Find strongest correlations (excluding self-correlation)
            strong_correlations = []
            for i, ind1 in enumerate(available_indicators):
                for ind2 in available_indicators[i+1:]:
                    corr_val = correlation_matrix.loc[ind1, ind2]
                    strong_correlations.append({
                        "indicator1": ind1,
                        "indicator2": ind2,
                        "correlation": float(corr_val)
                    })

            # Sort by absolute correlation
            strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            return json.dumps({
                "success": True,
                "correlation_matrix": formatted_corr,
                "top_correlations": strong_correlations[:10]
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class FindExtremeValuesInput(BaseModel):
    """Input schema for FindExtremeValuesTool."""
    indicator: str = Field(..., description="Indicator name to analyze")
    n: int = Field(default=10, description="Number of extreme values to return")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")


class FindExtremeValuesTool(BaseTool):
    name: str = "Find Extreme Values"
    description: str = (
        "Finds the highest and lowest values for a specific indicator. "
        "Useful for identifying market crashes, peaks, or extreme economic conditions."
    )
    args_schema: Type[BaseModel] = FindExtremeValuesInput

    def _run(self, indicator: str, n: int = 10, start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> str:
        try:
            # Find indicator in datasets
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            if indicator in macro_df.columns:
                df = macro_df
            elif indicator in market_df.columns:
                df = market_df
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Indicator '{indicator}' not found"
                })

            # Filter by date range
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

            # Find highest and lowest values
            highest = series.nlargest(n)
            lowest = series.nsmallest(n)

            highest_list = [
                {"date": str(date), "value": float(val)}
                for date, val in highest.items()
            ]

            lowest_list = [
                {"date": str(date), "value": float(val)}
                for date, val in lowest.items()
            ]

            return json.dumps({
                "success": True,
                "indicator": indicator,
                "highest_values": highest_list,
                "lowest_values": lowest_list
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })

