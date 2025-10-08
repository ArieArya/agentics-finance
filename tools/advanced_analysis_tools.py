"""
Advanced analysis tools for financial data.
Includes returns, drawdowns, moving averages, and comparative analysis.
"""

from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
from utils.data_loader import load_macro_factors, load_market_factors


class ReturnsAnalysisInput(BaseModel):
    """Input schema for ReturnsAnalysisTool."""
    indicator: str = Field(..., description="Indicator name to analyze returns")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    period: str = Field(default="daily", description="Return period: 'daily', 'weekly', 'monthly', 'yearly'")


class ReturnsAnalysisTool(BaseTool):
    name: str = "Calculate Returns"
    description: str = (
        "Calculates returns for an indicator over a specified period. "
        "Returns can be daily, weekly, monthly, or yearly. "
        "Useful for analyzing performance and comparing different time periods."
    )
    args_schema: Type[BaseModel] = ReturnsAnalysisInput

    def _run(self, indicator: str, start_date: str, end_date: str, period: str = "daily") -> str:
        try:
            # Load data
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            if indicator in macro_df.columns:
                df = macro_df
            elif indicator in market_df.columns:
                df = market_df
            else:
                return json.dumps({"success": False, "error": f"Indicator '{indicator}' not found"})

            # Filter by date range
            series = df.loc[start_date:end_date, indicator].dropna()

            # Calculate returns based on period
            if period == "daily":
                returns = series.pct_change().dropna()
            elif period == "weekly":
                returns = series.resample('W').last().pct_change().dropna()
            elif period == "monthly":
                returns = series.resample('M').last().pct_change().dropna()
            elif period == "yearly":
                returns = series.resample('Y').last().pct_change().dropna()
            else:
                return json.dumps({"success": False, "error": f"Invalid period: {period}"})

            # Calculate statistics
            total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100
            annualized_return = ((1 + total_return/100) ** (252 / len(returns)) - 1) * 100 if period == "daily" else None

            stats = {
                "indicator": indicator,
                "period": period,
                "total_return_pct": float(total_return),
                "mean_return_pct": float(returns.mean() * 100),
                "volatility_pct": float(returns.std() * 100),
                "best_return_pct": float(returns.max() * 100),
                "worst_return_pct": float(returns.min() * 100),
                "positive_periods": int((returns > 0).sum()),
                "negative_periods": int((returns < 0).sum()),
                "start_value": float(series.iloc[0]),
                "end_value": float(series.iloc[-1])
            }

            if annualized_return:
                stats["annualized_return_pct"] = float(annualized_return)

            return json.dumps({"success": True, "statistics": stats}, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class DrawdownAnalysisInput(BaseModel):
    """Input schema for DrawdownAnalysisTool."""
    indicator: str = Field(..., description="Indicator name to analyze drawdowns")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")


class DrawdownAnalysisTool(BaseTool):
    name: str = "Analyze Drawdowns"
    description: str = (
        "Analyzes maximum drawdown (peak-to-trough decline) for an indicator. "
        "Identifies the worst drawdown periods and recovery times. "
        "Critical for risk assessment and understanding worst-case scenarios."
    )
    args_schema: Type[BaseModel] = DrawdownAnalysisInput

    def _run(self, indicator: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        try:
            # Load data
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            if indicator in macro_df.columns:
                df = macro_df
            elif indicator in market_df.columns:
                df = market_df
            else:
                return json.dumps({"success": False, "error": f"Indicator '{indicator}' not found"})

            # Filter by date range
            if start_date and end_date:
                series = df.loc[start_date:end_date, indicator]
            elif start_date:
                series = df.loc[start_date:, indicator]
            elif end_date:
                series = df.loc[:end_date, indicator]
            else:
                series = df[indicator]

            series = series.dropna()

            # Calculate running maximum
            running_max = series.expanding().max()

            # Calculate drawdown
            drawdown = (series - running_max) / running_max * 100

            # Find maximum drawdown
            max_dd = drawdown.min()
            max_dd_date = drawdown.idxmin()

            # Find the peak before max drawdown
            peak_date = running_max.loc[:max_dd_date].idxmax()
            peak_value = series.loc[peak_date]
            trough_value = series.loc[max_dd_date]

            # Find recovery date (if recovered)
            recovery_date = None
            recovery_days = None
            if len(series.loc[max_dd_date:]) > 1:
                recovered = series.loc[max_dd_date:] >= peak_value
                if recovered.any():
                    recovery_date = series.loc[max_dd_date:][recovered].index[0]
                    recovery_days = (recovery_date - max_dd_date).days

            # Find top 5 drawdowns
            drawdown_sorted = drawdown.nsmallest(10)
            top_drawdowns = [
                {
                    "date": str(date),
                    "drawdown_pct": float(dd),
                    "value": float(series[date])
                }
                for date, dd in drawdown_sorted.items()
            ]

            result = {
                "indicator": indicator,
                "max_drawdown_pct": float(max_dd),
                "peak_date": str(peak_date),
                "peak_value": float(peak_value),
                "trough_date": str(max_dd_date),
                "trough_value": float(trough_value),
                "decline_days": (max_dd_date - peak_date).days,
                "recovery_date": str(recovery_date) if recovery_date else None,
                "recovery_days": recovery_days,
                "top_drawdowns": top_drawdowns[:5]
            }

            return json.dumps({"success": True, "analysis": result}, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class MovingAverageInput(BaseModel):
    """Input schema for MovingAverageTool."""
    indicator: str = Field(..., description="Indicator name")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    windows: str = Field(..., description="Comma-separated list of MA windows (e.g., '20,50,200')")


class MovingAverageTool(BaseTool):
    name: str = "Calculate Moving Averages"
    description: str = (
        "Calculates simple moving averages (SMA) for an indicator. "
        "Common windows: 20-day (monthly), 50-day (quarterly), 200-day (yearly). "
        "Useful for trend identification and technical analysis."
    )
    args_schema: Type[BaseModel] = MovingAverageInput

    def _run(self, indicator: str, start_date: str, end_date: str, windows: str) -> str:
        try:
            # Parse windows
            window_list = [int(w.strip()) for w in windows.split(',')]

            # Load data
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            if indicator in macro_df.columns:
                df = macro_df
            elif indicator in market_df.columns:
                df = market_df
            else:
                return json.dumps({"success": False, "error": f"Indicator '{indicator}' not found"})

            # Filter by date range
            series = df.loc[start_date:end_date, indicator].dropna()

            # Calculate moving averages
            ma_data = {}
            for window in window_list:
                ma = series.rolling(window=window).mean()
                ma_data[f"MA_{window}"] = {
                    "current": float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else None,
                    "mean": float(ma.mean()),
                    "min": float(ma.min()),
                    "max": float(ma.max())
                }

            # Identify crossovers (if multiple MAs)
            crossovers = []
            if len(window_list) >= 2:
                short_ma = series.rolling(window=min(window_list)).mean()
                long_ma = series.rolling(window=max(window_list)).mean()

                # Find where short crosses above long
                crosses_above = ((short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)))
                crosses_below = ((short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1)))

                for date in crosses_above[crosses_above].index[-5:]:
                    crossovers.append({
                        "date": str(date),
                        "type": "golden_cross",
                        "price": float(series[date])
                    })

                for date in crosses_below[crosses_below].index[-5:]:
                    crossovers.append({
                        "date": str(date),
                        "type": "death_cross",
                        "price": float(series[date])
                    })

            result = {
                "indicator": indicator,
                "current_price": float(series.iloc[-1]),
                "moving_averages": ma_data,
                "recent_crossovers": sorted(crossovers, key=lambda x: x['date'], reverse=True)[:5]
            }

            return json.dumps({"success": True, "analysis": result}, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class PercentageChangeInput(BaseModel):
    """Input schema for PercentageChangeTool."""
    indicator: str = Field(..., description="Indicator name")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")


class PercentageChangeTool(BaseTool):
    name: str = "Calculate Percentage Change"
    description: str = (
        "Calculates percentage change for an indicator over a specific period. "
        "Also provides interim milestones (25%, 50%, 75% through the period). "
        "Useful for understanding growth rates and comparing performance."
    )
    args_schema: Type[BaseModel] = PercentageChangeInput

    def _run(self, indicator: str, start_date: str, end_date: str) -> str:
        try:
            # Load data
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            if indicator in macro_df.columns:
                df = macro_df
            elif indicator in market_df.columns:
                df = market_df
            else:
                return json.dumps({"success": False, "error": f"Indicator '{indicator}' not found"})

            # Filter by date range
            series = df.loc[start_date:end_date, indicator].dropna()

            if len(series) < 2:
                return json.dumps({"success": False, "error": "Not enough data points"})

            # Calculate total change
            start_value = series.iloc[0]
            end_value = series.iloc[-1]
            total_change_pct = ((end_value - start_value) / start_value) * 100

            # Calculate interim milestones
            milestones = []
            for pct in [0.25, 0.5, 0.75]:
                idx = int(len(series) * pct)
                milestone_date = series.index[idx]
                milestone_value = series.iloc[idx]
                change_from_start = ((milestone_value - start_value) / start_value) * 100

                milestones.append({
                    "progress": f"{int(pct*100)}%",
                    "date": str(milestone_date),
                    "value": float(milestone_value),
                    "change_from_start_pct": float(change_from_start)
                })

            # Find max and min during period
            max_value = series.max()
            max_date = series.idxmax()
            min_value = series.min()
            min_date = series.idxmin()

            result = {
                "indicator": indicator,
                "start_date": start_date,
                "start_value": float(start_value),
                "end_date": end_date,
                "end_value": float(end_value),
                "total_change_pct": float(total_change_pct),
                "absolute_change": float(end_value - start_value),
                "milestones": milestones,
                "peak": {
                    "date": str(max_date),
                    "value": float(max_value),
                    "change_from_start_pct": float(((max_value - start_value) / start_value) * 100)
                },
                "trough": {
                    "date": str(min_date),
                    "value": float(min_value),
                    "change_from_start_pct": float(((min_value - start_value) / start_value) * 100)
                }
            }

            return json.dumps({"success": True, "analysis": result}, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class YearOverYearInput(BaseModel):
    """Input schema for YearOverYearTool."""
    indicator: str = Field(..., description="Indicator name")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")


class YearOverYearTool(BaseTool):
    name: str = "Calculate Year-over-Year Growth"
    description: str = (
        "Calculates year-over-year (YoY) growth rates for an indicator. "
        "Essential for analyzing inflation, GDP growth, employment trends, etc. "
        "Removes seasonal effects by comparing to same period last year."
    )
    args_schema: Type[BaseModel] = YearOverYearInput

    def _run(self, indicator: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        try:
            # Load data
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            if indicator in macro_df.columns:
                df = macro_df
            elif indicator in market_df.columns:
                df = market_df
            else:
                return json.dumps({"success": False, "error": f"Indicator '{indicator}' not found"})

            # Filter by date range
            if start_date and end_date:
                series = df.loc[start_date:end_date, indicator]
            elif start_date:
                series = df.loc[start_date:, indicator]
            elif end_date:
                series = df.loc[:end_date, indicator]
            else:
                series = df[indicator]

            series = series.dropna()

            # Calculate YoY change (365 days ago)
            yoy_change = series.pct_change(periods=365) * 100
            yoy_change = yoy_change.dropna()

            if len(yoy_change) == 0:
                return json.dumps({"success": False, "error": "Not enough data for YoY calculation"})

            # Calculate statistics
            stats = {
                "indicator": indicator,
                "current_yoy_pct": float(yoy_change.iloc[-1]) if len(yoy_change) > 0 else None,
                "mean_yoy_pct": float(yoy_change.mean()),
                "max_yoy_pct": float(yoy_change.max()),
                "min_yoy_pct": float(yoy_change.min()),
                "std_yoy_pct": float(yoy_change.std())
            }

            # Find periods of highest/lowest growth
            top_growth = yoy_change.nlargest(5)
            lowest_growth = yoy_change.nsmallest(5)

            stats["top_growth_periods"] = [
                {"date": str(date), "yoy_pct": float(val)}
                for date, val in top_growth.items()
            ]

            stats["lowest_growth_periods"] = [
                {"date": str(date), "yoy_pct": float(val)}
                for date, val in lowest_growth.items()
            ]

            return json.dumps({"success": True, "statistics": stats}, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

