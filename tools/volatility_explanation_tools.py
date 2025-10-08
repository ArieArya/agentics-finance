"""
Comprehensive Volatility Explanation Tools

Tools specifically designed to explain volatility spikes using all available data:
market indicators, macro factors, and news headlines.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
import json
import ast
from datetime import datetime, timedelta
import pandas as pd
from utils.data_loader import load_macro_factors, load_market_factors


class ComprehensiveVolatilityExplanationInput(BaseModel):
    """Input schema for ComprehensiveVolatilityExplanationTool."""
    indicator: str = Field(..., description="Indicator to analyze (e.g., '^GSPC', '^VIX', 'BTC-USD')")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    top_n_dates: int = Field(default=5, description="Number of highest volatility dates to explain (default: 5)")


class ComprehensiveVolatilityExplanationTool(BaseTool):
    name: str = "Explain Volatility Comprehensively"
    description: str = (
        "Provides comprehensive explanation for volatility spikes by analyzing the indicator's volatility, "
        "identifying high volatility dates, and providing context from: "
        "1) News headlines for those dates, "
        "2) Macro economic factors (inflation, rates, unemployment, etc.), "
        "3) Related market indicators (VIX, other markets). "
        "This is the PRIMARY tool for answering 'why was X volatile' questions."
    )
    args_schema: Type[BaseModel] = ComprehensiveVolatilityExplanationInput

    def _run(
        self,
        indicator: str,
        start_date: str,
        end_date: str,
        top_n_dates: int = 5
    ) -> str:
        try:
            # Load data
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            # Find indicator
            if indicator in macro_df.columns:
                df = macro_df
                indicator_type = "macro"
            elif indicator in market_df.columns:
                df = market_df
                indicator_type = "market"
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Indicator '{indicator}' not found"
                })

            # Filter by date range
            filtered = df.loc[start_date:end_date, [indicator]].copy()

            # Calculate daily returns and volatility
            filtered['returns'] = filtered[indicator].pct_change()
            filtered['abs_returns'] = filtered['returns'].abs()
            filtered['volatility'] = filtered['returns'].rolling(window=5).std() * 100

            # Find significant volatility spikes
            volatility_threshold = filtered['volatility'].quantile(0.90)
            high_vol_dates = filtered[filtered['volatility'] > volatility_threshold].copy()
            high_vol_dates = high_vol_dates.dropna()
            high_vol_dates = high_vol_dates.sort_values('volatility', ascending=False).head(top_n_dates)

            # For each high volatility date, gather comprehensive context
            explanations = []

            for date, row in high_vol_dates.iterrows():
                date_str = str(date.date())

                # Get a 3-day window around the spike (day before, day of, day after)
                window_start = date - timedelta(days=1)
                window_end = date + timedelta(days=1)

                # 1. Get headlines
                headlines = []
                try:
                    if date in market_df.index:
                        headlines_raw = market_df.loc[date, 'Headlines']
                        if headlines_raw and str(headlines_raw) != 'nan':
                            try:
                                headlines_list = ast.literal_eval(str(headlines_raw))
                                if isinstance(headlines_list, list):
                                    headlines = [str(h).strip() for h in headlines_list[:5] if h and str(h).strip()]
                            except:
                                pass
                except:
                    pass

                # 2. Get macro factors context (key economic indicators)
                macro_context = {}
                key_macro_indicators = ['FEDFUNDS', 'CPIAUCSL', 'UNRATE', 'DGS10', 'DGS2']
                for macro_ind in key_macro_indicators:
                    if macro_ind in macro_df.columns:
                        try:
                            macro_window = macro_df.loc[window_start:window_end, [macro_ind]]
                            if len(macro_window) > 0:
                                values = macro_window[macro_ind].dropna()
                                if len(values) > 0:
                                    # Calculate change during window
                                    if len(values) > 1:
                                        change = float(values.iloc[-1] - values.iloc[0])
                                        pct_change = float((change / values.iloc[0]) * 100) if values.iloc[0] != 0 else 0
                                    else:
                                        change = 0
                                        pct_change = 0

                                    macro_context[macro_ind] = {
                                        "value": float(values.iloc[-1]) if len(values) > 0 else None,
                                        "change": change,
                                        "pct_change": pct_change
                                    }
                        except:
                            pass

                # 3. Get related market indicators
                market_context = {}
                key_market_indicators = ['^VIX', '^GSPC', 'GLD', 'DGS10', 'DCOILBRENTEU']
                # Don't include the indicator being analyzed
                key_market_indicators = [m for m in key_market_indicators if m != indicator]

                for market_ind in key_market_indicators:
                    if market_ind in market_df.columns:
                        try:
                            market_window = market_df.loc[window_start:window_end, [market_ind]]
                            if len(market_window) > 0:
                                values = market_window[market_ind].dropna()
                                if len(values) > 0:
                                    # Calculate change during window
                                    if len(values) > 1:
                                        change = float(values.iloc[-1] - values.iloc[0])
                                        pct_change = float((change / values.iloc[0]) * 100) if values.iloc[0] != 0 else 0
                                    else:
                                        change = 0
                                        pct_change = 0

                                    market_context[market_ind] = {
                                        "value": float(values.iloc[-1]) if len(values) > 0 else None,
                                        "change": change,
                                        "pct_change": pct_change
                                    }
                        except:
                            pass

                # Build comprehensive explanation for this date
                explanations.append({
                    "date": date_str,
                    "volatility_metrics": {
                        "volatility": float(row['volatility']),
                        "daily_return": float(row['returns']) * 100,
                        "indicator_value": float(row[indicator])
                    },
                    "headlines": headlines if headlines else ["No headlines available"],
                    "macro_factors_context": macro_context,
                    "market_indicators_context": market_context
                })

            return json.dumps({
                "success": True,
                "indicator": indicator,
                "date_range": {"start": start_date, "end": end_date},
                "volatility_summary": {
                    "avg_volatility": float(filtered['volatility'].mean()),
                    "max_volatility": float(filtered['volatility'].max()),
                    "threshold_for_high_vol": float(volatility_threshold),
                    "num_high_vol_days": len(high_vol_dates)
                },
                "high_volatility_dates_explained": explanations,
                "interpretation_guide": {
                    "volatility": "5-day rolling standard deviation of returns (higher = more volatile)",
                    "macro_factors": "Key economic indicators during volatility spike",
                    "market_indicators": "Related market movements during volatility spike",
                    "headlines": "News events that may have caused the volatility"
                }
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class IdentifyCorrelatedMovementsInput(BaseModel):
    """Input schema for IdentifyCorrelatedMovementsTool."""
    reference_date: str = Field(..., description="Reference date to analyze in YYYY-MM-DD format")
    window_days: int = Field(default=3, description="Number of days to look around the reference date (default: 3)")


class IdentifyCorrelatedMovementsTool(BaseTool):
    name: str = "Identify Correlated Market Movements"
    description: str = (
        "Identifies which market and macro indicators had significant movements "
        "around a specific date (e.g., a volatility spike). "
        "Helps understand if a volatility event was isolated or market-wide. "
        "Shows what else was moving at the same time."
    )
    args_schema: Type[BaseModel] = IdentifyCorrelatedMovementsInput

    def _run(self, reference_date: str, window_days: int = 3) -> str:
        try:
            ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
            start_date = ref_date - timedelta(days=window_days)
            end_date = ref_date + timedelta(days=window_days)

            # Load data
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            # Analyze all indicators
            significant_movements = []

            # Check market indicators
            for col in market_df.columns:
                if col == 'Headlines':
                    continue

                try:
                    window_data = market_df.loc[start_date:end_date, col]
                    window_data = window_data.dropna()

                    if len(window_data) > 1:
                        change = float(window_data.iloc[-1] - window_data.iloc[0])
                        pct_change = float((change / window_data.iloc[0]) * 100) if window_data.iloc[0] != 0 else 0

                        # Consider significant if > 2% change
                        if abs(pct_change) > 2:
                            significant_movements.append({
                                "indicator": col,
                                "type": "market",
                                "change": change,
                                "pct_change": pct_change,
                                "direction": "increase" if change > 0 else "decrease",
                                "start_value": float(window_data.iloc[0]),
                                "end_value": float(window_data.iloc[-1])
                            })
                except:
                    continue

            # Check macro indicators
            for col in macro_df.columns:
                try:
                    window_data = macro_df.loc[start_date:end_date, col]
                    window_data = window_data.dropna()

                    if len(window_data) > 1:
                        change = float(window_data.iloc[-1] - window_data.iloc[0])
                        pct_change = float((change / window_data.iloc[0]) * 100) if window_data.iloc[0] != 0 else 0

                        # Consider significant if > 1% change (macro moves slower)
                        if abs(pct_change) > 1:
                            significant_movements.append({
                                "indicator": col,
                                "type": "macro",
                                "change": change,
                                "pct_change": pct_change,
                                "direction": "increase" if change > 0 else "decrease",
                                "start_value": float(window_data.iloc[0]),
                                "end_value": float(window_data.iloc[-1])
                            })
                except:
                    continue

            # Sort by magnitude of change
            significant_movements.sort(key=lambda x: abs(x['pct_change']), reverse=True)

            # Get headlines for reference date
            headlines = []
            try:
                if ref_date in market_df.index:
                    headlines_raw = market_df.loc[ref_date, 'Headlines']
                    if headlines_raw and str(headlines_raw) != 'nan':
                        try:
                            headlines_list = ast.literal_eval(str(headlines_raw))
                            if isinstance(headlines_list, list):
                                headlines = [str(h).strip() for h in headlines_list[:10] if h and str(h).strip()]
                        except:
                            pass
            except:
                pass

            return json.dumps({
                "success": True,
                "reference_date": reference_date,
                "window": {
                    "start": str(start_date.date()),
                    "end": str(end_date.date()),
                    "days": window_days * 2 + 1
                },
                "significant_movements": significant_movements,
                "total_indicators_moved": len(significant_movements),
                "headlines_on_reference_date": headlines if headlines else ["No headlines available"],
                "interpretation": (
                    "Market-wide event" if len(significant_movements) > 10
                    else "Isolated movement" if len(significant_movements) < 5
                    else "Moderate correlation"
                )
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })

