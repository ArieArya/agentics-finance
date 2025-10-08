"""
News and Event Analysis Tools

Tools for fetching financial news headlines and correlating them with market events.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
import json
import os
from datetime import datetime, timedelta
import ast
from utils.data_loader import load_macro_factors, load_market_factors


class HeadlinesFetcherInput(BaseModel):
    """Input schema for HeadlinesFetcherTool."""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    max_headlines: int = Field(
        default=10,
        description="Maximum number of headlines to return per day (default: 10)"
    )


class HeadlinesFetcherTool(BaseTool):
    name: str = "Fetch Financial News Headlines"
    description: str = (
        "Fetches financial news headlines for a specific date range from the dataset's Headlines column. "
        "Use this to understand market context, explain volatility, or find events that impacted markets. "
        "Returns headlines with their publication dates. "
        "Useful for answering 'why' questions about market movements and explaining volatility."
    )
    args_schema: Type[BaseModel] = HeadlinesFetcherInput

    def _run(
        self,
        start_date: str,
        end_date: str,
        max_headlines: int = 10
    ) -> str:
        try:
            # Load market factors data which contains Headlines column
            market_df = load_market_factors()

            # Filter by date range
            filtered = market_df.loc[start_date:end_date].copy()

            if len(filtered) == 0:
                return json.dumps({
                    "success": False,
                    "error": f"No data found for date range {start_date} to {end_date}"
                })

            # Extract headlines
            all_headlines = []
            for date, row in filtered.iterrows():
                headlines_raw = row.get('Headlines')

                if not headlines_raw or str(headlines_raw) == 'nan':
                    continue

                # Parse the headlines (they're stored as string representation of list)
                try:
                    # Try to parse as Python literal
                    headlines_list = ast.literal_eval(str(headlines_raw))

                    if isinstance(headlines_list, list):
                        # Take only the first max_headlines per day
                        for headline in headlines_list[:max_headlines]:
                            if headline and str(headline).strip():
                                all_headlines.append({
                                    "date": str(date.date()),
                                    "headline": str(headline).strip()
                                })
                except:
                    # If parsing fails, treat as single headline
                    if str(headlines_raw).strip():
                        all_headlines.append({
                            "date": str(date.date()),
                            "headline": str(headlines_raw).strip()
                        })

            if not all_headlines:
                return json.dumps({
                    "success": True,
                    "date_range": {"start": start_date, "end": end_date},
                    "message": "No headlines found for this date range",
                    "headlines": []
                })

            return json.dumps({
                "success": True,
                "date_range": {"start": start_date, "end": end_date},
                "total_headlines": len(all_headlines),
                "headlines": all_headlines
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class VolatilityNewsCorrelationInput(BaseModel):
    """Input schema for VolatilityNewsCorrelationTool."""
    indicator: str = Field(..., description="Indicator to analyze (e.g., '^GSPC', '^VIX', 'BTC-USD')")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")


class VolatilityNewsCorrelationTool(BaseTool):
    name: str = "Correlate Volatility with News Events"
    description: str = (
        "Analyzes volatility patterns and identifies dates with significant volatility spikes, "
        "then suggests fetching news for those specific dates to understand the cause. "
        "Use this to explain WHY volatility behaved a certain way."
    )
    args_schema: Type[BaseModel] = VolatilityNewsCorrelationInput

    def _run(self, indicator: str, start_date: str, end_date: str) -> str:
        try:
            # Load data
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            # Find indicator
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
            filtered = df.loc[start_date:end_date, [indicator]].copy()

            # Calculate daily returns and volatility
            filtered['returns'] = filtered[indicator].pct_change()
            filtered['abs_returns'] = filtered['returns'].abs()

            # Calculate rolling volatility (5-day)
            filtered['volatility'] = filtered['returns'].rolling(window=5).std() * 100

            # Find significant volatility spikes (top 10%)
            volatility_threshold = filtered['volatility'].quantile(0.90)

            # Find dates with high volatility
            high_vol_dates = filtered[filtered['volatility'] > volatility_threshold].copy()
            high_vol_dates = high_vol_dates.dropna()

            # Sort by volatility
            high_vol_dates = high_vol_dates.sort_values('volatility', ascending=False).head(10)

            # Get headlines for these high volatility dates
            # Load market data to get headlines
            market_data_full = load_market_factors()

            # Format results with headlines
            significant_dates = []
            for date, row in high_vol_dates.iterrows():
                date_str = str(date.date())

                # Get headlines for this date
                headlines_for_date = []
                try:
                    if date in market_data_full.index:
                        headlines_raw = market_data_full.loc[date, 'Headlines']
                        if headlines_raw and str(headlines_raw) != 'nan':
                            try:
                                headlines_list = ast.literal_eval(str(headlines_raw))
                                if isinstance(headlines_list, list):
                                    # Take top 5 headlines
                                    headlines_for_date = [str(h).strip() for h in headlines_list[:5] if h and str(h).strip()]
                            except:
                                pass
                except:
                    pass

                significant_dates.append({
                    "date": date_str,
                    "value": float(row[indicator]),
                    "daily_return": float(row['returns']) * 100,
                    "volatility": float(row['volatility']),
                    "headlines": headlines_for_date if headlines_for_date else ["No headlines available for this date"]
                })

            return json.dumps({
                "success": True,
                "indicator": indicator,
                "date_range": {"start": start_date, "end": end_date},
                "analysis": {
                    "avg_volatility": float(filtered['volatility'].mean()),
                    "max_volatility": float(filtered['volatility'].max()),
                    "volatility_threshold": float(volatility_threshold)
                },
                "significant_volatility_dates": significant_dates,
                "note": "Headlines are automatically included for high volatility dates to explain the causes."
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class EventTimelineInput(BaseModel):
    """Input schema for EventTimelineTool."""
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    indicators: str = Field(..., description="Comma-separated list of indicators to track (e.g., '^GSPC,^VIX')")


class EventTimelineTool(BaseTool):
    name: str = "Create Event Timeline"
    description: str = (
        "Creates a timeline showing significant price movements across multiple indicators, "
        "identifying dates where major events likely occurred. "
        "Helps understand market-wide impacts and correlate events across different assets."
    )
    args_schema: Type[BaseModel] = EventTimelineInput

    def _run(self, start_date: str, end_date: str, indicators: str) -> str:
        try:
            indicator_list = [ind.strip() for ind in indicators.split(',')]

            # Load data
            macro_df = load_macro_factors()
            market_df = load_market_factors()

            timeline_events = {}

            for indicator in indicator_list:
                # Find indicator
                if indicator in macro_df.columns:
                    series = macro_df.loc[start_date:end_date, indicator]
                elif indicator in market_df.columns:
                    series = market_df.loc[start_date:end_date, indicator]
                else:
                    continue

                # Calculate returns
                returns = series.pct_change() * 100

                # Find significant moves (> 2 std deviations)
                threshold = returns.std() * 2
                significant = returns[abs(returns) > threshold].dropna()

                for date, ret in significant.items():
                    date_str = str(date.date())
                    if date_str not in timeline_events:
                        timeline_events[date_str] = []

                    timeline_events[date_str].append({
                        "indicator": indicator,
                        "return": float(ret),
                        "direction": "up" if ret > 0 else "down"
                    })

            # Sort by date
            sorted_timeline = [
                {
                    "date": date,
                    "movements": events,
                    "num_indicators_affected": len(events)
                }
                for date, events in sorted(timeline_events.items())
            ]

            return json.dumps({
                "success": True,
                "date_range": {"start": start_date, "end": end_date},
                "indicators": indicator_list,
                "timeline": sorted_timeline,
                "recommendation": "Dates with multiple indicators affected likely indicate major market events. Use news tool to investigate these dates."
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })

