"""
Advanced visualization tools for financial data.
Includes scatter plots, comparative charts, moving averages, and multi-indicator dashboards.
"""

from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import pandas as pd
import json
import os
import uuid
from datetime import datetime
from utils.data_loader import load_macro_factors, load_market_factors


VIZ_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)


def generate_viz_id() -> str:
    """Generate a unique visualization ID."""
    return f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class ScatterPlotInput(BaseModel):
    """Input schema for ScatterPlotTool."""
    x_indicator: str = Field(..., description="Indicator for X-axis")
    y_indicator: str = Field(..., description="Indicator for Y-axis")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    title: Optional[str] = Field(None, description="Chart title")


class ScatterPlotTool(BaseTool):
    name: str = "Create Scatter Plot"
    description: str = (
        "Creates a scatter plot showing the relationship between two indicators. "
        "Useful for visualizing correlations and identifying patterns. "
        "E.g., unemployment vs stock market, oil prices vs inflation."
    )
    args_schema: Type[BaseModel] = ScatterPlotInput

    def _run(self, x_indicator: str, y_indicator: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None, title: Optional[str] = None) -> str:
        try:
            # Load datasets
            macro_df = load_macro_factors()
            market_df = load_market_factors()
            combined_df = pd.concat([macro_df, market_df], axis=1)

            # Filter by date range
            if start_date and end_date:
                combined_df = combined_df.loc[start_date:end_date]
            elif start_date:
                combined_df = combined_df.loc[start_date:]
            elif end_date:
                combined_df = combined_df.loc[:end_date]

            # Check if indicators exist
            if x_indicator not in combined_df.columns or y_indicator not in combined_df.columns:
                return json.dumps({
                    "success": False,
                    "error": f"One or both indicators not found"
                })

            # Get data and drop NaN
            scatter_data = combined_df[[x_indicator, y_indicator]].dropna()

            # Prepare plot data
            plot_data = [
                {
                    "x": float(row[x_indicator]),
                    "y": float(row[y_indicator]),
                    "date": str(date)
                }
                for date, row in scatter_data.iterrows()
            ]

            # Calculate correlation
            correlation = scatter_data[x_indicator].corr(scatter_data[y_indicator])

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "scatter",
                "id": viz_id,
                "title": title or f"{y_indicator} vs {x_indicator}",
                "data": plot_data,
                "x_indicator": x_indicator,
                "y_indicator": y_indicator,
                "correlation": float(correlation),
                "count": len(plot_data)
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Scatter plot created showing relationship between {x_indicator} and {y_indicator}",
                "correlation": float(correlation),
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class ComparativePerformanceInput(BaseModel):
    """Input schema for ComparativePerformanceTool."""
    indicators: str = Field(..., description="Comma-separated list of indicators to compare")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    title: Optional[str] = Field(None, description="Chart title")


class ComparativePerformanceTool(BaseTool):
    name: str = "Create Comparative Performance Chart"
    description: str = (
        "Creates a chart comparing normalized performance of multiple indicators. "
        "All indicators start at 100 for easy comparison. "
        "Perfect for comparing asset performance, e.g., stocks vs gold vs Bitcoin."
    )
    args_schema: Type[BaseModel] = ComparativePerformanceInput

    def _run(self, indicators: str, start_date: str, end_date: str, title: Optional[str] = None) -> str:
        try:
            # Parse indicators
            indicator_list = [ind.strip() for ind in indicators.split(',')]

            if len(indicator_list) < 2:
                return json.dumps({"success": False, "error": "Need at least 2 indicators"})

            # Load datasets
            macro_df = load_macro_factors()
            market_df = load_market_factors()
            combined_df = pd.concat([macro_df, market_df], axis=1)

            # Filter by date range
            combined_df = combined_df.loc[start_date:end_date]

            # Check if indicators exist
            available_indicators = [ind for ind in indicator_list if ind in combined_df.columns]

            if len(available_indicators) < 2:
                return json.dumps({
                    "success": False,
                    "error": f"Not enough valid indicators. Found: {available_indicators}"
                })

            # Normalize all indicators to start at 100
            plot_data = []
            for indicator in available_indicators:
                series = combined_df[indicator].dropna()
                if len(series) > 0:
                    normalized = (series / series.iloc[0]) * 100
                    for date, value in normalized.items():
                        plot_data.append({
                            "date": str(date),
                            "indicator": indicator,
                            "value": float(value)
                        })

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "comparative_performance",
                "id": viz_id,
                "title": title or f"Comparative Performance: {', '.join(available_indicators)}",
                "data": plot_data,
                "indicators": available_indicators,
                "base_value": 100
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Comparative performance chart created for {len(available_indicators)} indicators",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class MovingAveragePlotInput(BaseModel):
    """Input schema for MovingAveragePlotTool."""
    indicator: str = Field(..., description="Indicator to plot")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    windows: str = Field(..., description="Comma-separated MA windows (e.g., '20,50,200')")
    title: Optional[str] = Field(None, description="Chart title")


class MovingAveragePlotTool(BaseTool):
    name: str = "Create Moving Average Chart"
    description: str = (
        "Creates a chart with price and multiple moving averages. "
        "Common periods: 20-day (short), 50-day (medium), 200-day (long). "
        "Helps identify trends and potential entry/exit points."
    )
    args_schema: Type[BaseModel] = MovingAveragePlotInput

    def _run(self, indicator: str, start_date: str, end_date: str, windows: str,
             title: Optional[str] = None) -> str:
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

            # Prepare plot data
            plot_data = []
            for date, value in series.items():
                entry = {
                    "date": str(date),
                    "price": float(value)
                }

                # Calculate moving averages for this date
                for window in window_list:
                    ma = series.loc[:date].tail(window).mean()
                    if not pd.isna(ma):
                        entry[f"MA_{window}"] = float(ma)

                plot_data.append(entry)

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "moving_average",
                "id": viz_id,
                "title": title or f"{indicator} with Moving Averages",
                "data": plot_data,
                "indicator": indicator,
                "windows": window_list
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Moving average chart created for {indicator}",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class DrawdownChartInput(BaseModel):
    """Input schema for DrawdownChartTool."""
    indicator: str = Field(..., description="Indicator to plot")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    title: Optional[str] = Field(None, description="Chart title")


class DrawdownChartTool(BaseTool):
    name: str = "Create Drawdown Chart"
    description: str = (
        "Creates a chart showing price and drawdown from peak over time. "
        "Visualizes periods of decline and recovery. "
        "Essential for understanding risk and volatility."
    )
    args_schema: Type[BaseModel] = DrawdownChartInput

    def _run(self, indicator: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None, title: Optional[str] = None) -> str:
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

            # Calculate running maximum and drawdown
            running_max = series.expanding().max()
            drawdown = (series - running_max) / running_max * 100

            # Prepare plot data
            plot_data = []
            for date in series.index:
                plot_data.append({
                    "date": str(date),
                    "price": float(series[date]),
                    "drawdown_pct": float(drawdown[date])
                })

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "drawdown",
                "id": viz_id,
                "title": title or f"{indicator} Drawdown Analysis",
                "data": plot_data,
                "indicator": indicator,
                "max_drawdown": float(drawdown.min())
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Drawdown chart created for {indicator}",
                "max_drawdown": float(drawdown.min()),
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


class MultiIndicatorPlotInput(BaseModel):
    """Input schema for MultiIndicatorPlotTool."""
    indicators: str = Field(..., description="Comma-separated list of related indicators")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    title: Optional[str] = Field(None, description="Chart title")


class MultiIndicatorPlotTool(BaseTool):
    name: str = "Create Multi-Indicator Dashboard"
    description: str = (
        "Creates a multi-panel dashboard showing several related indicators. "
        "Each indicator on its own scale for easy comparison of trends. "
        "Useful for economic dashboards, e.g., unemployment + GDP + inflation."
    )
    args_schema: Type[BaseModel] = MultiIndicatorPlotInput

    def _run(self, indicators: str, start_date: str, end_date: str, title: Optional[str] = None) -> str:
        try:
            # Parse indicators
            indicator_list = [ind.strip() for ind in indicators.split(',')]

            if len(indicator_list) < 2:
                return json.dumps({"success": False, "error": "Need at least 2 indicators"})

            # Load datasets
            macro_df = load_macro_factors()
            market_df = load_market_factors()
            combined_df = pd.concat([macro_df, market_df], axis=1)

            # Filter by date range
            combined_df = combined_df.loc[start_date:end_date]

            # Check if indicators exist
            available_indicators = [ind for ind in indicator_list if ind in combined_df.columns]

            if len(available_indicators) < 2:
                return json.dumps({
                    "success": False,
                    "error": f"Not enough valid indicators. Found: {available_indicators}"
                })

            # Prepare plot data (organized by indicator for subplots)
            plot_data = {}
            for indicator in available_indicators:
                series = combined_df[indicator].dropna()
                plot_data[indicator] = [
                    {
                        "date": str(date),
                        "value": float(value)
                    }
                    for date, value in series.items()
                ]

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "multi_indicator",
                "id": viz_id,
                "title": title or f"Multi-Indicator Dashboard",
                "data": plot_data,
                "indicators": available_indicators
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Multi-indicator dashboard created with {len(available_indicators)} indicators",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

