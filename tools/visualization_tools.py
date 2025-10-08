"""
Tools for creating visualizations.
Outputs JSON files that can be rendered by Streamlit.
"""

from crewai.tools import BaseTool
from typing import Type, Optional, List
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


class TimeSeriesPlotInput(BaseModel):
    """Input schema for TimeSeriesPlotTool."""
    indicators: str = Field(..., description="Comma-separated list of indicators to plot")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    title: Optional[str] = Field(None, description="Chart title")


class TimeSeriesPlotTool(BaseTool):
    name: str = "Create Time Series Plot"
    description: str = (
        "Creates a time series line chart for one or more indicators. "
        "The visualization will be displayed in the Streamlit interface. "
        "Use this to show trends over time."
    )
    args_schema: Type[BaseModel] = TimeSeriesPlotInput

    def _run(self, indicators: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None, title: Optional[str] = None) -> str:
        try:
            # Parse indicators
            indicator_list = [ind.strip() for ind in indicators.split(',')]

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

            # Filter indicators
            available_indicators = [ind for ind in indicator_list if ind in combined_df.columns]

            if not available_indicators:
                return json.dumps({
                    "success": False,
                    "error": f"No valid indicators found. Requested: {indicator_list}"
                })

            # Prepare data for plotting
            plot_data = []
            for indicator in available_indicators:
                series = combined_df[indicator].dropna()
                for date, value in series.items():
                    plot_data.append({
                        "date": str(date),
                        "indicator": indicator,
                        "value": float(value)
                    })

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "time_series",
                "id": viz_id,
                "title": title or f"Time Series: {', '.join(available_indicators)}",
                "data": plot_data,
                "indicators": available_indicators,
                "date_range": {
                    "start": str(combined_df.index.min()),
                    "end": str(combined_df.index.max())
                }
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Time series plot created with {len(available_indicators)} indicator(s)",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class CorrelationHeatmapInput(BaseModel):
    """Input schema for CorrelationHeatmapTool."""
    indicators: str = Field(..., description="Comma-separated list of indicators for correlation matrix")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    title: Optional[str] = Field(None, description="Chart title")


class CorrelationHeatmapTool(BaseTool):
    name: str = "Create Correlation Heatmap"
    description: str = (
        "Creates a correlation heatmap showing relationships between multiple indicators. "
        "The visualization will be displayed in the Streamlit interface. "
        "Use this to visualize correlation matrices."
    )
    args_schema: Type[BaseModel] = CorrelationHeatmapInput

    def _run(self, indicators: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None, title: Optional[str] = None) -> str:
        try:
            # Parse indicators
            indicator_list = [ind.strip() for ind in indicators.split(',')]

            if len(indicator_list) < 2:
                return json.dumps({
                    "success": False,
                    "error": "Need at least 2 indicators for correlation heatmap"
                })

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

            # Filter indicators
            available_indicators = [ind for ind in indicator_list if ind in combined_df.columns]

            if len(available_indicators) < 2:
                return json.dumps({
                    "success": False,
                    "error": f"Not enough valid indicators. Found: {available_indicators}"
                })

            # Calculate correlation matrix
            correlation_matrix = combined_df[available_indicators].corr()

            # Convert to heatmap data format
            heatmap_data = []
            for i, ind1 in enumerate(available_indicators):
                for j, ind2 in enumerate(available_indicators):
                    heatmap_data.append({
                        "x": ind2,
                        "y": ind1,
                        "correlation": float(correlation_matrix.loc[ind1, ind2])
                    })

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "correlation_heatmap",
                "id": viz_id,
                "title": title or "Correlation Heatmap",
                "data": heatmap_data,
                "indicators": available_indicators
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Correlation heatmap created with {len(available_indicators)} indicator(s)",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class VolatilityPlotInput(BaseModel):
    """Input schema for VolatilityPlotTool."""
    indicator: str = Field(..., description="Indicator to plot volatility")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    window: int = Field(default=30, description="Rolling window for volatility calculation")
    title: Optional[str] = Field(None, description="Chart title")


class VolatilityPlotTool(BaseTool):
    name: str = "Create Volatility Plot"
    description: str = (
        "Creates a visualization showing the rolling volatility of an indicator over time. "
        "Shows both the indicator value and its volatility. "
        "Use this to visualize periods of high and low volatility."
    )
    args_schema: Type[BaseModel] = VolatilityPlotInput

    def _run(self, indicator: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None, window: int = 30,
             title: Optional[str] = None) -> str:
        try:
            # Load datasets
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

            # Calculate returns and rolling volatility
            returns = series.pct_change().dropna()
            rolling_vol = returns.rolling(window=window).std()

            # Prepare plot data
            plot_data = []
            for date in series.index:
                entry = {
                    "date": str(date),
                    "value": float(series[date])
                }
                if date in rolling_vol.index and not pd.isna(rolling_vol[date]):
                    entry["volatility"] = float(rolling_vol[date])
                plot_data.append(entry)

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "volatility_plot",
                "id": viz_id,
                "title": title or f"Volatility Analysis: {indicator}",
                "data": plot_data,
                "indicator": indicator,
                "window": window
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Volatility plot created for {indicator}",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class DistributionPlotInput(BaseModel):
    """Input schema for DistributionPlotTool."""
    indicator: str = Field(..., description="Indicator to plot distribution")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    title: Optional[str] = Field(None, description="Chart title")


class DistributionPlotTool(BaseTool):
    name: str = "Create Distribution Plot"
    description: str = (
        "Creates a histogram showing the distribution of values for an indicator. "
        "Use this to understand the frequency distribution and identify outliers."
    )
    args_schema: Type[BaseModel] = DistributionPlotInput

    def _run(self, indicator: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None, title: Optional[str] = None) -> str:
        try:
            # Load datasets
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

            # Convert to list
            values = [float(v) for v in series.values]

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "distribution",
                "id": viz_id,
                "title": title or f"Distribution: {indicator}",
                "data": values,
                "indicator": indicator,
                "stats": {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std())
                }
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Distribution plot created for {indicator}",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })

