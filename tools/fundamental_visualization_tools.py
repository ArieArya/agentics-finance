"""
Visualization tools for company fundamental data.
Outputs JSON files that can be rendered by Streamlit.
"""

from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import json
import os
import uuid
from datetime import datetime
from utils.firm_data_loader import (
    get_company_fundamentals_history, get_multiple_companies_latest,
    calculate_valuation_metrics, get_latest_data
)


VIZ_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)


def generate_viz_id() -> str:
    """Generate a unique visualization ID."""
    return f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class CompanyComparisonChartInput(BaseModel):
    """Input schema for CompanyComparisonChartTool."""
    tickers: str = Field(..., description="Comma-separated list of ticker symbols to compare")
    metrics: str = Field(..., description="Comma-separated list of metrics to compare (e.g., 'ROE,ROA,GRM,PE_RATIO')")
    date: Optional[str] = Field(None, description="Date to compare as of (YYYY-MM-DD)")
    title: Optional[str] = Field(None, description="Chart title")


class CompanyComparisonChartTool(BaseTool):
    name: str = "Create Company Comparison Chart"
    description: str = (
        "Creates a bar chart comparing fundamental metrics across multiple companies. "
        "Use this to visualize side-by-side comparison of ROE, ROA, P/E ratios, margins, etc. "
        "The visualization will be displayed in the Streamlit interface. "
        "Supports comparing 2-10 companies across multiple metrics."
    )
    args_schema: Type[BaseModel] = CompanyComparisonChartInput

    def _run(self, tickers: str, metrics: str, date: Optional[str] = None, title: Optional[str] = None) -> str:
        try:
            ticker_list = [t.strip().upper() for t in tickers.split(',')]
            metric_list = [m.strip().upper() for m in metrics.split(',')]

            if len(ticker_list) < 2:
                return json.dumps({
                    "success": False,
                    "error": "Need at least 2 companies to compare"
                })

            # Collect data for all companies
            comparison_data = []
            for ticker in ticker_list:
                data = get_latest_data(ticker, date)
                if data and data.get('PRICE'):
                    valuation = calculate_valuation_metrics(ticker, date)

                    company_metrics = {
                        "ticker": ticker,
                        "ROE": data.get('ROE'),
                        "ROA": data.get('ROA'),
                        "GRM": data.get('GRM'),
                        "EPS": data.get('EPS'),
                        "PE_RATIO": valuation.get('pe_ratio'),
                        "PB_RATIO": valuation.get('pb_ratio'),
                        "DIVIDEND_YIELD": valuation.get('dividend_yield'),
                        "EPS_GROWTH": data.get('FVYRGRO_EPS')
                    }
                    comparison_data.append(company_metrics)

            if not comparison_data:
                return json.dumps({
                    "success": False,
                    "error": "No data found for any of the tickers"
                })

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "company_comparison",
                "id": viz_id,
                "title": title or f"Company Comparison: {', '.join(ticker_list)}",
                "data": comparison_data,
                "metrics": metric_list,
                "tickers": ticker_list
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Comparison chart created for {len(ticker_list)} companies across {len(metric_list)} metrics",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class FundamentalTimeSeriesInput(BaseModel):
    """Input schema for FundamentalTimeSeriesPlotTool."""
    ticker: str = Field(..., description="Company ticker symbol")
    metrics: str = Field(..., description="Comma-separated list of metrics to plot (e.g., 'EPS,ROE,PRICE')")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    title: Optional[str] = Field(None, description="Chart title")


class FundamentalTimeSeriesPlotTool(BaseTool):
    name: str = "Create Fundamental Time Series Plot"
    description: str = (
        "Creates a time series line chart showing evolution of company fundamentals over time. "
        "Use this to visualize trends in EPS, ROE, ROA, price, or other metrics. "
        "The visualization will be displayed in the Streamlit interface. "
        "Supports plotting multiple metrics on the same chart with dual y-axes if needed."
    )
    args_schema: Type[BaseModel] = FundamentalTimeSeriesInput

    def _run(self, ticker: str, metrics: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None, title: Optional[str] = None) -> str:
        try:
            ticker = ticker.upper()
            metric_list = [m.strip().upper() for m in metrics.split(',')]

            # Get historical data
            history = get_company_fundamentals_history(ticker, start_date, end_date)

            if history.empty:
                return json.dumps({
                    "success": False,
                    "error": f"No historical data found for {ticker}"
                })

            # Prepare data for plotting
            plot_data = []
            available_metrics = [m for m in metric_list if m in history.columns]

            if not available_metrics:
                return json.dumps({
                    "success": False,
                    "error": f"No valid metrics found. Available: {', '.join(history.columns)}"
                })

            for date, row in history.iterrows():
                for metric in available_metrics:
                    value = row.get(metric)
                    if pd.notna(value):
                        plot_data.append({
                            "date": str(date.date()),
                            "metric": metric,
                            "value": float(value)
                        })

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "fundamental_time_series",
                "id": viz_id,
                "title": title or f"{ticker} Fundamentals Over Time",
                "data": plot_data,
                "metrics": available_metrics,
                "ticker": ticker,
                "date_range": {
                    "start": str(history.index.min().date()),
                    "end": str(history.index.max().date())
                }
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Time series plot created for {ticker} with {len(available_metrics)} metrics",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class ValuationScatterPlotInput(BaseModel):
    """Input schema for ValuationScatterPlotTool."""
    tickers: str = Field(..., description="Comma-separated list of ticker symbols")
    x_metric: str = Field(..., description="Metric for x-axis (e.g., ROE, EPS_GROWTH)")
    y_metric: str = Field(..., description="Metric for y-axis (e.g., PE_RATIO, PB_RATIO)")
    date: Optional[str] = Field(None, description="Date to analyze (YYYY-MM-DD)")
    title: Optional[str] = Field(None, description="Chart title")


class ValuationScatterPlotTool(BaseTool):
    name: str = "Create Valuation Scatter Plot"
    description: str = (
        "Creates a scatter plot showing relationship between two fundamental metrics. "
        "Use this to identify value vs. quality trade-offs, growth vs. valuation relationships, etc. "
        "The visualization will be displayed in the Streamlit interface. "
        "Each point represents a company, labeled with its ticker. "
        "Useful for identifying outliers and patterns across companies."
    )
    args_schema: Type[BaseModel] = ValuationScatterPlotInput

    def _run(self, tickers: str, x_metric: str, y_metric: str,
             date: Optional[str] = None, title: Optional[str] = None) -> str:
        try:
            ticker_list = [t.strip().upper() for t in tickers.split(',')]
            x_metric = x_metric.upper()
            y_metric = y_metric.upper()

            # Collect data for all companies
            scatter_data = []
            for ticker in ticker_list:
                data = get_latest_data(ticker, date)
                if not data or not data.get('PRICE'):
                    continue

                valuation = calculate_valuation_metrics(ticker, date)

                # Map metric names to values
                metric_values = {
                    "ROE": data.get('ROE'),
                    "ROA": data.get('ROA'),
                    "GRM": data.get('GRM'),
                    "EPS": data.get('EPS'),
                    "PE_RATIO": valuation.get('pe_ratio'),
                    "PB_RATIO": valuation.get('pb_ratio'),
                    "EPS_GROWTH": data.get('FVYRGRO_EPS'),
                    "DIVIDEND_YIELD": valuation.get('dividend_yield'),
                    "PRICE": data.get('PRICE')
                }

                x_value = metric_values.get(x_metric)
                y_value = metric_values.get(y_metric)

                if x_value is not None and y_value is not None:
                    scatter_data.append({
                        "ticker": ticker,
                        "x": float(x_value),
                        "y": float(y_value)
                    })

            if len(scatter_data) < 2:
                return json.dumps({
                    "success": False,
                    "error": f"Insufficient data for scatter plot. Found {len(scatter_data)} companies with both metrics."
                })

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "valuation_scatter",
                "id": viz_id,
                "title": title or f"{y_metric} vs {x_metric}",
                "data": scatter_data,
                "x_metric": x_metric,
                "y_metric": y_metric,
                "tickers": ticker_list
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Scatter plot created with {len(scatter_data)} companies",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class PortfolioRecommendationChartInput(BaseModel):
    """Input schema for PortfolioRecommendationChartTool."""
    long_tickers: str = Field(..., description="Comma-separated list of tickers to long")
    short_tickers: str = Field(..., description="Comma-separated list of tickers to short")
    date: Optional[str] = Field(None, description="Date to analyze (YYYY-MM-DD)")
    title: Optional[str] = Field(None, description="Chart title")


class PortfolioRecommendationChartTool(BaseTool):
    name: str = "Create Portfolio Recommendation Chart"
    description: str = (
        "Creates a visualization showing long/short portfolio recommendations with key metrics. "
        "Displays recommended long positions (green) and short positions (red) side-by-side. "
        "Shows ROE, P/E ratio, and EPS growth for each recommended position. "
        "Use this to visualize portfolio recommendations from the portfolio recommendation tool."
    )
    args_schema: Type[BaseModel] = PortfolioRecommendationChartInput

    def _run(self, long_tickers: str, short_tickers: str,
             date: Optional[str] = None, title: Optional[str] = None) -> str:
        try:
            long_list = [t.strip().upper() for t in long_tickers.split(',')]
            short_list = [t.strip().upper() for t in short_tickers.split(',')]

            # Collect data for long positions
            long_data = []
            for ticker in long_list:
                data = get_latest_data(ticker, date)
                if data and data.get('PRICE'):
                    valuation = calculate_valuation_metrics(ticker, date)
                    long_data.append({
                        "ticker": ticker,
                        "position": "LONG",
                        "roe": data.get('ROE'),
                        "pe_ratio": valuation.get('pe_ratio'),
                        "eps_growth": data.get('FVYRGRO_EPS'),
                        "price": data.get('PRICE')
                    })

            # Collect data for short positions
            short_data = []
            for ticker in short_list:
                data = get_latest_data(ticker, date)
                if data and data.get('PRICE'):
                    valuation = calculate_valuation_metrics(ticker, date)
                    short_data.append({
                        "ticker": ticker,
                        "position": "SHORT",
                        "roe": data.get('ROE'),
                        "pe_ratio": valuation.get('pe_ratio'),
                        "eps_growth": data.get('FVYRGRO_EPS'),
                        "price": data.get('PRICE')
                    })

            # Generate visualization ID
            viz_id = generate_viz_id()

            # Create visualization config
            viz_config = {
                "type": "portfolio_recommendation",
                "id": viz_id,
                "title": title or "Long/Short Portfolio Recommendations",
                "long_positions": long_data,
                "short_positions": short_data
            }

            # Save to file
            viz_file = os.path.join(VIZ_DIR, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            return json.dumps({
                "success": True,
                "visualization_id": viz_id,
                "message": f"Portfolio chart created with {len(long_data)} long and {len(short_data)} short positions",
                "file": viz_file
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


# Import pandas for time series tool
import pandas as pd


