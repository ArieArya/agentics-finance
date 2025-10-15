"""
Visualization tools for DJ30 stock data.
Provides tools for creating price charts, performance comparisons, and portfolio visualizations.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional
import json
import os
import uuid
from datetime import datetime
from utils.dj30_data_loader import get_ticker_data, get_multiple_tickers_data, get_available_tickers


class PriceChartInput(BaseModel):
    """Input for Price Chart Tool."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    chart_type: str = Field(default="candlestick", description="Chart type: 'candlestick', 'line', or 'ohlc'")
    include_volume: bool = Field(default=True, description="Whether to include volume subplot")


class PriceChartTool(BaseTool):
    name: str = "Create Price Chart"
    description: str = (
        "Creates a price chart (candlestick, line, or OHLC) for a stock. "
        "Can include volume subplot. Use this to visualize price movements and patterns."
    )
    args_schema: type[BaseModel] = PriceChartInput

    def _run(self, ticker: str, start_date: str, end_date: str, chart_type: str = "candlestick", include_volume: bool = True) -> str:
        import pandas as pd

        ticker = ticker.upper()

        # Validate ticker
        if ticker not in get_available_tickers():
            return json.dumps({"error": f"Ticker {ticker} not found in DJ30 dataset"})

        # Get data
        df = get_ticker_data(ticker, start_date, end_date)

        if df.empty:
            return json.dumps({"error": f"No data found for {ticker} in date range"})

        # Prepare data for visualization
        chart_data = []
        for _, row in df.iterrows():
            data_point = {
                "date": row['Date'].strftime('%Y-%m-%d'),
                "open": float(row['open']) if not pd.isna(row['open']) else None,
                "high": float(row['high']) if not pd.isna(row['high']) else None,
                "low": float(row['low']) if not pd.isna(row['low']) else None,
                "close": float(row['close']) if not pd.isna(row['close']) else None,
                "volume": int(row['volume']) if not pd.isna(row['volume']) else 0
            }
            chart_data.append(data_point)

        # Create visualization config
        viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        viz_id = f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        viz_config = {
            "type": "price_chart",
            "id": viz_id,
            "title": f"{ticker} Price Chart ({start_date} to {end_date})",
            "ticker": ticker,
            "chart_type": chart_type,
            "include_volume": include_volume,
            "data": chart_data
        }

        viz_file = os.path.join(viz_dir, f"{viz_id}.json")
        with open(viz_file, 'w') as f:
            json.dump(viz_config, f, indent=2)

        result = {
            "success": True,
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "chart_type": chart_type,
            "data_points": len(chart_data),
            "visualization_id": viz_id,
            "message": f"Created {chart_type} price chart for {ticker} with {len(chart_data)} data points (Visualization ID: {viz_id})"
        }

        return json.dumps(result, indent=2)


class PerformanceComparisonChartInput(BaseModel):
    """Input for Performance Comparison Chart Tool."""
    tickers: str = Field(..., description="Comma-separated ticker symbols (e.g., 'AAPL,MSFT,GOOGL')")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    normalize: bool = Field(default=True, description="Normalize prices to 100 at start date for comparison")


class PerformanceComparisonChartTool(BaseTool):
    name: str = "Create Performance Comparison Chart"
    description: str = (
        "Creates a chart comparing price performance across multiple stocks. "
        "Can normalize prices to show relative performance. "
        "Use this to visualize which stocks outperformed/underperformed."
    )
    args_schema: type[BaseModel] = PerformanceComparisonChartInput

    def _run(self, tickers: str, start_date: str, end_date: str, normalize: bool = True) -> str:
        import pandas as pd

        # Parse tickers
        ticker_list = [t.strip().upper() for t in tickers.split(',')]

        # Validate tickers
        available_tickers = get_available_tickers()
        invalid_tickers = [t for t in ticker_list if t not in available_tickers]
        if invalid_tickers:
            return json.dumps({"error": f"Invalid tickers: {', '.join(invalid_tickers)}"})

        # Get data for all tickers
        df = get_multiple_tickers_data(ticker_list, start_date, end_date)

        if df.empty:
            return json.dumps({"error": "No data found for specified tickers and date range"})

        # Prepare data for each ticker
        series_data = {}

        for ticker in ticker_list:
            ticker_df = df[df['ticker'] == ticker].copy()

            if ticker_df.empty:
                continue

            # Sort by date
            ticker_df = ticker_df.sort_values('Date')

            if normalize:
                # Normalize to 100 at start
                base_price = ticker_df['adj_close'].iloc[0]
                ticker_df['normalized_price'] = (ticker_df['adj_close'] / base_price) * 100
                price_column = 'normalized_price'
            else:
                price_column = 'adj_close'

            # Create series data
            series_data[ticker] = [
                {"date": row['Date'].strftime('%Y-%m-%d'), "value": float(row[price_column])}
                for _, row in ticker_df.iterrows()
            ]

        if not series_data:
            return json.dumps({"error": "No valid data for any ticker"})

        # Create visualization config
        viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        viz_id = f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        viz_config = {
            "type": "performance_comparison",
            "id": viz_id,
            "title": f"Performance Comparison: {', '.join(ticker_list)} ({start_date} to {end_date})",
            "normalized": normalize,
            "series": series_data
        }

        viz_file = os.path.join(viz_dir, f"{viz_id}.json")
        with open(viz_file, 'w') as f:
            json.dump(viz_config, f, indent=2)

        result = {
            "success": True,
            "tickers": ticker_list,
            "start_date": start_date,
            "end_date": end_date,
            "normalized": normalize,
            "visualization_id": viz_id,
            "message": f"Created performance comparison chart for {len(ticker_list)} stocks (Visualization ID: {viz_id})"
        }

        return json.dumps(result, indent=2)


class VolatilityChartInput(BaseModel):
    """Input for Volatility Chart Tool."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    window: int = Field(default=30, description="Rolling window for volatility calculation (days)")


class VolatilityChartTool(BaseTool):
    name: str = "Create Volatility Chart"
    description: str = (
        "Creates a chart showing rolling volatility over time for a stock. "
        "Use this to visualize how volatility changes and identify volatile periods."
    )
    args_schema: type[BaseModel] = VolatilityChartInput

    def _run(self, ticker: str, start_date: str, end_date: str, window: int = 30) -> str:
        import pandas as pd
        import numpy as np

        ticker = ticker.upper()

        # Validate ticker
        if ticker not in get_available_tickers():
            return json.dumps({"error": f"Ticker {ticker} not found in DJ30 dataset"})

        # Get data
        df = get_ticker_data(ticker, start_date, end_date)

        if df.empty or len(df) < window:
            return json.dumps({"error": f"Insufficient data for {ticker} in date range"})

        # Calculate daily returns
        df['daily_return'] = df['adj_close'].pct_change()

        # Calculate rolling volatility (annualized)
        df['rolling_vol'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252) * 100

        # Prepare data for visualization
        vol_data = []
        for _, row in df.iterrows():
            if not pd.isna(row['rolling_vol']):
                vol_data.append({
                    "date": row['Date'].strftime('%Y-%m-%d'),
                    "volatility": float(row['rolling_vol'])
                })

        # Create visualization config
        viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        viz_id = f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        viz_config = {
            "type": "volatility_chart",
            "id": viz_id,
            "title": f"{ticker} Rolling {window}-Day Volatility ({start_date} to {end_date})",
            "ticker": ticker,
            "window": window,
            "data": vol_data
        }

        viz_file = os.path.join(viz_dir, f"{viz_id}.json")
        with open(viz_file, 'w') as f:
            json.dump(viz_config, f, indent=2)

        result = {
            "success": True,
            "ticker": ticker,
            "window": window,
            "start_date": start_date,
            "end_date": end_date,
            "data_points": len(vol_data),
            "visualization_id": viz_id,
            "message": f"Created {window}-day rolling volatility chart for {ticker} (Visualization ID: {viz_id})"
        }

        return json.dumps(result, indent=2)

