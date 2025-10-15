"""
Price analysis tools for DJ30 stock data.
Provides tools for analyzing returns, volatility, momentum, and performance.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional
import json
import pandas as pd
import numpy as np
from utils.dj30_data_loader import get_ticker_data, get_multiple_tickers_data, get_available_tickers


class ReturnsAnalysisInput(BaseModel):
    """Input for Returns Analysis Tool."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, MSFT)")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    period: str = Field(default="daily", description="Period for returns: 'daily', 'weekly', 'monthly'")


class ReturnsAnalysisTool(BaseTool):
    name: str = "Analyze Stock Returns"
    description: str = (
        "Analyzes stock returns for a specific ticker over a date range. "
        "Calculates daily/weekly/monthly returns, cumulative returns, and return statistics. "
        "Use this to understand how a stock has performed over time."
    )
    args_schema: type[BaseModel] = ReturnsAnalysisInput

    def _run(self, ticker: str, start_date: str, end_date: str, period: str = "daily") -> str:
        ticker = ticker.upper()

        # Validate ticker
        if ticker not in get_available_tickers():
            return json.dumps({"error": f"Ticker {ticker} not found in DJ30 dataset"})

        # Get data
        df = get_ticker_data(ticker, start_date, end_date)

        if df.empty or len(df) < 2:
            return json.dumps({"error": f"Insufficient data for {ticker} in date range"})

        # Calculate daily returns
        df['daily_return'] = df['adj_close'].pct_change()

        # Resample if needed
        if period == "weekly":
            df_period = df.set_index('Date').resample('W')['adj_close'].last().pct_change()
            period_label = "Weekly"
        elif period == "monthly":
            df_period = df.set_index('Date').resample('M')['adj_close'].last().pct_change()
            period_label = "Monthly"
        else:
            df_period = df['daily_return']
            period_label = "Daily"

        # Calculate statistics
        returns = df_period.dropna()
        cumulative_return = (1 + returns).prod() - 1

        result = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "period": period,
            "statistics": {
                "cumulative_return": float(cumulative_return * 100),  # %
                "annualized_return": float((np.mean(returns) * 252) * 100) if period == "daily" else float((np.mean(returns) * 52) * 100) if period == "weekly" else float((np.mean(returns) * 12) * 100),  # %
                "mean_return": float(np.mean(returns) * 100),  # %
                "median_return": float(np.median(returns) * 100),  # %
                "std_dev": float(np.std(returns) * 100),  # %
                "min_return": float(np.min(returns) * 100),  # %
                "max_return": float(np.max(returns) * 100),  # %
                "positive_days": int((returns > 0).sum()),
                "negative_days": int((returns < 0).sum()),
                "total_periods": len(returns)
            },
            "summary": f"{period_label} returns for {ticker} from {start_date} to {end_date}:\n" +
                      f"  Cumulative Return: {cumulative_return * 100:.2f}%\n" +
                      f"  Mean {period_label} Return: {np.mean(returns) * 100:.2f}%\n" +
                      f"  Volatility (Std Dev): {np.std(returns) * 100:.2f}%\n" +
                      f"  Best {period_label}: {np.max(returns) * 100:.2f}%\n" +
                      f"  Worst {period_label}: {np.min(returns) * 100:.2f}%"
        }

        return json.dumps(result, indent=2)


class VolatilityAnalysisInput(BaseModel):
    """Input for Volatility Analysis Tool."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    window: int = Field(default=30, description="Rolling window for volatility calculation (days)")


class VolatilityAnalysisTool(BaseTool):
    name: str = "Analyze Stock Volatility"
    description: str = (
        "Analyzes historical volatility for a stock over a date range. "
        "Calculates realized volatility, rolling volatility, and identifies high/low volatility periods. "
        "Use this to understand stock price stability and risk."
    )
    args_schema: type[BaseModel] = VolatilityAnalysisInput

    def _run(self, ticker: str, start_date: str, end_date: str, window: int = 30) -> str:
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

        # Calculate realized volatility (annualized)
        realized_vol = df['daily_return'].std() * np.sqrt(252)

        # Calculate rolling volatility
        df['rolling_vol'] = df['daily_return'].rolling(window=window).std() * np.sqrt(252)

        # Find high/low volatility periods
        df_vol = df.dropna(subset=['rolling_vol'])
        if not df_vol.empty:
            max_vol_idx = df_vol['rolling_vol'].idxmax()
            min_vol_idx = df_vol['rolling_vol'].idxmin()

            max_vol_date = df.loc[max_vol_idx, 'Date'].strftime('%Y-%m-%d')
            max_vol_value = df.loc[max_vol_idx, 'rolling_vol']
            min_vol_date = df.loc[min_vol_idx, 'Date'].strftime('%Y-%m-%d')
            min_vol_value = df.loc[min_vol_idx, 'rolling_vol']
        else:
            max_vol_date = min_vol_date = "N/A"
            max_vol_value = min_vol_value = 0

        result = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "window": window,
            "volatility_metrics": {
                "realized_volatility": float(realized_vol * 100),  # %
                "average_rolling_vol": float(df['rolling_vol'].mean() * 100) if not df['rolling_vol'].isna().all() else 0,  # %
                "max_volatility": {
                    "date": max_vol_date,
                    "value": float(max_vol_value * 100)  # %
                },
                "min_volatility": {
                    "date": min_vol_date,
                    "value": float(min_vol_value * 100)  # %
                }
            },
            "summary": f"Volatility analysis for {ticker} from {start_date} to {end_date}:\n" +
                      f"  Realized Volatility: {realized_vol * 100:.2f}% (annualized)\n" +
                      f"  Average {window}-day Rolling Vol: {df['rolling_vol'].mean() * 100:.2f}%\n" +
                      f"  Highest Volatility: {max_vol_value * 100:.2f}% on {max_vol_date}\n" +
                      f"  Lowest Volatility: {min_vol_value * 100:.2f}% on {min_vol_date}"
        }

        return json.dumps(result, indent=2)


class PerformanceComparisonInput(BaseModel):
    """Input for Performance Comparison Tool."""
    tickers: str = Field(..., description="Comma-separated ticker symbols (e.g., 'AAPL,MSFT,GOOGL')")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    metric: str = Field(default="returns", description="Comparison metric: 'returns', 'volatility', 'sharpe'")


class PerformanceComparisonTool(BaseTool):
    name: str = "Compare Stock Performance"
    description: str = (
        "Compares performance metrics across multiple stocks over a date range. "
        "Can compare returns, volatility, or Sharpe ratios. "
        "Use this to identify best/worst performers or compare investment options."
    )
    args_schema: type[BaseModel] = PerformanceComparisonInput

    def _run(self, tickers: str, start_date: str, end_date: str, metric: str = "returns") -> str:
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

        # Calculate metrics for each ticker
        results = []

        for ticker in ticker_list:
            ticker_df = df[df['ticker'] == ticker].copy()

            if len(ticker_df) < 2:
                continue

            # Calculate daily returns
            ticker_df['daily_return'] = ticker_df['adj_close'].pct_change()
            returns = ticker_df['daily_return'].dropna()

            if len(returns) == 0:
                continue

            # Calculate metrics
            cumulative_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)  # Annualized
            mean_return = returns.mean() * 252  # Annualized
            sharpe_ratio = (mean_return / volatility) if volatility > 0 else 0

            results.append({
                "ticker": ticker,
                "cumulative_return": float(cumulative_return * 100),
                "annualized_return": float(mean_return * 100),
                "volatility": float(volatility * 100),
                "sharpe_ratio": float(sharpe_ratio)
            })

        if not results:
            return json.dumps({"error": "Unable to calculate metrics for any ticker"})

        # Sort by selected metric
        if metric == "returns":
            results.sort(key=lambda x: x['cumulative_return'], reverse=True)
            sort_metric = "cumulative_return"
        elif metric == "volatility":
            results.sort(key=lambda x: x['volatility'])
            sort_metric = "volatility"
        elif metric == "sharpe":
            results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
            sort_metric = "sharpe_ratio"
        else:
            sort_metric = "cumulative_return"

        # Create summary
        summary = f"Performance comparison ({metric}) from {start_date} to {end_date}:\n\n"
        for i, r in enumerate(results, 1):
            summary += f"{i}. {r['ticker']}: "
            if metric == "returns":
                summary += f"Return: {r['cumulative_return']:.2f}%, Volatility: {r['volatility']:.2f}%\n"
            elif metric == "volatility":
                summary += f"Volatility: {r['volatility']:.2f}%, Return: {r['cumulative_return']:.2f}%\n"
            else:
                summary += f"Sharpe: {r['sharpe_ratio']:.2f}, Return: {r['cumulative_return']:.2f}%, Vol: {r['volatility']:.2f}%\n"

        result = {
            "start_date": start_date,
            "end_date": end_date,
            "metric": metric,
            "comparison": results,
            "summary": summary
        }

        return json.dumps(result, indent=2)


class PriceRangeAnalysisInput(BaseModel):
    """Input for Price Range Analysis Tool."""
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")


class PriceRangeAnalysisTool(BaseTool):
    name: str = "Analyze Price Range"
    description: str = (
        "Analyzes price range and trading range for a stock. "
        "Identifies 52-week high/low, average trading range, and current price position. "
        "Use this to understand support/resistance levels and price extremes."
    )
    args_schema: type[BaseModel] = PriceRangeAnalysisInput

    def _run(self, ticker: str, start_date: str, end_date: str) -> str:
        ticker = ticker.upper()

        # Validate ticker
        if ticker not in get_available_tickers():
            return json.dumps({"error": f"Ticker {ticker} not found in DJ30 dataset"})

        # Get data
        df = get_ticker_data(ticker, start_date, end_date)

        if df.empty:
            return json.dumps({"error": f"No data found for {ticker} in date range"})

        # Calculate price range metrics
        max_price = df['high'].max()
        min_price = df['low'].min()
        current_price = df['adj_close'].iloc[-1]
        start_price = df['adj_close'].iloc[0]

        # Find dates for extremes
        max_date = df.loc[df['high'].idxmax(), 'Date'].strftime('%Y-%m-%d')
        min_date = df.loc[df['low'].idxmin(), 'Date'].strftime('%Y-%m-%d')

        # Calculate average daily range
        df['daily_range'] = (df['high'] - df['low']) / df['low'] * 100
        avg_range = df['daily_range'].mean()

        # Current price position in range
        price_position = (current_price - min_price) / (max_price - min_price) * 100 if max_price > min_price else 50

        result = {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "price_metrics": {
                "current_price": float(current_price),
                "start_price": float(start_price),
                "price_change": float((current_price - start_price) / start_price * 100),  # %
                "max_price": float(max_price),
                "max_date": max_date,
                "min_price": float(min_price),
                "min_date": min_date,
                "price_range": float(max_price - min_price),
                "avg_daily_range": float(avg_range),  # %
                "position_in_range": float(price_position)  # %
            },
            "summary": f"Price range analysis for {ticker} from {start_date} to {end_date}:\n" +
                      f"  Current Price: ${current_price:.2f} (started at ${start_price:.2f})\n" +
                      f"  Price Change: {(current_price - start_price) / start_price * 100:+.2f}%\n" +
                      f"  52-Week High: ${max_price:.2f} on {max_date}\n" +
                      f"  52-Week Low: ${min_price:.2f} on {min_date}\n" +
                      f"  Average Daily Range: {avg_range:.2f}%\n" +
                      f"  Current Position: {price_position:.1f}% of range (0%=low, 100%=high)"
        }

        return json.dumps(result, indent=2)

