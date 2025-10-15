"""
Data loader for DJ30 stock price data.
Provides loading, caching, and basic access functions for DJ30 price time series.
"""

import pandas as pd
import os
from functools import lru_cache
from datetime import datetime
from typing import Optional, List

DJ30_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dj30_data_full.csv")

@lru_cache(maxsize=1)
def load_dj30_data():
    """Load and preprocess DJ30 price data with caching."""
    if not os.path.exists(DJ30_DATA_PATH):
        raise FileNotFoundError(f"DJ30 data file not found: {DJ30_DATA_PATH}")

    df = pd.read_csv(DJ30_DATA_PATH)

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by ticker and date
    df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    return df


def get_available_tickers():
    """Get list of available tickers in DJ30 dataset."""
    df = load_dj30_data()
    return sorted(df['ticker'].unique().tolist())


def get_ticker_data(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Get price data for a specific ticker within date range.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)

    Returns:
        DataFrame with price data for the ticker
    """
    df = load_dj30_data()

    # Filter by ticker
    ticker_df = df[df['ticker'] == ticker].copy()

    if ticker_df.empty:
        return pd.DataFrame()

    # Filter by date range if provided
    if start_date:
        ticker_df = ticker_df[ticker_df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        ticker_df = ticker_df[ticker_df['Date'] <= pd.to_datetime(end_date)]

    return ticker_df.reset_index(drop=True)


def get_multiple_tickers_data(tickers: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Get price data for multiple tickers within date range.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)

    Returns:
        DataFrame with price data for all specified tickers
    """
    df = load_dj30_data()

    # Filter by tickers
    tickers_df = df[df['ticker'].isin(tickers)].copy()

    # Filter by date range if provided
    if start_date:
        tickers_df = tickers_df[tickers_df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        tickers_df = tickers_df[tickers_df['Date'] <= pd.to_datetime(end_date)]

    return tickers_df.reset_index(drop=True)


def get_sector_tickers(sector: str):
    """Get list of tickers in a specific sector."""
    df = load_dj30_data()
    sector_tickers = df[df['sector'] == sector]['ticker'].unique().tolist()
    return sorted(sector_tickers)


def get_all_sectors():
    """Get list of all sectors in DJ30 dataset."""
    df = load_dj30_data()
    return sorted(df['sector'].unique().tolist())


def get_ticker_info(ticker: str):
    """Get basic information about a ticker (sector, industry, etc.)."""
    df = load_dj30_data()
    ticker_df = df[df['ticker'] == ticker]

    if ticker_df.empty:
        return {}

    # Get the most recent row for this ticker
    latest = ticker_df.iloc[-1]

    return {
        "ticker": ticker,
        "sector": latest['sector'],
        "industry": latest['industry'],
        "currency": latest['currency'],
        "shares": latest['shares']
    }


def get_date_range():
    """Get the date range of available data."""
    df = load_dj30_data()
    return {
        "start": df['Date'].min().strftime('%Y-%m-%d'),
        "end": df['Date'].max().strftime('%Y-%m-%d')
    }


def get_dj30_data_summary():
    """Get summary of DJ30 dataset for display."""
    df = load_dj30_data()
    date_range = get_date_range()
    tickers = get_available_tickers()
    sectors = get_all_sectors()

    return f"""
**DJ30 Stock Price Data:**
- **Tickers**: {len(tickers)} DJ30 companies ({', '.join(tickers[:10])}...)
- **Date Range**: {date_range['start']} to {date_range['end']}
- **Total Records**: {len(df):,} daily price observations
- **Sectors**: {len(sectors)} sectors ({', '.join(sectors)})
- **Data Fields**: OHLCV (Open, High, Low, Close, Volume), Adjusted Close, Dividends, Splits
"""


def get_dj30_column_descriptions():
    """Get descriptions of DJ30 data columns."""
    return {
        "Price Data": [
            "**Date**: Trading date",
            "**open**: Opening price",
            "**high**: Highest price of the day",
            "**low**: Lowest price of the day",
            "**close**: Closing price",
            "**adj_close**: Adjusted closing price (accounts for splits/dividends)",
            "**volume**: Number of shares traded"
        ],
        "Corporate Actions": [
            "**dividend**: Dividend paid on this date",
            "**split_ratio**: Stock split ratio",
            "**shares**: Outstanding shares"
        ],
        "Fundamental Metrics": [
            "**forwardPE**: Forward P/E ratio",
            "**priceToBook**: Price-to-Book ratio",
            "**dividendYield**: Dividend yield (%)",
            "**returnOnEquity**: Return on Equity (ROE) (%)",
            "**enterpriseValue**: Enterprise value",
            "**enterpriseToEbitda**: EV/EBITDA ratio"
        ],
        "Classification": [
            "**ticker**: Stock ticker symbol",
            "**sector**: Business sector",
            "**industry**: Specific industry",
            "**currency**: Trading currency"
        ]
    }

