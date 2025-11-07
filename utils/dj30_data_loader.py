"""
Data loader for DJ30 stock price data from merged dataset.
Provides loading, caching, and basic access functions for DJ30 price time series.
"""

import pandas as pd
import os
from functools import lru_cache
from datetime import datetime
from typing import Optional, List
import re
from .csv_reader import read_merged_data_csv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# DJ30 tickers (30 Dow Jones companies)
DJ30_TICKERS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

# Price column prefixes
PRICE_COLUMNS = ['open', 'high', 'low', 'close', 'adj_close', 'volume',
                 'dividend', 'dividendyield', 'enterprisetoebitda', 'enterprisevalue',
                 'forwardpe', 'industry', 'pricetobook', 'returnonequity', 'sector',
                 'shares', 'split_ratio']


@lru_cache(maxsize=1)
def _load_merged_data():
    """Load the merged dataset (cached)."""
    df = read_merged_data_csv(DATA_DIR)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


@lru_cache(maxsize=1)
def load_dj30_data():
    """Load and preprocess DJ30 price data from merged dataset, converting from wide to long format."""
    df = _load_merged_data()

    # Collect all rows for long format
    long_data = []

    for ticker in DJ30_TICKERS:
        # Extract columns for this ticker
        ticker_cols = {}
        for prefix in PRICE_COLUMNS:
            col_name = f"{prefix}_{ticker}"
            if col_name in df.columns:
                ticker_cols[prefix] = col_name

        if not ticker_cols:
            continue

        # Create a dataframe for this ticker
        ticker_df = df[['Date'] + list(ticker_cols.values())].copy()

        # Rename columns to remove ticker suffix
        rename_dict = {v: k for k, v in ticker_cols.items()}
        ticker_df = ticker_df.rename(columns=rename_dict)

        # Add ticker column
        ticker_df['ticker'] = ticker

        # Get sector/industry from any available column (they should be the same across dates)
        sector_col = f"sector_{ticker}"
        industry_col = f"industry_{ticker}"
        if sector_col in df.columns:
            ticker_df['sector'] = df[sector_col].iloc[0] if len(df) > 0 else None
        if industry_col in df.columns:
            ticker_df['industry'] = df[industry_col].iloc[0] if len(df) > 0 else None

        # Add currency (default to USD)
        ticker_df['currency'] = 'USD'

        # Drop rows where all price columns are NaN
        price_cols = ['open', 'high', 'low', 'close']
        ticker_df = ticker_df.dropna(subset=price_cols, how='all')

        if not ticker_df.empty:
            long_data.append(ticker_df)

    if not long_data:
        return pd.DataFrame(columns=['Date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume'])

    # Combine all tickers
    result_df = pd.concat(long_data, ignore_index=True)

    # Sort by ticker and date
    result_df = result_df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    return result_df


def get_available_tickers():
    """Get list of available tickers in DJ30 dataset."""
    df = load_dj30_data()
    if df.empty:
        return []
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
    ticker_df = df[df['ticker'] == ticker.upper()].copy()

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
    tickers_df = df[df['ticker'].isin([t.upper() for t in tickers])].copy()

    # Filter by date range if provided
    if start_date:
        tickers_df = tickers_df[tickers_df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        tickers_df = tickers_df[tickers_df['Date'] <= pd.to_datetime(end_date)]

    return tickers_df.reset_index(drop=True)


def get_sector_tickers(sector: str):
    """Get list of tickers in a specific sector."""
    df = load_dj30_data()
    if df.empty or 'sector' not in df.columns:
        return []
    sector_tickers = df[df['sector'] == sector]['ticker'].unique().tolist()
    return sorted(sector_tickers)


def get_all_sectors():
    """Get list of all sectors in DJ30 dataset."""
    df = load_dj30_data()
    if df.empty or 'sector' not in df.columns:
        return []
    return sorted(df['sector'].dropna().unique().tolist())


def get_ticker_info(ticker: str):
    """Get basic information about a ticker (sector, industry, etc.)."""
    df = load_dj30_data()
    ticker_df = df[df['ticker'] == ticker.upper()]

    if ticker_df.empty:
        return {}

    # Get the most recent row for this ticker
    latest = ticker_df.iloc[-1]

    return {
        "ticker": ticker.upper(),
        "sector": latest.get('sector', None),
        "industry": latest.get('industry', None),
        "currency": latest.get('currency', 'USD'),
        "shares": latest.get('shares', None)
    }


def get_date_range():
    """Get the date range of available data."""
    df = load_dj30_data()
    if df.empty:
        return {"start": None, "end": None}
    return {
        "start": df['Date'].min().strftime('%Y-%m-%d'),
        "end": df['Date'].max().strftime('%Y-%m-%d')
    }


def get_dj30_data_summary():
    """Get summary of DJ30 dataset for display."""
    df = load_dj30_data()
    if df.empty:
        return "**DJ30 Stock Price Data:** No data available"

    date_range = get_date_range()
    tickers = get_available_tickers()
    sectors = get_all_sectors()

    return f"""
**DJ30 Stock Price Data:**
- **Tickers**: {len(tickers)} DJ30 companies ({', '.join(tickers[:10])}...)
- **Date Range**: {date_range['start']} to {date_range['end']}
- **Total Records**: {len(df):,} daily price observations
- **Sectors**: {len(sectors)} sectors ({', '.join(sectors) if sectors else 'N/A'})
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
            "**forwardpe**: Forward P/E ratio",
            "**pricetobook**: Price-to-Book ratio",
            "**dividendyield**: Dividend yield (%)",
            "**returnonequity**: Return on Equity (ROE) (%)",
            "**enterprisevalue**: Enterprise value",
            "**enterprisetoebitda**: EV/EBITDA ratio"
        ],
        "Classification": [
            "**ticker**: Stock ticker symbol",
            "**sector**: Business sector",
            "**industry**: Specific industry",
            "**currency**: Trading currency"
        ]
    }
