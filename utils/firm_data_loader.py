"""
Utility for loading and caching firm-level fundamental data from merged dataset.
"""

import pandas as pd
import os
from functools import lru_cache
import re
from .csv_reader import read_merged_data_csv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Fundamental metric prefixes (without ticker suffix)
FUNDAMENTAL_METRICS = [
    'BPS', 'CPS', 'CPX', 'CSH', 'DPS', 'EBS', 'EPS', 'GRM',
    'NAV', 'NDT', 'NET', 'ROA', 'ROE', 'SAL'
]

# Suffixes that indicate data type
DATA_SUFFIXES = ['_MEDEST', '_MEANEST', '_ACTUAL']


@lru_cache(maxsize=1)
def _load_merged_data() -> pd.DataFrame:
    """Load the merged dataset (cached)."""
    df = read_merged_data_csv(DATA_DIR)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


@lru_cache(maxsize=1)
def load_firm_data() -> pd.DataFrame:
    """
    Load firm fundamental data from merged dataset, converting from wide to long format.

    Returns:
        DataFrame with columns:
        - TICKER: Company ticker symbol
        - STATPERS: Statement period (date)
        - PRICE: Stock price (from close_* columns)
        - EBS: Earnings Before Shares (Operating Income)
        - EPS: Earnings Per Share
        - DPS: Dividends Per Share
        - ROA: Return on Assets (%)
        - ROE: Return on Equity (%)
        - NAV: Net Asset Value
        - GRM: Gross Margin (%)
        - BPS: Book Value Per Share
        - CPS: Cash Per Share
        - SAL: Sales
        - NET: Net Income
    """
    df = _load_merged_data()

    # Extract all tickers from fundamental columns
    all_tickers = set()
    for col in df.columns:
        for metric in FUNDAMENTAL_METRICS:
            for suffix in DATA_SUFFIXES:
                pattern = f"^{metric}_([A-Z]+){suffix}$"
                match = re.match(pattern, col)
                if match:
                    all_tickers.add(match.group(1))

    if not all_tickers:
        return pd.DataFrame(columns=['TICKER', 'STATPERS'])

    # Collect all rows for long format
    long_data = []

    for ticker in sorted(all_tickers):
        # Extract fundamental columns for this ticker
        ticker_metrics = {}

        # Map metric names (use MEDEST as primary, fallback to ACTUAL)
        metric_mapping = {
            'EPS': 'EPS',
            'DPS': 'DPS',
            'ROA': 'ROA',
            'ROE': 'ROE',
            'NAV': 'NAV',
            'GRM': 'GRM',
            'EBS': 'EBS',
            'BPS': 'BPS',
            'CPS': 'CPS',
            'SAL': 'SAL',
            'NET': 'NET'
        }

        for metric, col_name in metric_mapping.items():
            # Try MEDEST first, then ACTUAL
            for suffix in ['_MEDEST', '_ACTUAL']:
                col = f"{metric}_{ticker}{suffix}"
                if col in df.columns:
                    ticker_metrics[col_name] = col
                    break

        # Also get price from close column
        price_col = f"close_{ticker}"
        if price_col in df.columns:
            ticker_metrics['PRICE'] = price_col

        if not ticker_metrics:
            continue

        # Create dataframe for this ticker
        ticker_df = df[['Date'] + list(ticker_metrics.values())].copy()

        # Rename columns
        rename_dict = {v: k for k, v in ticker_metrics.items()}
        ticker_df = ticker_df.rename(columns=rename_dict)
        ticker_df = ticker_df.rename(columns={'Date': 'STATPERS'})

        # Add ticker column
        ticker_df['TICKER'] = ticker

        # Drop rows where all fundamental columns are NaN
        fundamental_cols = [k for k in ticker_metrics.keys() if k != 'PRICE']
        ticker_df = ticker_df.dropna(subset=fundamental_cols, how='all')

        if not ticker_df.empty:
            long_data.append(ticker_df)

    if not long_data:
        return pd.DataFrame(columns=['TICKER', 'STATPERS'])

    # Combine all tickers
    result_df = pd.concat(long_data, ignore_index=True)

    # Sort by ticker and date
    result_df = result_df.sort_values(['TICKER', 'STATPERS']).reset_index(drop=True)

    return result_df


@lru_cache(maxsize=128)
def get_company_data(ticker: str) -> pd.DataFrame:
    """
    Get all data for a specific company.

    Args:
        ticker: Company ticker symbol

    Returns:
        DataFrame with all periods for that company
    """
    df = load_firm_data()
    company_df = df[df['TICKER'] == ticker.upper()].copy()

    if company_df.empty:
        return pd.DataFrame()

    # Set date as index
    company_df = company_df.set_index('STATPERS')

    return company_df


def get_available_tickers() -> list:
    """
    Get list of all available company tickers.

    Returns:
        List of ticker symbols
    """
    df = load_firm_data()
    if df.empty:
        return []
    return sorted(df['TICKER'].unique().tolist())


def get_latest_data(ticker: str, date: str = None) -> dict:
    """
    Get most recent fundamental data for a company.

    Args:
        ticker: Company ticker symbol
        date: Optional date to get data as of (YYYY-MM-DD). If None, gets latest available.

    Returns:
        Dictionary with latest fundamental metrics
    """
    company_df = get_company_data(ticker)

    if company_df.empty:
        return {}

    if date:
        # Get data as of specified date (last available before that date)
        date_dt = pd.to_datetime(date)
        filtered = company_df[company_df.index <= date_dt]
        if filtered.empty:
            return {}
        latest = filtered.iloc[-1]
    else:
        # Get most recent data
        latest = company_df.iloc[-1]

    # Convert to dictionary, handling NaN values
    result = latest.to_dict()
    result['TICKER'] = ticker.upper()
    result['STATPERS'] = str(latest.name.date())

    # Clean up NaN values
    result = {k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float)) else v)
              for k, v in result.items()}

    return result


def get_company_fundamentals_history(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Get historical fundamental data for a company within a date range.

    Args:
        ticker: Company ticker symbol
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        DataFrame with historical fundamentals
    """
    company_df = get_company_data(ticker)

    if company_df.empty:
        return pd.DataFrame()

    # Filter by date range
    if start_date:
        company_df = company_df[company_df.index >= pd.to_datetime(start_date)]
    if end_date:
        company_df = company_df[company_df.index <= pd.to_datetime(end_date)]

    return company_df


def get_multiple_companies_latest(tickers: list, date: str = None) -> pd.DataFrame:
    """
    Get latest data for multiple companies for comparison.

    Args:
        tickers: List of company ticker symbols
        date: Optional date to get data as of (YYYY-MM-DD)

    Returns:
        DataFrame with latest data for each company
    """
    data = []
    for ticker in tickers:
        ticker_data = get_latest_data(ticker, date)
        if ticker_data:
            data.append(ticker_data)

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def calculate_valuation_metrics(ticker: str, date: str = None) -> dict:
    """
    Calculate valuation metrics for a company.

    Args:
        ticker: Company ticker symbol
        date: Optional date to calculate as of (YYYY-MM-DD)

    Returns:
        Dictionary with valuation metrics (P/E, P/B, dividend yield, etc.)
    """
    data = get_latest_data(ticker, date)

    if not data or not data.get('PRICE'):
        return {}

    price = data['PRICE']
    metrics = {
        'ticker': ticker.upper(),
        'price': price,
        'date': data['STATPERS']
    }

    # P/E Ratio
    if data.get('EPS') and data['EPS'] > 0:
        metrics['pe_ratio'] = price / data['EPS']

    # P/B Ratio (Price to Net Asset Value)
    if data.get('NAV') and data['NAV'] > 0:
        metrics['pb_ratio'] = price / data['NAV']
    elif data.get('BPS') and data['BPS'] > 0:
        metrics['pb_ratio'] = price / data['BPS']

    # Dividend Yield
    if data.get('DPS') and data['DPS'] > 0:
        metrics['dividend_yield'] = (data['DPS'] / price) * 100

    # Copy fundamental metrics
    for key in ['EPS', 'DPS', 'ROA', 'ROE', 'GRM', 'EBS', 'NAV', 'BPS', 'CPS', 'SAL', 'NET']:
        if data.get(key) is not None:
            metrics[key.lower()] = data[key]

    return metrics


def get_firm_data_summary() -> dict:
    """
    Get summary information about the firm dataset.

    Returns:
        Dictionary with dataset statistics
    """
    df = load_firm_data()

    if df.empty:
        return {
            "total_records": 0,
            "unique_tickers": 0,
            "date_range": {"start": None, "end": None},
            "tickers": [],
            "columns": []
        }

    return {
        "total_records": len(df),
        "unique_tickers": df['TICKER'].nunique(),
        "date_range": {
            "start": str(df['STATPERS'].min().date()),
            "end": str(df['STATPERS'].max().date())
        },
        "tickers": sorted(df['TICKER'].unique().tolist()),
        "columns": df.columns.tolist()
    }


def get_column_descriptions() -> dict:
    """
    Get descriptions of all columns in firm dataset.

    Returns:
        Dictionary mapping column names to descriptions
    """
    return {
        "TICKER": "Company ticker symbol",
        "STATPERS": "Statement period (monthly frequency)",
        "PRICE": "Stock price",
        "EBS": "Earnings Before Shares (Operating Income/EBIT)",
        "EPS": "Earnings Per Share",
        "DPS": "Dividends Per Share",
        "ROA": "Return on Assets (%)",
        "ROE": "Return on Equity (%)",
        "NAV": "Net Asset Value per share",
        "GRM": "Gross Margin (%)",
        "BPS": "Book Value Per Share",
        "CPS": "Cash Per Share",
        "SAL": "Sales",
        "NET": "Net Income"
    }
