"""
Utility for loading and caching firm-level fundamental data.
"""

import pandas as pd
import os
from functools import lru_cache


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FIRM_DATA_FILE = os.path.join(DATA_DIR, "firm_summary.csv")


@lru_cache(maxsize=1)
def load_firm_data() -> pd.DataFrame:
    """
    Load firm fundamental data from CSV.

    Returns:
        DataFrame with columns:
        - TICKER: Company ticker symbol
        - STATPERS: Statement period (date)
        - PRICE: Stock price
        - EBS: Earnings Before Shares (Operating Income)
        - EPS: Earnings Per Share
        - DPS: Dividends Per Share
        - ROA: Return on Assets (%)
        - ROE: Return on Equity (%)
        - NAV: Net Asset Value
        - GRM: Gross Margin (%)
        - FVYRGRO_*: Forward 1-year growth estimates
        - FVYSTA_*: Forward 1-year volatility estimates
    """
    df = pd.read_csv(FIRM_DATA_FILE)

    # Convert STATPERS to datetime and set as index
    df['STATPERS'] = pd.to_datetime(df['STATPERS'])

    # Sort by ticker and date
    df = df.sort_values(['TICKER', 'STATPERS'])

    return df


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
    company_df = df[df['TICKER'] == ticker].copy()

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
    result['TICKER'] = ticker
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
        'ticker': ticker,
        'price': price,
        'date': data['STATPERS']
    }

    # P/E Ratio
    if data.get('EPS') and data['EPS'] > 0:
        metrics['pe_ratio'] = price / data['EPS']

    # P/B Ratio (Price to Net Asset Value)
    if data.get('NAV') and data['NAV'] > 0:
        metrics['pb_ratio'] = price / data['NAV']

    # Dividend Yield
    if data.get('DPS') and data['DPS'] > 0:
        metrics['dividend_yield'] = (data['DPS'] / price) * 100

    # Forward P/E (if forward EPS available)
    if data.get('EPS') and data.get('FVYRGRO_EPS'):
        forward_eps = data['EPS'] * (1 + data['FVYRGRO_EPS'] / 100)
        if forward_eps > 0:
            metrics['forward_pe'] = price / forward_eps

    # Copy fundamental metrics
    for key in ['EPS', 'DPS', 'ROA', 'ROE', 'GRM', 'EBS', 'NAV']:
        if data.get(key) is not None:
            metrics[key.lower()] = data[key]

    # Copy growth metrics
    for key in ['FVYRGRO_EPS', 'FVYRGRO_ROE', 'FVYRGRO_ROA']:
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
        "FVYRGRO_EBS": "1-year forward EBS growth estimate (%)",
        "FVYRGRO_EPS": "1-year forward EPS growth estimate (%)",
        "FVYRGRO_DPS": "1-year forward DPS growth estimate (%)",
        "FVYRGRO_ROA": "1-year forward ROA growth estimate (%)",
        "FVYRGRO_ROE": "1-year forward ROE growth estimate (%)",
        "FVYRGRO_NAV": "1-year forward NAV growth estimate (%)",
        "FVYRGRO_GRM": "1-year forward GRM growth estimate (%)",
        "FVYSTA_EBS": "1-year forward EBS volatility estimate",
        "FVYSTA_EPS": "1-year forward EPS volatility estimate",
        "FVYSTA_DPS": "1-year forward DPS volatility estimate",
        "FVYSTA_ROA": "1-year forward ROA volatility estimate",
        "FVYSTA_ROE": "1-year forward ROE volatility estimate",
        "FVYSTA_NAV": "1-year forward NAV volatility estimate",
        "FVYSTA_GRM": "1-year forward GRM volatility estimate"
    }



