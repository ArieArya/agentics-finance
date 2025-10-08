"""
Data loading utilities for financial datasets.
Handles loading and caching of macro and market factor data.
"""

import pandas as pd
import os
from typing import Tuple
from functools import lru_cache

# Get the data directory path
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MACRO_FILE = os.path.join(DATA_DIR, "macro_factors_new.csv")
MARKET_FILE = os.path.join(DATA_DIR, "market_factors_new.csv")


@lru_cache(maxsize=1)
def load_macro_factors() -> pd.DataFrame:
    """
    Load macro economic factors data.

    Returns:
        pd.DataFrame: Macro factors with Date as index
    """
    df = pd.read_csv(MACRO_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


@lru_cache(maxsize=1)
def load_market_factors() -> pd.DataFrame:
    """
    Load market factors data.

    Returns:
        pd.DataFrame: Market factors with Date as index
    """
    df = pd.read_csv(MARKET_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both macro and market factors.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (macro_df, market_df)
    """
    return load_macro_factors(), load_market_factors()


def get_data_summary() -> dict:
    """
    Get summary statistics about the datasets.

    Returns:
        dict: Summary information about both datasets
    """
    macro_df = load_macro_factors()
    market_df = load_market_factors()

    return {
        "macro_factors": {
            "columns": list(macro_df.columns),
            "date_range": {
                "start": str(macro_df.index.min()),
                "end": str(macro_df.index.max())
            },
            "rows": len(macro_df)
        },
        "market_factors": {
            "columns": list(market_df.columns),
            "date_range": {
                "start": str(market_df.index.min()),
                "end": str(market_df.index.max())
            },
            "rows": len(market_df)
        }
    }


def get_column_descriptions() -> dict:
    """
    Get human-readable descriptions of all columns.

    Returns:
        dict: Column descriptions for both datasets
    """
    return {
        "macro_factors": {
            "FEDFUNDS": "Federal Funds Effective Rate (%)",
            "TB3MS": "3-Month Treasury Bill Secondary Market Rate (%)",
            "T10Y3M": "10-Year Treasury Minus 3-Month Treasury (%)",
            "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
            "CPILFESL": "Consumer Price Index Less Food & Energy",
            "PCEPI": "Personal Consumption Expenditures Price Index",
            "PCEPILFE": "PCE Price Index Excluding Food and Energy",
            "UNRATE": "Unemployment Rate (%)",
            "PAYEMS": "Total Nonfarm Payroll Employment (thousands)",
            "INDPRO": "Industrial Production Index",
            "RSAFS": "Retail Sales (millions of dollars)"
        },
        "market_factors": {
            "^GSPC": "S&P 500 Index",
            "^STOXX50E": "Euro Stoxx 50 Index",
            "BTC-USD": "Bitcoin Price (USD)",
            "^VIX": "CBOE Volatility Index (VIX)",
            "GSG": "iShares S&P GSCI Commodity-Indexed Trust",
            "DGS2": "2-Year Treasury Constant Maturity Rate (%)",
            "DGS10": "10-Year Treasury Constant Maturity Rate (%)",
            "DTWEXBGS": "Trade Weighted U.S. Dollar Index",
            "DCOILBRENTEU": "Crude Oil Prices: Brent - Europe ($/Barrel)",
            "GLD": "SPDR Gold Trust (Gold ETF)",
            "US10Y2Y": "10-Year minus 2-Year Treasury Yield Spread (%)",
            "Headlines": "News headlines for the date"
        }
    }

