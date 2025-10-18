"""
Tools for analyzing company fundamental data.
"""

from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import json
import pandas as pd
from utils.firm_data_loader import (
    get_company_data, get_latest_data, get_available_tickers,
    get_multiple_companies_latest, calculate_valuation_metrics,
    get_company_fundamentals_history
)


class CompanyFundamentalsInput(BaseModel):
    """Input schema for CompanyFundamentalsQueryTool."""
    ticker: str = Field(..., description="Company ticker symbol (e.g., AAPL, MSFT)")
    date: Optional[str] = Field(None, description="Optional date to get data as of (YYYY-MM-DD). If not provided, returns latest available data.")


class CompanyFundamentalsQueryTool(BaseTool):
    name: str = "Query Company Fundamentals"
    description: str = (
        "Retrieves fundamental financial data for a specific company, including: "
        "Price, EPS, DPS, ROA, ROE, NAV, Gross Margin, and forward growth estimates. "
        "Use this to get current or historical fundamental data for a single company. "
        "Returns metrics like P/E ratio, dividend yield, profitability metrics, and valuation data."
    )
    args_schema: Type[BaseModel] = CompanyFundamentalsInput

    def _run(self, ticker: str, date: Optional[str] = None) -> str:
        try:
            # Get fundamental data
            data = get_latest_data(ticker.upper(), date)

            if not data:
                return json.dumps({
                    "success": False,
                    "error": f"No data found for ticker {ticker}. Available tickers: {', '.join(get_available_tickers()[:10])}..."
                })

            # Calculate valuation metrics
            valuation = calculate_valuation_metrics(ticker.upper(), date)

            # Combine data
            result = {
                "success": True,
                "ticker": ticker.upper(),
                "date": data['STATPERS'],
                "price": data.get('PRICE'),
                "fundamentals": {
                    "eps": data.get('EPS'),
                    "dps": data.get('DPS'),
                    "ebs": data.get('EBS'),
                    "roa": data.get('ROA'),
                    "roe": data.get('ROE'),
                    "nav": data.get('NAV'),
                    "gross_margin": data.get('GRM')
                },
                "valuation": {
                    "pe_ratio": valuation.get('pe_ratio'),
                    "pb_ratio": valuation.get('pb_ratio'),
                    "dividend_yield": valuation.get('dividend_yield'),
                    "forward_pe": valuation.get('forward_pe')
                },
                "growth_estimates": {
                    "eps_growth": data.get('FVYRGRO_EPS'),
                    "roe_growth": data.get('FVYRGRO_ROE'),
                    "roa_growth": data.get('FVYRGRO_ROA')
                }
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class CompareFundamentalsInput(BaseModel):
    """Input schema for CompareFundamentalsTool."""
    tickers: str = Field(..., description="Comma-separated list of ticker symbols to compare (e.g., 'AAPL,MSFT,GOOGL')")
    date: Optional[str] = Field(None, description="Optional date to compare as of (YYYY-MM-DD)")


class CompareFundamentalsTool(BaseTool):
    name: str = "Compare Company Fundamentals"
    description: str = (
        "Compares fundamental metrics across multiple companies side-by-side. "
        "Returns P/E ratios, ROE, ROA, EPS, gross margins, and other key metrics for comparison. "
        "Use this to evaluate relative valuation and performance across companies. "
        "Supports comparing 2-10 companies at once."
    )
    args_schema: Type[BaseModel] = CompareFundamentalsInput

    def _run(self, tickers: str, date: Optional[str] = None) -> str:
        try:
            ticker_list = [t.strip().upper() for t in tickers.split(',')]

            if len(ticker_list) < 2:
                return json.dumps({
                    "success": False,
                    "error": "Need at least 2 tickers to compare"
                })

            if len(ticker_list) > 10:
                return json.dumps({
                    "success": False,
                    "error": "Maximum 10 companies can be compared at once"
                })

            # Get data for all companies
            comparison_data = []
            for ticker in ticker_list:
                data = get_latest_data(ticker, date)
                if data:
                    valuation = calculate_valuation_metrics(ticker, date)
                    comparison_data.append({
                        "ticker": ticker,
                        "date": data['STATPERS'],
                        "price": data.get('PRICE'),
                        "eps": data.get('EPS'),
                        "pe_ratio": valuation.get('pe_ratio'),
                        "pb_ratio": valuation.get('pb_ratio'),
                        "roa": data.get('ROA'),
                        "roe": data.get('ROE'),
                        "gross_margin": data.get('GRM'),
                        "dividend_yield": valuation.get('dividend_yield'),
                        "eps_growth": data.get('FVYRGRO_EPS'),
                        "roe_growth": data.get('FVYRGRO_ROE')
                    })

            if not comparison_data:
                return json.dumps({
                    "success": False,
                    "error": f"No data found for any of the tickers: {tickers}"
                })

            return json.dumps({
                "success": True,
                "companies": comparison_data,
                "count": len(comparison_data)
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class ScreenCompaniesInput(BaseModel):
    """Input schema for ScreenCompaniesTool."""
    min_roe: Optional[float] = Field(None, description="Minimum ROE threshold (%)")
    min_roa: Optional[float] = Field(None, description="Minimum ROA threshold (%)")
    min_gross_margin: Optional[float] = Field(None, description="Minimum gross margin threshold (%)")
    max_pe_ratio: Optional[float] = Field(None, description="Maximum P/E ratio threshold")
    min_eps_growth: Optional[float] = Field(None, description="Minimum EPS growth estimate (%)")
    date: Optional[str] = Field(None, description="Date to screen as of (YYYY-MM-DD)")


class ScreenCompaniesTool(BaseTool):
    name: str = "Screen Companies by Fundamentals"
    description: str = (
        "Screens all available companies based on fundamental criteria. "
        "Use this to find companies that meet specific financial requirements. "
        "Supports filtering by ROE, ROA, gross margin, P/E ratio, and EPS growth. "
        "Returns list of companies that pass all specified criteria. "
        "Useful for identifying investment opportunities or building watchlists."
    )
    args_schema: Type[BaseModel] = ScreenCompaniesInput

    def _run(self, min_roe: Optional[float] = None, min_roa: Optional[float] = None,
             min_gross_margin: Optional[float] = None, max_pe_ratio: Optional[float] = None,
             min_eps_growth: Optional[float] = None, date: Optional[str] = None) -> str:
        try:
            tickers = get_available_tickers()
            passed_companies = []

            for ticker in tickers:
                data = get_latest_data(ticker, date)
                if not data or not data.get('PRICE'):
                    continue

                valuation = calculate_valuation_metrics(ticker, date)

                # Apply filters
                if min_roe is not None and (data.get('ROE') is None or data['ROE'] < min_roe):
                    continue
                if min_roa is not None and (data.get('ROA') is None or data['ROA'] < min_roa):
                    continue
                if min_gross_margin is not None and (data.get('GRM') is None or data['GRM'] < min_gross_margin):
                    continue
                if max_pe_ratio is not None and (valuation.get('pe_ratio') is None or valuation['pe_ratio'] > max_pe_ratio):
                    continue
                if min_eps_growth is not None and (data.get('FVYRGRO_EPS') is None or data['FVYRGRO_EPS'] < min_eps_growth):
                    continue

                # Passed all filters
                passed_companies.append({
                    "ticker": ticker,
                    "price": data.get('PRICE'),
                    "roe": data.get('ROE'),
                    "roa": data.get('ROA'),
                    "gross_margin": data.get('GRM'),
                    "pe_ratio": valuation.get('pe_ratio'),
                    "eps_growth": data.get('FVYRGRO_EPS'),
                    "eps": data.get('EPS')
                })

            # Sort by ROE (descending)
            passed_companies.sort(key=lambda x: x.get('roe') or 0, reverse=True)

            return json.dumps({
                "success": True,
                "companies": passed_companies,
                "count": len(passed_companies),
                "criteria": {
                    "min_roe": min_roe,
                    "min_roa": min_roa,
                    "min_gross_margin": min_gross_margin,
                    "max_pe_ratio": max_pe_ratio,
                    "min_eps_growth": min_eps_growth
                }
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class CompanyValuationInput(BaseModel):
    """Input schema for CompanyValuationTool."""
    ticker: str = Field(..., description="Company ticker symbol")
    date: Optional[str] = Field(None, description="Date to calculate valuation as of (YYYY-MM-DD)")


class CompanyValuationTool(BaseTool):
    name: str = "Calculate Company Valuation Metrics"
    description: str = (
        "Calculates comprehensive valuation metrics for a company including: "
        "P/E ratio, P/B ratio, dividend yield, forward P/E, and relative valuation analysis. "
        "Use this for detailed valuation assessment of a single company. "
        "Returns both absolute metrics and context for interpretation."
    )
    args_schema: Type[BaseModel] = CompanyValuationInput

    def _run(self, ticker: str, date: Optional[str] = None) -> str:
        try:
            ticker = ticker.upper()
            valuation = calculate_valuation_metrics(ticker, date)

            if not valuation:
                return json.dumps({
                    "success": False,
                    "error": f"Unable to calculate valuation metrics for {ticker}"
                })

            # Add interpretation context
            result = {
                "success": True,
                "ticker": ticker,
                "date": valuation.get('date'),
                "price": valuation.get('price'),
                "valuation_metrics": {
                    "pe_ratio": valuation.get('pe_ratio'),
                    "pb_ratio": valuation.get('pb_ratio'),
                    "forward_pe": valuation.get('forward_pe'),
                    "dividend_yield": valuation.get('dividend_yield')
                },
                "fundamentals": {
                    "eps": valuation.get('eps'),
                    "nav": valuation.get('nav'),
                    "roe": valuation.get('roe'),
                    "roa": valuation.get('roa'),
                    "gross_margin": valuation.get('grm')
                },
                "growth_metrics": {
                    "eps_growth_fwd": valuation.get('fvyrgro_eps'),
                    "roe_growth_fwd": valuation.get('fvyrgro_roe'),
                    "roa_growth_fwd": valuation.get('fvyrgro_roa')
                }
            }

            # Add interpretation notes
            notes = []
            if valuation.get('pe_ratio'):
                pe = valuation['pe_ratio']
                if pe < 15:
                    notes.append("Low P/E ratio may indicate undervaluation or low growth expectations")
                elif pe > 30:
                    notes.append("High P/E ratio may indicate overvaluation or high growth expectations")

            if valuation.get('roe'):
                roe = valuation['roe']
                if roe > 20:
                    notes.append("Strong ROE (>20%) indicates excellent profitability")
                elif roe < 10:
                    notes.append("Low ROE (<10%) may indicate weak profitability")

            if notes:
                result['interpretation_notes'] = notes

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


class FundamentalHistoryInput(BaseModel):
    """Input schema for FundamentalHistoryTool."""
    ticker: str = Field(..., description="Company ticker symbol")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    metrics: Optional[str] = Field(None, description="Comma-separated list of metrics to retrieve (e.g., 'EPS,ROE,ROA'). If not specified, returns key metrics.")


class FundamentalHistoryTool(BaseTool):
    name: str = "Get Fundamental History"
    description: str = (
        "Retrieves historical fundamental data for a company over time. "
        "Use this to analyze trends in EPS, ROE, ROA, margins, and other metrics. "
        "Returns time series of fundamental data for trend analysis and visualization. "
        "Useful for tracking company performance evolution."
    )
    args_schema: Type[BaseModel] = FundamentalHistoryInput

    def _run(self, ticker: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None, metrics: Optional[str] = None) -> str:
        try:
            ticker = ticker.upper()
            history = get_company_fundamentals_history(ticker, start_date, end_date)

            if history.empty:
                return json.dumps({
                    "success": False,
                    "error": f"No historical data found for {ticker}"
                })

            # Select metrics to return
            if metrics:
                metric_list = [m.strip().upper() for m in metrics.split(',')]
                # Always include date and price
                available_metrics = ['PRICE'] + [m for m in metric_list if m in history.columns]
            else:
                # Default key metrics
                available_metrics = ['PRICE', 'EPS', 'ROE', 'ROA', 'GRM']
                available_metrics = [m for m in available_metrics if m in history.columns]

            # Prepare summary data (not full time series to avoid token overload)
            summary = {
                "ticker": ticker,
                "date_range": {
                    "start": str(history.index.min().date()),
                    "end": str(history.index.max().date())
                },
                "periods": len(history),
                "metrics": {}
            }

            for metric in available_metrics:
                series = history[metric].dropna()
                if len(series) > 0:
                    summary["metrics"][metric] = {
                        "start_value": float(series.iloc[0]),
                        "end_value": float(series.iloc[-1]),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": float(series.mean()),
                        "change": float(series.iloc[-1] - series.iloc[0]),
                        "pct_change": float(((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100) if series.iloc[0] != 0 else None
                    }

            return json.dumps({
                "success": True,
                **summary
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })


