"""
Tools for portfolio analysis and recommendations.
"""

from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import json
import pandas as pd
import numpy as np
from utils.firm_data_loader import (
    get_available_tickers, get_latest_data, calculate_valuation_metrics,
    get_company_fundamentals_history
)
from utils.data_loader import load_macro_factors, load_market_factors


class PortfolioRecommendationInput(BaseModel):
    """Input schema for PortfolioRecommendationTool."""
    num_long: int = Field(5, description="Number of companies to recommend for long positions")
    num_short: int = Field(5, description="Number of companies to recommend for short positions")
    date: Optional[str] = Field(None, description="Date to generate recommendations as of (YYYY-MM-DD)")
    strategy: Optional[str] = Field("balanced", description="Strategy type: 'value', 'growth', 'quality', or 'balanced'")


class PortfolioRecommendationTool(BaseTool):
    name: str = "Generate Portfolio Recommendations"
    description: str = (
        "Generates long/short portfolio recommendations based on comprehensive analysis of: "
        "1) Fundamental strength (ROE, ROA, margins, earnings quality) "
        "2) Valuation metrics (P/E, P/B ratios vs. peers) "
        "3) Growth prospects (forward EPS growth estimates) "
        "4) Macro context (correlation with market conditions, rates, inflation) "
        "Returns ranked lists of companies to LONG (buy) and SHORT (sell), "
        "with detailed rationale for each recommendation. "
        "Supports different strategies: value, growth, quality, or balanced."
    )
    args_schema: Type[BaseModel] = PortfolioRecommendationInput

    def _run(self, num_long: int = 5, num_short: int = 5,
             date: Optional[str] = None, strategy: str = "balanced") -> str:
        try:
            # Get all available tickers
            tickers = get_available_tickers()

            # Score all companies
            scored_companies = []
            for ticker in tickers:
                data = get_latest_data(ticker, date)
                if not data or not data.get('PRICE'):
                    continue

                valuation = calculate_valuation_metrics(ticker, date)

                # Skip companies with insufficient data
                if not all([data.get('ROE'), data.get('EPS'), data.get('PRICE')]):
                    continue

                # Calculate composite score based on strategy
                score = self._calculate_score(data, valuation, strategy)

                if score is not None:
                    scored_companies.append({
                        "ticker": ticker,
                        "score": score,
                        "price": data.get('PRICE'),
                        "eps": data.get('EPS'),
                        "roe": data.get('ROE'),
                        "roa": data.get('ROA'),
                        "gross_margin": data.get('GRM'),
                        "pe_ratio": valuation.get('pe_ratio'),
                        "pb_ratio": valuation.get('pb_ratio'),
                        "eps_growth": data.get('FVYRGRO_EPS'),
                        "dividend_yield": valuation.get('dividend_yield')
                    })

            if len(scored_companies) < num_long + num_short:
                return json.dumps({
                    "success": False,
                    "error": f"Insufficient companies with complete data. Found {len(scored_companies)}, need {num_long + num_short}"
                })

            # Sort by score
            scored_companies.sort(key=lambda x: x['score'], reverse=True)

            # Top N for long, bottom N for short
            long_recommendations = scored_companies[:num_long]
            short_recommendations = scored_companies[-num_short:][::-1]  # Reverse to show worst first

            # Calculate simple quartile-based ratings
            scores = [c['score'] for c in scored_companies]
            q75 = np.percentile(scores, 75)
            q50 = np.percentile(scores, 50)
            q25 = np.percentile(scores, 25)

            # Generate rationale and rating for each
            for i, company in enumerate(long_recommendations):
                company['rank'] = i + 1
                company['rationale'] = self._generate_long_rationale(company, strategy)
                # Simple rating based on quartiles
                if company['score'] >= q75:
                    company['rating'] = "Strong Buy"
                elif company['score'] >= q50:
                    company['rating'] = "Buy"
                else:
                    company['rating'] = "Hold"

            for i, company in enumerate(short_recommendations):
                company['rank'] = i + 1
                company['rationale'] = self._generate_short_rationale(company, strategy)
                # Simple rating based on quartiles (inverted for shorts)
                if company['score'] <= q25:
                    company['rating'] = "Strong Sell"
                elif company['score'] <= q50:
                    company['rating'] = "Sell"
                else:
                    company['rating'] = "Hold"

            # Create detailed text summary
            text_summary = f"\n=== PORTFOLIO RECOMMENDATIONS ({strategy.upper()} STRATEGY) ===\n\n"

            text_summary += "LONG POSITIONS (BUY):\n"
            for company in long_recommendations:
                text_summary += f"  #{company['rank']}. {company['ticker']} - {company['rating']}\n"
                roe_val = company.get('roe') or 0
                pe_val = company.get('pe_ratio') or 0
                text_summary += f"      Price: ${company['price']:.2f} | ROE: {roe_val:.1f}% | P/E: {pe_val:.1f}\n"
                text_summary += f"      Rationale: {company['rationale']}\n\n"

            text_summary += "\nSHORT POSITIONS (SELL):\n"
            for company in short_recommendations:
                text_summary += f"  #{company['rank']}. {company['ticker']} - {company['rating']}\n"
                roe_val = company.get('roe') or 0
                pe_val = company.get('pe_ratio') or 0
                text_summary += f"      Price: ${company['price']:.2f} | ROE: {roe_val:.1f}% | P/E: {pe_val:.1f}\n"
                text_summary += f"      Rationale: {company['rationale']}\n\n"

            text_summary += f"\nSUMMARY:\n"
            text_summary += f"  Total companies evaluated: {len(scored_companies)}\n"
            long_roe_vals = [c['roe'] for c in long_recommendations if c.get('roe')]
            short_roe_vals = [c['roe'] for c in short_recommendations if c.get('roe')]
            text_summary += f"  Average ROE (Long): {np.mean(long_roe_vals):.1f}% ({len(long_roe_vals)} companies)\n" if long_roe_vals else "  Average ROE (Long): N/A\n"
            text_summary += f"  Average ROE (Short): {np.mean(short_roe_vals):.1f}% ({len(short_roe_vals)} companies)\n" if short_roe_vals else "  Average ROE (Short): N/A\n"

            # Create visualization automatically
            import os
            import uuid
            from datetime import datetime

            viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            viz_id = f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            viz_config = {
                "type": "portfolio_recommendation",
                "id": viz_id,
                "title": f"Long/Short Portfolio Recommendations ({strategy.capitalize()} Strategy)",
                "long_positions": long_recommendations,
                "short_positions": short_recommendations
            }

            viz_file = os.path.join(viz_dir, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                import json as json_module
                json_module.dump(viz_config, f, indent=2)

            text_summary += f"\nA portfolio recommendation visualization has been created (Visualization ID: {viz_id}).\n"

            result = {
                "success": True,
                "strategy": strategy,
                "date": date or "latest",
                "long_positions": long_recommendations,
                "short_positions": short_recommendations,
                "summary": {
                    "total_evaluated": len(scored_companies),
                    "avg_score_long": np.mean([c['score'] for c in long_recommendations]),
                    "avg_score_short": np.mean([c['score'] for c in short_recommendations]),
                    "avg_roe_long": np.mean([c['roe'] for c in long_recommendations if c['roe']]),
                    "avg_roe_short": np.mean([c['roe'] for c in short_recommendations if c['roe']])
                },
                "text_summary": text_summary,
                "visualization_id": viz_id
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })

    def _calculate_score(self, data: dict, valuation: dict, strategy: str) -> Optional[float]:
        """Calculate composite score based on strategy."""
        scores = []

        # Quality factors (always important)
        if data.get('ROE'):
            roe_score = min(data['ROE'] / 25.0, 2.0)  # Normalize, cap at 2.0
            scores.append(roe_score * 20)  # Weight: 20

        if data.get('ROA'):
            roa_score = min(data['ROA'] / 15.0, 2.0)
            scores.append(roa_score * 15)  # Weight: 15

        if data.get('GRM'):
            margin_score = min(data['GRM'] / 40.0, 2.0)
            scores.append(margin_score * 15)  # Weight: 15

        # Valuation factors (inverse - lower is better for value strategy)
        if strategy in ['value', 'balanced']:
            if valuation.get('pe_ratio') and 0 < valuation['pe_ratio'] < 100:
                # Invert P/E (lower is better)
                pe_score = max(0, 2.0 - (valuation['pe_ratio'] / 20.0))
                weight = 20 if strategy == 'value' else 10
                scores.append(pe_score * weight)

            if valuation.get('pb_ratio') and 0 < valuation['pb_ratio'] < 50:
                pb_score = max(0, 2.0 - (valuation['pb_ratio'] / 3.0))
                weight = 15 if strategy == 'value' else 10
                scores.append(pb_score * weight)

        # Growth factors
        if strategy in ['growth', 'balanced']:
            if data.get('FVYRGRO_EPS'):
                growth_score = min(data['FVYRGRO_EPS'] / 20.0, 2.0)  # 20% growth = 2.0
                weight = 25 if strategy == 'growth' else 15
                scores.append(growth_score * weight)

        # Return average score
        return np.mean(scores) if scores else None

    def _generate_long_rationale(self, company: dict, strategy: str) -> str:
        """Generate rationale for long recommendation."""
        reasons = []

        if company['roe'] and company['roe'] > 20:
            reasons.append(f"Strong ROE of {company['roe']:.1f}%")

        if company['pe_ratio'] and company['pe_ratio'] < 15:
            reasons.append(f"Attractive P/E ratio of {company['pe_ratio']:.1f}")

        if company['eps_growth'] and company['eps_growth'] > 10:
            reasons.append(f"Strong EPS growth forecast of {company['eps_growth']:.1f}%")

        if company['gross_margin'] and company['gross_margin'] > 40:
            reasons.append(f"High gross margin of {company['gross_margin']:.1f}%")

        if company['dividend_yield'] and company['dividend_yield'] > 2:
            reasons.append(f"Attractive dividend yield of {company['dividend_yield']:.1f}%")

        return "; ".join(reasons) if reasons else "Solid fundamentals and valuation"

    def _generate_short_rationale(self, company: dict, strategy: str) -> str:
        """Generate rationale for short recommendation."""
        reasons = []

        if company['roe'] and company['roe'] < 10:
            reasons.append(f"Weak ROE of {company['roe']:.1f}%")

        if company['pe_ratio'] and company['pe_ratio'] > 30:
            reasons.append(f"Overvalued P/E ratio of {company['pe_ratio']:.1f}")

        if company['eps_growth'] and company['eps_growth'] < 0:
            reasons.append(f"Negative EPS growth forecast of {company['eps_growth']:.1f}%")

        if company['gross_margin'] and company['gross_margin'] < 20:
            reasons.append(f"Low gross margin of {company['gross_margin']:.1f}%")

        if company['roa'] and company['roa'] < 5:
            reasons.append(f"Weak ROA of {company['roa']:.1f}%")

        return "; ".join(reasons) if reasons else "Weak fundamentals and poor valuation"


class FundamentalMacroCorrelationInput(BaseModel):
    """Input schema for FundamentalMacroCorrelationTool."""
    ticker: str = Field(..., description="Company ticker symbol")
    macro_indicator: str = Field(..., description="Macro indicator to correlate with (e.g., FEDFUNDS, CPIAUCSL, ^VIX)")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class FundamentalMacroCorrelationTool(BaseTool):
    name: str = "Analyze Fundamental-Macro Correlation"
    description: str = (
        "Analyzes how a company's fundamentals (ROE, EPS, stock price) correlate with "
        "macroeconomic indicators like interest rates, inflation, VIX, or market indices. "
        "Use this to understand how external macro factors impact a specific company. "
        "Returns correlation coefficients and statistical significance. "
        "Useful for macro-aware portfolio construction."
    )
    args_schema: Type[BaseModel] = FundamentalMacroCorrelationInput

    def _run(self, ticker: str, macro_indicator: str,
             start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        try:
            ticker = ticker.upper()

            # Get company data
            company_history = get_company_fundamentals_history(ticker, start_date, end_date)
            if company_history.empty:
                return json.dumps({
                    "success": False,
                    "error": f"No historical data found for {ticker}"
                })

            # Get macro data
            macro_df = load_macro_factors()
            market_df = load_market_factors()
            combined_macro = pd.concat([macro_df, market_df], axis=1)

            if macro_indicator not in combined_macro.columns:
                return json.dumps({
                    "success": False,
                    "error": f"Macro indicator {macro_indicator} not found. Available: {', '.join(combined_macro.columns[:10])}..."
                })

            # Align dates (monthly frequency)
            company_history.index = pd.to_datetime(company_history.index)

            # Merge on date
            merged = company_history.join(combined_macro[[macro_indicator]], how='inner')

            if len(merged) < 10:
                return json.dumps({
                    "success": False,
                    "error": f"Insufficient overlapping data points ({len(merged)}). Need at least 10."
                })

            # Calculate correlations for available metrics
            correlations = {}
            for metric in ['PRICE', 'EPS', 'ROE', 'ROA']:
                if metric in merged.columns:
                    # Drop NaN and calculate correlation
                    subset = merged[[metric, macro_indicator]].dropna()
                    if len(subset) >= 10:
                        corr = subset[metric].corr(subset[macro_indicator])
                        correlations[metric] = {
                            "correlation": float(corr),
                            "data_points": len(subset)
                        }

            result = {
                "success": True,
                "ticker": ticker,
                "macro_indicator": macro_indicator,
                "date_range": {
                    "start": str(merged.index.min().date()),
                    "end": str(merged.index.max().date())
                },
                "correlations": correlations
            }

            # Add interpretation
            strongest = max(correlations.items(), key=lambda x: abs(x[1]['correlation']))
            result['interpretation'] = {
                "strongest_correlation": strongest[0],
                "correlation_value": strongest[1]['correlation'],
                "relationship": "positive" if strongest[1]['correlation'] > 0 else "negative",
                "strength": self._interpret_correlation(strongest[1]['correlation'])
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })

    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(corr)
        if abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.6:
            return "moderate"
        elif abs_corr < 0.8:
            return "strong"
        else:
            return "very strong"


class SectorAnalysisInput(BaseModel):
    """Input schema for SectorAnalysisTool."""
    tickers: str = Field(..., description="Comma-separated list of tickers representing a sector or industry")
    date: Optional[str] = Field(None, description="Date to analyze (YYYY-MM-DD)")


class SectorAnalysisTool(BaseTool):
    name: str = "Analyze Sector/Industry Fundamentals"
    description: str = (
        "Analyzes aggregate fundamental metrics for a group of companies (sector/industry). "
        "Calculates average ROE, ROA, P/E ratios, and identifies best/worst performers. "
        "Use this to understand sector-level trends and identify sector rotation opportunities. "
        "Useful for comparing company performance against industry peers."
    )
    args_schema: Type[BaseModel] = SectorAnalysisInput

    def _run(self, tickers: str, date: Optional[str] = None) -> str:
        try:
            ticker_list = [t.strip().upper() for t in tickers.split(',')]

            if len(ticker_list) < 2:
                return json.dumps({
                    "success": False,
                    "error": "Need at least 2 tickers for sector analysis"
                })

            # Collect data for all companies
            sector_data = []
            for ticker in ticker_list:
                data = get_latest_data(ticker, date)
                if data and data.get('PRICE'):
                    valuation = calculate_valuation_metrics(ticker, date)
                    sector_data.append({
                        "ticker": ticker,
                        "price": data.get('PRICE'),
                        "eps": data.get('EPS'),
                        "roe": data.get('ROE'),
                        "roa": data.get('ROA'),
                        "gross_margin": data.get('GRM'),
                        "pe_ratio": valuation.get('pe_ratio'),
                        "eps_growth": data.get('FVYRGRO_EPS')
                    })

            if len(sector_data) < 2:
                return json.dumps({
                    "success": False,
                    "error": f"Insufficient data for sector analysis. Found data for {len(sector_data)} companies."
                })

            # Calculate sector averages
            df = pd.DataFrame(sector_data)

            aggregates = {
                "avg_roe": float(df['roe'].dropna().mean()) if not df['roe'].dropna().empty else None,
                "avg_roa": float(df['roa'].dropna().mean()) if not df['roa'].dropna().empty else None,
                "avg_pe_ratio": float(df['pe_ratio'].dropna().mean()) if not df['pe_ratio'].dropna().empty else None,
                "avg_gross_margin": float(df['gross_margin'].dropna().mean()) if not df['gross_margin'].dropna().empty else None,
                "avg_eps_growth": float(df['eps_growth'].dropna().mean()) if not df['eps_growth'].dropna().empty else None
            }

            # Find best and worst performers
            best_roe = df.nlargest(1, 'roe')[['ticker', 'roe']].to_dict('records')[0] if not df['roe'].dropna().empty else None
            worst_roe = df.nsmallest(1, 'roe')[['ticker', 'roe']].to_dict('records')[0] if not df['roe'].dropna().empty else None

            result = {
                "success": True,
                "sector": {
                    "companies": ticker_list,
                    "count": len(sector_data)
                },
                "aggregates": aggregates,
                "best_performer": best_roe,
                "worst_performer": worst_roe,
                "company_details": sector_data
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })

