"""
Portfolio construction tools for DJ30 stocks.
Provides tools for building portfolios based on volatility, momentum, sector diversification, etc.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional
import json
import pandas as pd
import numpy as np
import os
import uuid
from datetime import datetime
from utils.dj30_data_loader import get_multiple_tickers_data, get_available_tickers, get_all_sectors, get_sector_tickers


class VolatilityBasedPortfolioInput(BaseModel):
    """Input for Volatility-Based Portfolio Tool."""
    start_date: str = Field(..., description="Start date for calculating volatility (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for calculating volatility (YYYY-MM-DD)")
    portfolio_type: str = Field(
        default="long_short",
        description="Type of portfolio: 'long_low_vol' (long stable stocks), 'long_high_vol' (long volatile stocks), 'long_short' (arbitrage strategy), 'short_only' (short high volatility)"
    )
    num_positions: int = Field(default=5, description="Number of positions in the portfolio")
    lookback_days: int = Field(default=252, description="Number of days to look back for volatility calculation")


class VolatilityBasedPortfolioTool(BaseTool):
    name: str = "Create Volatility-Based Portfolio"
    description: str = (
        "Constructs portfolios based on stock volatility. Supports multiple strategies: "
        "'long_low_vol' for defensive portfolios (long stable, low-volatility stocks), "
        "'long_high_vol' for aggressive portfolios (long volatile stocks with growth potential), "
        "'long_short' for volatility arbitrage (long high-vol, short low-vol), "
        "'short_only' to short high-volatility stocks. "
        "Calculates historical volatility over a specified period."
    )
    args_schema: type[BaseModel] = VolatilityBasedPortfolioInput

    def _run(self, start_date: str, end_date: str, portfolio_type: str = "long_short", num_positions: int = 5, lookback_days: int = 252) -> str:
        # Get all DJ30 tickers
        all_tickers = get_available_tickers()

        # Get data for all tickers
        df = get_multiple_tickers_data(all_tickers, start_date, end_date)

        if df.empty:
            return json.dumps({"error": "No data found for date range"})

        # Calculate volatility for each ticker
        volatility_data = []

        for ticker in all_tickers:
            ticker_df = df[df['ticker'] == ticker].copy()

            if len(ticker_df) < max(2, lookback_days // 2):  # Need minimum data
                continue

            # Calculate daily returns
            ticker_df['daily_return'] = ticker_df['adj_close'].pct_change()
            returns = ticker_df['daily_return'].dropna()

            if len(returns) < 2:
                continue

            # Calculate annualized volatility
            volatility = returns.std() * np.sqrt(252)
            mean_return = returns.mean() * 252
            current_price = ticker_df['adj_close'].iloc[-1]

            # Get sector info
            sector = ticker_df['sector'].iloc[-1] if 'sector' in ticker_df.columns else "Unknown"
            
            # Get dividend yield if available
            dividend_yield = ticker_df['dividendYield'].iloc[-1] if 'dividendYield' in ticker_df.columns and pd.notna(ticker_df['dividendYield'].iloc[-1]) else 0.0

            volatility_data.append({
                "ticker": ticker,
                "volatility": float(volatility * 100),  # %
                "annualized_return": float(mean_return * 100),  # %
                "current_price": float(current_price),
                "sector": sector,
                "dividend_yield": float(dividend_yield)
            })

        # Sort by volatility
        sorted_by_vol = sorted(volatility_data, key=lambda x: x['volatility'], reverse=True)

        # Select positions based on portfolio type
        long_positions = []
        short_positions = []
        
        if portfolio_type == "long_low_vol":
            # Long the LEAST volatile stocks (defensive portfolio)
            long_positions = sorted_by_vol[-num_positions:]
            long_positions.reverse()  # Show lowest vol first
        elif portfolio_type == "long_high_vol":
            # Long the MOST volatile stocks (aggressive portfolio)
            long_positions = sorted_by_vol[:num_positions]
        elif portfolio_type == "short_only":
            # Short the MOST volatile stocks
            short_positions = sorted_by_vol[:num_positions]
        elif portfolio_type == "long_short":
            # Traditional arbitrage: long high vol, short low vol
            num_each = num_positions
            long_positions = sorted_by_vol[:num_each]
            short_positions = sorted_by_vol[-num_each:]
            short_positions.reverse()
        else:
            return json.dumps({"error": f"Invalid portfolio_type: {portfolio_type}. Must be one of: long_low_vol, long_high_vol, short_only, long_short"})
        
        if not long_positions and not short_positions:
            return json.dumps({"error": "No positions generated. Check your parameters."})

        # Add rationale based on portfolio type
        for i, pos in enumerate(long_positions, 1):
            pos['rank'] = i
            if portfolio_type == "long_low_vol":
                pos['rationale'] = f"Low volatility of {pos['volatility']:.2f}% indicates price stability and defensive characteristics. Annual return: {pos['annualized_return']:+.2f}%. Dividend yield: {pos['dividend_yield']:.2f}%."
            else:  # long_high_vol or long_short
                pos['rationale'] = f"High volatility of {pos['volatility']:.2f}% indicates potential for large price movements. Annual return: {pos['annualized_return']:+.2f}%."

        for i, pos in enumerate(short_positions, 1):
            pos['rank'] = i
            if portfolio_type == "short_only":
                pos['rationale'] = f"High volatility of {pos['volatility']:.2f}% makes this suitable for shorting. Annual return: {pos['annualized_return']:+.2f}%."
            else:  # long_short
                pos['rationale'] = f"Low volatility of {pos['volatility']:.2f}% suggests price stability. Annual return: {pos['annualized_return']:+.2f}%."

        # Calculate portfolio statistics
        avg_vol_long = np.mean([p['volatility'] for p in long_positions]) if long_positions else 0
        avg_vol_short = np.mean([p['volatility'] for p in short_positions]) if short_positions else 0

        # Create summary based on portfolio type
        portfolio_names = {
            "long_low_vol": "Low-Volatility Long Portfolio (Defensive)",
            "long_high_vol": "High-Volatility Long Portfolio (Aggressive)",
            "short_only": "High-Volatility Short Portfolio",
            "long_short": "Volatility-Based Long/Short Portfolio (Arbitrage)"
        }
        
        summary = f"\n=== {portfolio_names.get(portfolio_type, 'VOLATILITY-BASED PORTFOLIO')} ===\n"
        summary += f"Period: {start_date} to {end_date}\n\n"
        
        if long_positions:
            if portfolio_type == "long_low_vol":
                summary += "LONG POSITIONS (Low Volatility - Defensive):\n"
            elif portfolio_type == "long_high_vol":
                summary += "LONG POSITIONS (High Volatility - Aggressive):\n"
            else:
                summary += "LONG POSITIONS (High Volatility):\n"
                
            for pos in long_positions:
                summary += f"  #{pos['rank']}. {pos['ticker']} ({pos['sector']})\n"
                summary += f"      Price: ${pos['current_price']:.2f} | Volatility: {pos['volatility']:.2f}% | Return: {pos['annualized_return']:+.2f}%"
                if pos['dividend_yield'] > 0:
                    summary += f" | Div Yield: {pos['dividend_yield']:.2f}%"
                summary += f"\n      {pos['rationale']}\n\n"

        if short_positions:
            summary += "\nSHORT POSITIONS (High Volatility):\n"
            for pos in short_positions:
                summary += f"  #{pos['rank']}. {pos['ticker']} ({pos['sector']})\n"
                summary += f"      Price: ${pos['current_price']:.2f} | Volatility: {pos['volatility']:.2f}% | Return: {pos['annualized_return']:+.2f}%\n"
                summary += f"      {pos['rationale']}\n\n"

        summary += f"\nPORTFOLIO STATISTICS:\n"
        if long_positions:
            summary += f"  Average Volatility (Long): {avg_vol_long:.2f}%\n"
        if short_positions:
            summary += f"  Average Volatility (Short): {avg_vol_short:.2f}%\n"
        if long_positions and short_positions:
            summary += f"  Volatility Spread: {avg_vol_long - avg_vol_short:.2f}%\n"

        # Create visualization
        viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        viz_id = f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        viz_config = {
            "type": "volatility_portfolio",
            "id": viz_id,
            "title": f"{portfolio_names.get(portfolio_type, 'Volatility Portfolio')} ({start_date} to {end_date})",
            "portfolio_type": portfolio_type,
            "long_positions": long_positions,
            "short_positions": short_positions
        }

        viz_file = os.path.join(viz_dir, f"{viz_id}.json")
        with open(viz_file, 'w') as f:
            json.dump(viz_config, f, indent=2)

        summary += f"\nA portfolio visualization has been created (Visualization ID: {viz_id}).\n"

        # Build statistics dict
        statistics = {}
        if long_positions:
            statistics["avg_volatility_long"] = float(avg_vol_long)
        if short_positions:
            statistics["avg_volatility_short"] = float(avg_vol_short)
        if long_positions and short_positions:
            statistics["volatility_spread"] = float(avg_vol_long - avg_vol_short)

        result = {
            "success": True,
            "strategy": "volatility-based",
            "portfolio_type": portfolio_type,
            "period": {"start": start_date, "end": end_date},
            "long_positions": long_positions,
            "short_positions": short_positions,
            "statistics": statistics,
            "summary": summary,
            "visualization_id": viz_id
        }

        return json.dumps(result, indent=2)


class MomentumBasedPortfolioInput(BaseModel):
    """Input for Momentum-Based Portfolio Tool."""
    start_date: str = Field(..., description="Start date for calculating momentum (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for calculating momentum (YYYY-MM-DD)")
    num_long: int = Field(default=5, description="Number of stocks to long (high momentum)")
    num_short: int = Field(default=5, description="Number of stocks to short (low momentum)")
    lookback_days: int = Field(default=252, description="Number of days to look back for momentum calculation")


class MomentumBasedPortfolioTool(BaseTool):
    name: str = "Create Momentum-Based Portfolio"
    description: str = (
        "Constructs a long/short portfolio based on price momentum. "
        "Goes long the best performing stocks and short the worst performing stocks. "
        "Calculates cumulative returns over a specified period. "
        "Use this for trend-following or momentum strategies."
    )
    args_schema: type[BaseModel] = MomentumBasedPortfolioInput

    def _run(self, start_date: str, end_date: str, num_long: int = 5, num_short: int = 5, lookback_days: int = 252) -> str:
        # Get all DJ30 tickers
        all_tickers = get_available_tickers()

        # Get data for all tickers
        df = get_multiple_tickers_data(all_tickers, start_date, end_date)

        if df.empty:
            return json.dumps({"error": "No data found for date range"})

        # Calculate momentum for each ticker
        momentum_data = []

        for ticker in all_tickers:
            ticker_df = df[df['ticker'] == ticker].copy()

            if len(ticker_df) < max(2, lookback_days // 2):  # Need minimum data
                continue

            # Calculate cumulative return
            start_price = ticker_df['adj_close'].iloc[0]
            end_price = ticker_df['adj_close'].iloc[-1]
            momentum = (end_price - start_price) / start_price

            # Calculate daily returns for volatility
            ticker_df['daily_return'] = ticker_df['adj_close'].pct_change()
            returns = ticker_df['daily_return'].dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

            # Get sector info
            sector = ticker_df['sector'].iloc[-1] if 'sector' in ticker_df.columns else "Unknown"

            momentum_data.append({
                "ticker": ticker,
                "momentum": float(momentum * 100),  # %
                "volatility": float(volatility * 100),  # %
                "start_price": float(start_price),
                "current_price": float(end_price),
                "sector": sector
            })

        if len(momentum_data) < (num_long + num_short):
            return json.dumps({"error": f"Insufficient data. Only {len(momentum_data)} stocks available."})

        # Sort by momentum
        sorted_by_momentum = sorted(momentum_data, key=lambda x: x['momentum'], reverse=True)

        # Select long (high momentum) and short (low momentum) positions
        long_positions = sorted_by_momentum[:num_long]
        short_positions = sorted_by_momentum[-num_short:]

        # Add rationale
        for i, pos in enumerate(long_positions, 1):
            pos['rank'] = i
            pos['rationale'] = f"Strong momentum of {pos['momentum']:.2f}% suggests continued upward trend. Volatility: {pos['volatility']:.2f}%."

        for i, pos in enumerate(short_positions, 1):
            pos['rank'] = i
            pos['rationale'] = f"Weak momentum of {pos['momentum']:.2f}% suggests continued downward trend. Volatility: {pos['volatility']:.2f}%."

        # Calculate portfolio statistics
        avg_momentum_long = np.mean([p['momentum'] for p in long_positions])
        avg_momentum_short = np.mean([p['momentum'] for p in short_positions])

        # Create summary
        summary = f"\n=== MOMENTUM-BASED PORTFOLIO (Period: {start_date} to {end_date}) ===\n\n"
        summary += "LONG POSITIONS (High Momentum):\n"
        for pos in long_positions:
            summary += f"  #{pos['rank']}. {pos['ticker']} ({pos['sector']})\n"
            summary += f"      Price: ${pos['current_price']:.2f} | Momentum: {pos['momentum']:.2f}% | Volatility: {pos['volatility']:.2f}%\n"
            summary += f"      {pos['rationale']}\n\n"

        summary += "\nSHORT POSITIONS (Low Momentum):\n"
        for pos in short_positions:
            summary += f"  #{pos['rank']}. {pos['ticker']} ({pos['sector']})\n"
            summary += f"      Price: ${pos['current_price']:.2f} | Momentum: {pos['momentum']:.2f}% | Volatility: {pos['volatility']:.2f}%\n"
            summary += f"      {pos['rationale']}\n\n"

        summary += f"\nPORTFOLIO STATISTICS:\n"
        summary += f"  Average Momentum (Long): {avg_momentum_long:.2f}%\n"
        summary += f"  Average Momentum (Short): {avg_momentum_short:.2f}%\n"
        summary += f"  Momentum Spread: {avg_momentum_long - avg_momentum_short:.2f}%\n"

        # Create visualization
        viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        viz_id = f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        viz_config = {
            "type": "momentum_portfolio",
            "id": viz_id,
            "title": f"Momentum-Based Long/Short Portfolio ({start_date} to {end_date})",
            "long_positions": long_positions,
            "short_positions": short_positions
        }

        viz_file = os.path.join(viz_dir, f"{viz_id}.json")
        with open(viz_file, 'w') as f:
            json.dump(viz_config, f, indent=2)

        summary += f"\nA portfolio visualization has been created (Visualization ID: {viz_id}).\n"

        result = {
            "success": True,
            "strategy": "momentum-based",
            "period": {"start": start_date, "end": end_date},
            "long_positions": long_positions,
            "short_positions": short_positions,
            "statistics": {
                "avg_momentum_long": float(avg_momentum_long),
                "avg_momentum_short": float(avg_momentum_short),
                "momentum_spread": float(avg_momentum_long - avg_momentum_short)
            },
            "summary": summary,
            "visualization_id": viz_id
        }

        return json.dumps(result, indent=2)


class SectorDiversifiedPortfolioInput(BaseModel):
    """Input for Sector Diversified Portfolio Tool."""
    start_date: str = Field(..., description="Start date for analysis (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for analysis (YYYY-MM-DD)")
    strategy: str = Field(default="best_per_sector", description="Strategy: 'best_per_sector' or 'sector_rotation'")


class SectorDiversifiedPortfolioTool(BaseTool):
    name: str = "Create Sector-Diversified Portfolio"
    description: str = (
        "Constructs a sector-diversified portfolio to reduce concentration risk. "
        "Can select best performers from each sector or implement sector rotation strategy. "
        "Use this for balanced, diversified portfolios."
    )
    args_schema: type[BaseModel] = SectorDiversifiedPortfolioInput

    def _run(self, start_date: str, end_date: str, strategy: str = "best_per_sector") -> str:
        # Get all sectors and tickers
        all_sectors = get_all_sectors()
        all_tickers = get_available_tickers()

        # Get data for all tickers
        df = get_multiple_tickers_data(all_tickers, start_date, end_date)

        if df.empty:
            return json.dumps({"error": "No data found for date range"})

        # Calculate performance for each ticker
        ticker_performance = []

        for ticker in all_tickers:
            ticker_df = df[df['ticker'] == ticker].copy()

            if len(ticker_df) < 2:
                continue

            # Calculate returns
            start_price = ticker_df['adj_close'].iloc[0]
            end_price = ticker_df['adj_close'].iloc[-1]
            total_return = (end_price - start_price) / start_price

            # Calculate volatility
            ticker_df['daily_return'] = ticker_df['adj_close'].pct_change()
            returns = ticker_df['daily_return'].dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

            # Sharpe ratio (assuming risk-free rate = 0)
            mean_return = returns.mean() * 252
            sharpe = (mean_return / volatility) if volatility > 0 else 0

            sector = ticker_df['sector'].iloc[-1] if 'sector' in ticker_df.columns else "Unknown"

            ticker_performance.append({
                "ticker": ticker,
                "sector": sector,
                "total_return": float(total_return * 100),
                "volatility": float(volatility * 100),
                "sharpe_ratio": float(sharpe),
                "current_price": float(end_price)
            })

        if strategy == "best_per_sector":
            # Select best performer from each sector
            portfolio = []

            for sector in all_sectors:
                sector_stocks = [t for t in ticker_performance if t['sector'] == sector]

                if not sector_stocks:
                    continue

                # Select best Sharpe ratio stock from sector
                best_stock = max(sector_stocks, key=lambda x: x['sharpe_ratio'])
                best_stock['rank'] = len(portfolio) + 1
                best_stock['rationale'] = f"Best risk-adjusted return in {sector} sector with Sharpe ratio of {best_stock['sharpe_ratio']:.2f}."
                portfolio.append(best_stock)

            # Create summary
            summary = f"\n=== SECTOR-DIVERSIFIED PORTFOLIO (Best Per Sector, {start_date} to {end_date}) ===\n\n"
            for pos in portfolio:
                summary += f"{pos['rank']}. {pos['ticker']} ({pos['sector']})\n"
                summary += f"   Price: ${pos['current_price']:.2f} | Return: {pos['total_return']:.2f}% | Sharpe: {pos['sharpe_ratio']:.2f}\n"
                summary += f"   {pos['rationale']}\n\n"

            summary += f"\nPORTFOLIO STATISTICS:\n"
            summary += f"  Total Stocks: {len(portfolio)}\n"
            summary += f"  Sectors Represented: {len(set([p['sector'] for p in portfolio]))}\n"
            summary += f"  Average Return: {np.mean([p['total_return'] for p in portfolio]):.2f}%\n"
            summary += f"  Average Sharpe Ratio: {np.mean([p['sharpe_ratio'] for p in portfolio]):.2f}\n"

            # Create visualization
            viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            viz_id = f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            viz_config = {
                "type": "sector_portfolio",
                "id": viz_id,
                "title": f"Sector-Diversified Portfolio ({start_date} to {end_date})",
                "positions": portfolio
            }

            viz_file = os.path.join(viz_dir, f"{viz_id}.json")
            with open(viz_file, 'w') as f:
                json.dump(viz_config, f, indent=2)

            summary += f"\nA portfolio visualization has been created (Visualization ID: {viz_id}).\n"

            result = {
                "success": True,
                "strategy": "sector-diversified",
                "period": {"start": start_date, "end": end_date},
                "positions": portfolio,
                "summary": summary,
                "visualization_id": viz_id
            }

            return json.dumps(result, indent=2)

        else:
            return json.dumps({"error": f"Strategy '{strategy}' not implemented"})



