"""
Financial Analyst Agent Configuration
"""

from agentics import AG
from crewai import Agent, Task, Crew
from tools import (
    DateRangeQueryTool,
    IndicatorStatsTool,
    AvailableIndicatorsTool,
    VolatilityAnalysisTool,
    CorrelationAnalysisTool,
    FindExtremeValuesTool,
    TimeSeriesPlotTool,
    CorrelationHeatmapTool,
    VolatilityPlotTool,
    DistributionPlotTool,
    ReturnsAnalysisTool,
    DrawdownAnalysisTool,
    MovingAverageTool,
    PercentageChangeTool,
    YearOverYearTool,
    ScatterPlotTool,
    ComparativePerformanceTool,
    MovingAveragePlotTool,
    DrawdownChartTool,
    MultiIndicatorPlotTool,
    HeadlinesFetcherTool,
    VolatilityNewsCorrelationTool,
    EventTimelineTool,
    ComprehensiveVolatilityExplanationTool,
    IdentifyCorrelatedMovementsTool,
    CompanyFundamentalsQueryTool,
    CompareFundamentalsTool,
    ScreenCompaniesTool,
    CompanyValuationTool,
    FundamentalHistoryTool,
    PortfolioRecommendationTool,
    FundamentalMacroCorrelationTool,
    SectorAnalysisTool,
    CompanyComparisonChartTool,
    FundamentalTimeSeriesPlotTool,
    ValuationScatterPlotTool,
    PortfolioRecommendationChartTool,
    DJ30ReturnsAnalysisTool,
    DJ30VolatilityAnalysisTool,
    PerformanceComparisonTool,
    PriceRangeAnalysisTool,
    VolatilityBasedPortfolioTool,
    MomentumBasedPortfolioTool,
    SectorDiversifiedPortfolioTool,
    PriceChartTool,
    PerformanceComparisonChartTool,
    VolatilityChartTool,
    UnifiedTransductionTool,
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_tool_categories():
    """
    Get list of all available tool categories with descriptions.

    Returns:
        dict: Dictionary mapping category names to descriptions
    """
    return {
        "Data Query": "Query financial data by date range, get indicator statistics",
        "Basic Analysis": "Volatility, correlation, and extreme value analysis",
        "Advanced Analysis": "Returns, drawdowns, moving averages, YoY analysis",
        "News & Events": "Fetch headlines and correlate with market events",
        "Volatility Explanation": "Comprehensive volatility analysis with context",
        "Company Fundamentals": "Query and compare company fundamental data",
        "Portfolio & Strategy": "Generate portfolio recommendations and strategies",
        "Basic Visualizations": "Time series, heatmaps, distributions",
        "Advanced Visualizations": "Scatter plots, comparative charts, drawdown plots",
        "Fundamental Visualizations": "Company comparison charts, valuation plots",
        "DJ30 Price Analysis": "Analyze DJ30 stock returns, volatility, performance",
        "DJ30 Portfolios": "Construct volatility/momentum/sector-based portfolios",
        "DJ30 Visualizations": "Price charts, performance comparisons for DJ30 stocks",
        "Advanced Transduction": "Deep analysis using AI-powered data reduction",
    }


def create_financial_analyst_agent(enabled_tool_categories=None):
    """
    Create a financial analyst agent with access to data analysis and visualization tools.

    Args:
        enabled_tool_categories: List of tool category names to enable. If None, all tools are enabled.

    Returns:
        Agent: Configured CrewAI agent
    """

    # Initialize LLM
    llm = AG.get_llm_provider()

    # Organize tools by category
    all_tools = {
        "Data Query": [
            AvailableIndicatorsTool(),
            DateRangeQueryTool(),
            IndicatorStatsTool(),
        ],
        "Basic Analysis": [
            VolatilityAnalysisTool(),
            CorrelationAnalysisTool(),
            FindExtremeValuesTool(),
        ],
        "Advanced Analysis": [
            ReturnsAnalysisTool(),
            DrawdownAnalysisTool(),
            MovingAverageTool(),
            PercentageChangeTool(),
            YearOverYearTool(),
        ],
        "News & Events": [
            HeadlinesFetcherTool(),
            VolatilityNewsCorrelationTool(),
            EventTimelineTool(),
        ],
        "Volatility Explanation": [
            ComprehensiveVolatilityExplanationTool(),
            IdentifyCorrelatedMovementsTool(),
        ],
        "Company Fundamentals": [
            CompanyFundamentalsQueryTool(),
            CompareFundamentalsTool(),
            ScreenCompaniesTool(),
            CompanyValuationTool(),
            FundamentalHistoryTool(),
        ],
        "Portfolio & Strategy": [
            PortfolioRecommendationTool(),
            FundamentalMacroCorrelationTool(),
            SectorAnalysisTool(),
        ],
        "Basic Visualizations": [
            TimeSeriesPlotTool(),
            CorrelationHeatmapTool(),
            VolatilityPlotTool(),
            DistributionPlotTool(),
        ],
        "Advanced Visualizations": [
            ScatterPlotTool(),
            ComparativePerformanceTool(),
            MovingAveragePlotTool(),
            DrawdownChartTool(),
            MultiIndicatorPlotTool(),
        ],
        "Fundamental Visualizations": [
            CompanyComparisonChartTool(),
            FundamentalTimeSeriesPlotTool(),
            ValuationScatterPlotTool(),
        ],
        "DJ30 Price Analysis": [
            DJ30ReturnsAnalysisTool(),
            DJ30VolatilityAnalysisTool(),
            PerformanceComparisonTool(),
            PriceRangeAnalysisTool(),
        ],
        "DJ30 Portfolios": [
            VolatilityBasedPortfolioTool(),
            MomentumBasedPortfolioTool(),
            SectorDiversifiedPortfolioTool(),
        ],
        "DJ30 Visualizations": [
            PriceChartTool(),
            PerformanceComparisonChartTool(),
            VolatilityChartTool(),
        ],
        "Advanced Transduction": [
            UnifiedTransductionTool(),
        ],
    }

    # Filter tools based on enabled categories
    if enabled_tool_categories is None:
        # Enable all tools if no filter specified
        tools = []
        for category_tools in all_tools.values():
            tools.extend(category_tools)
    else:
        # Enable only selected categories
        tools = []
        for category in enabled_tool_categories:
            if category in all_tools:
                tools.extend(all_tools[category])

    # Create agent
    agent = Agent(
        role="Senior Financial Data Analyst",
        goal=(
            "Provide comprehensive analysis of financial and macroeconomic data. "
            "Answer user questions with data-driven insights, statistical analysis, "
            "and clear visualizations."
        ),
        backstory=(
            "You are an expert financial analyst with deep knowledge of macroeconomics, "
            "financial markets, equity analysis, and data science. You have access to comprehensive datasets: "
            "1) Macroeconomic indicators (Fed Funds Rate, CPI, unemployment, retail sales, etc.) "
            "2) Market factors (S&P 500, VIX, Bitcoin, gold, oil prices, Treasury yields, etc.) "
            "3) Company fundamentals (61 publicly traded companies with EPS, ROE, ROA, P/E ratios, margins, etc.) "
            "4) DJ30 stock price data (30 Dow Jones Industrial Average stocks with daily OHLCV data from 2008-2025) "
            "All data spans from 2008 to present. "
            "You excel at identifying trends, correlations, anomalies, and investment opportunities. "
            "You can analyze company fundamentals, compare valuations, screen for opportunities, "
            "and generate long/short portfolio recommendations based on fundamental strength, "
            "macro context, and relative valuations. You also specialize in technical analysis, "
            "price momentum strategies, volatility-based trading, and quantitative portfolio construction. "
            "When users ask questions, you use your tools to query data, perform analysis, "
            "and create visualizations to support your insights. You always provide context "
            "and explain your findings in clear, accessible language."
        ),
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15,
    )

    return agent


def create_analysis_task(agent: Agent, user_question: str, conversation_history: list = None) -> Task:
    """
    Create a task for the financial analyst agent.

    Args:
        agent: The agent to assign the task to
        user_question: The user's question or request
        conversation_history: List of previous messages for context

    Returns:
        Task: Configured CrewAI task
    """
    # Build conversation context
    context = ""
    if conversation_history and len(conversation_history) > 0:
        context = "Previous Conversation:\n"
        for msg in conversation_history[-6:]:  # Include last 3 exchanges (6 messages)
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:500]  # Truncate long responses
            context += f"{role}: {content}\n\n"
        context += "---\n\n"

    task = Task(
        description=(
            f"{context}"
            f"Current User Question: {user_question}\n\n"
            "Instructions:\n"
            "1. Analyze the current question in the context of the conversation history\n"
            "2. If the question refers to a previous topic (e.g., 'similar analysis', 'same indicator'), use that context\n"
            "3. Choose the right tools for the task:\n"
            "   - For MARKET/MACRO VISUALIZATIONS: Use market visualization tools DIRECTLY (they load data themselves)\n"
            "   - For COMPANY COMPARISONS: Use CompanyComparisonChartTool for side-by-side fundamental metrics\n"
            "   - For PORTFOLIO RECOMMENDATIONS (fundamentals-based): Use PortfolioRecommendationTool\n"
            "   - For COMPANY ANALYSIS: Use CompanyFundamentalsQueryTool, CompanyValuationTool, FundamentalHistoryTool\n"
            "   - For SCREENING: Use ScreenCompaniesTool to filter companies by criteria\n"
            "   - For DJ30 PRICE ANALYSIS: Use DJ30ReturnsAnalysisTool, DJ30VolatilityAnalysisTool, PriceRangeAnalysisTool\n"
            "   - For DJ30 PRICE CHARTS: Use PriceChartTool (candlestick/OHLC), PerformanceComparisonChartTool, VolatilityChartTool\n"
            "   - For DJ30 PORTFOLIOS: Use VolatilityBasedPortfolioTool, MomentumBasedPortfolioTool, SectorDiversifiedPortfolioTool\n"
            "   - For SPECIFIC DATA VALUES: Use analysis tools or data query tools\n"
            "   - For COMPLEX QUESTIONS that standard tools cannot answer: Use UnifiedTransductionTool (specify dataset: macro/market/dj30/firm)\n"
            "   - DO NOT query data before creating visualizations (it creates token overload)\n"
            "4. When working with company fundamentals:\n"
            "   - Compare companies using CompareFundamentalsTool for analysis, CompanyComparisonChartTool for visualization\n"
            "   - Use ValuationScatterPlotTool to show relationships between metrics (e.g., ROE vs P/E)\n"
            "   - Use FundamentalTimeSeriesPlotTool to show evolution of company metrics over time\n"
            "5. When working with DJ30 stock prices:\n"
            "   - Use DJ30 tools for questions about specific stock prices, returns, volatility, or price patterns\n"
            "   - For volatility-based portfolios: Use VolatilityBasedPortfolioTool (automatically creates visualization)\n"
            "   - For momentum strategies: Use MomentumBasedPortfolioTool (automatically creates visualization)\n"
            "   - For sector diversification: Use SectorDiversifiedPortfolioTool (automatically creates visualization)\n"
            "   - Example: 'long 5 most volatile, short 5 least volatile' → Use VolatilityBasedPortfolioTool\n"
            "   - IMPORTANT: DJ30 portfolio tools automatically create visualizations and return Visualization IDs\n"
            "   - Always include the Visualization ID in your response when one is generated\n"
            "6. For portfolio recommendations:\n"
            "   - Use PortfolioRecommendationTool for fundamentals-based recommendations (value, growth, quality, balanced)\n"
            "   - Use DJ30 portfolio tools for price/technical-based recommendations (volatility, momentum, sector)\n"
            "   - These tools automatically create visualizations and return detailed recommendations\n"
            "   - IMPORTANT: When the tool outputs a Visualization ID (viz_YYYYMMDD_HHMMSS_XXXXXXXX), you MUST include this EXACT text in your response\n"
            "   - Copy the complete 'Visualization ID: viz_...' line from the tool output into your response\n"
            "   - Present the recommendations clearly and explain the strategy logic\n"
            "7. Provide a comprehensive, well-structured answer with:\n"
            "   - Key findings and insights\n"
            "   - Statistical evidence and data points\n"
            "   - Context and interpretation\n"
            "   - Any relevant visualizations created\n"
            "8. Be specific with dates, values, and indicators/tickers\n"
            "9. If you create visualizations, mention the visualization IDs in your response\n"
        ),
        agent=agent,
        expected_output=(
            "A comprehensive analysis that includes:\n"
            "- Direct answer to the user's question\n"
            "- Supporting data and statistics\n"
            "- Context and interpretation\n"
            "- References to any visualizations created (with their IDs)\n"
            "- Clear, accessible explanations"
        )
    )

    return task


def run_analysis(user_question: str, conversation_history: list = None, enabled_tool_categories: list = None) -> str:
    """
    Run a complete analysis for a user question with conversation context.

    Args:
        user_question: The user's question
        conversation_history: List of previous messages for context
        enabled_tool_categories: List of tool category names to enable. If None, all tools are enabled.

    Returns:
        str: The agent's analysis and response
    """
    try:
        # Create agent and task with filtered tools
        agent = create_financial_analyst_agent(enabled_tool_categories=enabled_tool_categories)
        task = create_analysis_task(agent, user_question, conversation_history)

        # Create crew and run
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )

        result = crew.kickoff()

        return str(result)

    except Exception as e:
        return f"Error during analysis: {str(e)}"

