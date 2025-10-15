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
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_financial_analyst_agent():
    """
    Create a financial analyst agent with access to data analysis and visualization tools.

    Returns:
        Agent: Configured CrewAI agent
    """

    # Initialize LLM
    llm = AG.get_llm_provider()

    # Initialize all tools
    tools = [
        # Data query tools
        AvailableIndicatorsTool(),
        DateRangeQueryTool(),
        IndicatorStatsTool(),
        # Basic analysis tools
        VolatilityAnalysisTool(),
        CorrelationAnalysisTool(),
        FindExtremeValuesTool(),
        # Advanced analysis tools
        ReturnsAnalysisTool(),
        DrawdownAnalysisTool(),
        MovingAverageTool(),
        PercentageChangeTool(),
        YearOverYearTool(),
        # News and event analysis tools
        HeadlinesFetcherTool(),
        VolatilityNewsCorrelationTool(),
        EventTimelineTool(),
        # Comprehensive volatility explanation tools
        ComprehensiveVolatilityExplanationTool(),
        IdentifyCorrelatedMovementsTool(),
        # Company fundamental analysis tools
        CompanyFundamentalsQueryTool(),
        CompareFundamentalsTool(),
        ScreenCompaniesTool(),
        CompanyValuationTool(),
        FundamentalHistoryTool(),
        # Portfolio and macro correlation tools
        PortfolioRecommendationTool(),
        FundamentalMacroCorrelationTool(),
        SectorAnalysisTool(),
        # Basic visualization tools
        TimeSeriesPlotTool(),
        CorrelationHeatmapTool(),
        VolatilityPlotTool(),
        DistributionPlotTool(),
        # Advanced visualization tools
        ScatterPlotTool(),
        ComparativePerformanceTool(),
        MovingAveragePlotTool(),
        DrawdownChartTool(),
        MultiIndicatorPlotTool(),
        # Fundamental visualization tools
        CompanyComparisonChartTool(),
        FundamentalTimeSeriesPlotTool(),
        ValuationScatterPlotTool(),
        PortfolioRecommendationChartTool(),
    ]

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
            "All data spans from 2008 to present. "
            "You excel at identifying trends, correlations, anomalies, and investment opportunities. "
            "You can analyze company fundamentals, compare valuations, screen for opportunities, "
            "and generate long/short portfolio recommendations based on fundamental strength, "
            "macro context, and relative valuations. "
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
            "   - For PORTFOLIO RECOMMENDATIONS: Use PortfolioRecommendationTool + PortfolioRecommendationChartTool\n"
            "   - For COMPANY ANALYSIS: Use CompanyFundamentalsQueryTool, CompanyValuationTool, FundamentalHistoryTool\n"
            "   - For SCREENING: Use ScreenCompaniesTool to filter companies by criteria\n"
            "   - For SPECIFIC DATA VALUES: Use analysis tools or data query tools\n"
            "   - DO NOT query data before creating visualizations (it creates token overload)\n"
            "4. When working with company fundamentals:\n"
            "   - Compare companies using CompareFundamentalsTool for analysis, CompanyComparisonChartTool for visualization\n"
            "   - Use ValuationScatterPlotTool to show relationships between metrics (e.g., ROE vs P/E)\n"
            "   - Use FundamentalTimeSeriesPlotTool to show evolution of company metrics over time\n"
            "5. For portfolio recommendations:\n"
            "   - Use PortfolioRecommendationTool with strategy parameter ('value', 'growth', 'quality', or 'balanced')\n"
            "   - Visualize recommendations with PortfolioRecommendationChartTool\n"
            "   - Explain rationale based on fundamentals and macro context\n"
            "6. Provide a comprehensive, well-structured answer with:\n"
            "   - Key findings and insights\n"
            "   - Statistical evidence and data points\n"
            "   - Context and interpretation\n"
            "   - Any relevant visualizations created\n"
            "7. Be specific with dates, values, and indicators/tickers\n"
            "8. If you create visualizations, mention the visualization IDs in your response\n"
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


def run_analysis(user_question: str, conversation_history: list = None) -> str:
    """
    Run a complete analysis for a user question with conversation context.

    Args:
        user_question: The user's question
        conversation_history: List of previous messages for context

    Returns:
        str: The agent's analysis and response
    """
    try:
        # Create agent and task
        agent = create_financial_analyst_agent()
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

