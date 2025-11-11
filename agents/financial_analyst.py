"""
Financial Analyst Agent Configuration
"""

from agentics import AG
from crewai import Agent, Task, Crew
import json
from tools import (
    # Data query tools
    AvailableIndicatorsTool,
    # Visualization tools
    TimeSeriesPlotTool,
    CorrelationHeatmapTool,
    VolatilityPlotTool,
    DistributionPlotTool,
    ScatterPlotTool,
    ComparativePerformanceTool,
    MovingAveragePlotTool,
    DrawdownChartTool,
    MultiIndicatorPlotTool,
    CompanyComparisonChartTool,
    FundamentalTimeSeriesPlotTool,
    ValuationScatterPlotTool,
    PortfolioRecommendationChartTool,
    PriceChartTool,
    PerformanceComparisonChartTool,
    VolatilityChartTool,
    # Transduction tool
    UnifiedTransductionTool,
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_financial_analyst_agent():
    """
    Create a financial analyst agent with access to transduction and visualization tools.

    Returns:
        Agent: Configured CrewAI agent
    """
    # Initialize LLM
    llm = AG.get_llm_provider()

    # All available tools (transduction + visualizations)
    tools = [
        # Data query tools
        AvailableIndicatorsTool(),
        # Transduction tool
        UnifiedTransductionTool(),
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
        # DJ30 visualization tools
        PriceChartTool(),
        PerformanceComparisonChartTool(),
        VolatilityChartTool(),
    ]

    # Create agent
    agent = Agent(
        role="Senior Financial Data Analyst",
        goal=(
            "Provide comprehensive analysis of financial and macroeconomic data using "
            "AI-powered transduction and visualization tools. Answer user questions with "
            "data-driven insights, statistical analysis, and clear visualizations."
        ),
        backstory=(
            "You are an expert financial analyst with deep knowledge of macroeconomics, "
            "financial markets, equity analysis, and data science. You have access to comprehensive datasets: "
            "1) Macroeconomic indicators (Fed Funds Rate, CPI, unemployment, retail sales, etc.) "
            "2) Market factors (S&P 500, VIX, Bitcoin, gold, oil prices, Treasury yields, etc.) "
            "3) Company fundamentals (61 publicly traded companies with EPS, ROE, ROA, P/E ratios, margins, etc.) "
            "4) DJ30 stock price data (30 Dow Jones Industrial Average stocks with daily OHLCV data from 2008-2025) "
            "All data spans from 2008 to present. "
            "You specialize in using AI-powered transduction to analyze complex financial questions "
            "by reducing large datasets into meaningful insights. You can create visualizations to "
            "support your analysis and help users understand patterns, trends, and relationships in the data. "
            "You always provide context and explain your findings in clear, accessible language. Your answer should be detailed and comprehensive. "
        ),
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15,
    )

    return agent


def create_analysis_task(agent: Agent, user_question: str, conversation_history: list = None, selected_columns: list = None) -> Task:
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

    # Note: Column selection is handled automatically by the tool based on UI selection
    # No need to pass selected_columns parameter - the tool reads it from the UI state

    task = Task(
        description=(
            f"{context}"
            f"Current User Question: {user_question}\n\n"
            "Instructions:\n"
            "1. Analyze the current question in the context of the conversation history\n"
            "2. If the question refers to a previous topic (e.g., 'similar analysis', 'same indicator'), use that context\n"
            "3. Choose the right tools for the task. You can use multiple tools and are encouraged to use visualization tools where appropriate:\n"
            "   - For DISCOVERING AVAILABLE INDICATORS: Use AvailableIndicatorsTool FIRST if you need to know what indicators are available\n"
            "     * Use this tool when you need to find valid indicator names for visualization tools\n"
            "     * Especially important before using ComparativePerformanceTool, TimeSeriesPlotTool, or MultiIndicatorPlotTool\n"
            "     * Returns lists of macroeconomic indicators, market factors, DJ30 stock tickers, and fundamental metrics\n"
            "   - For COMPLEX QUESTIONS requiring deep analysis: Use UnifiedTransductionTool\n"
            "     * Use this for questions about patterns, relationships, cause-effect analysis, or comprehensive summaries\n"
            "     * The transduction tool analyzes a comprehensive merged dataset containing macroeconomic indicators, market factors, DJ30 stock prices, company fundamentals, and news data\n"
            "     * The tool can analyze large date ranges and synthesize insights from multiple data points\n"
			"     * You are encouraged to use this tool alongside visualization tools to support your analysis\n"
            "     * Note: Column selection is handled automatically based on user's UI selection - you don't need to specify columns\n"
            "   - For VISUALIZATIONS: Use visualization tools to create charts and plots\n"
            "     * IMPORTANT: If you're unsure about valid indicator names, use AvailableIndicatorsTool first\n"
            "     * TimeSeriesPlotTool, CorrelationHeatmapTool, VolatilityPlotTool, DistributionPlotTool for basic charts\n"
            "     * ScatterPlotTool, ComparativePerformanceTool, MovingAveragePlotTool, DrawdownChartTool, MultiIndicatorPlotTool for advanced charts\n"
            "     * CompanyComparisonChartTool, FundamentalTimeSeriesPlotTool, ValuationScatterPlotTool for fundamental analysis charts\n"
            "     * PriceChartTool, PerformanceComparisonChartTool, VolatilityChartTool for DJ30 stock charts\n"
            "   - DO NOT query data before creating visualizations (it creates token overload)\n"
            "4. Provide a comprehensive, well-structured answer with:\n"
            "   - Key findings and insights\n"
            "   - Statistical evidence and data points\n"
            "   - Context and interpretation\n"
            "   - Any relevant visualizations created\n"
            "5. Be specific with dates, values, and indicators/tickers\n"
            "6. If you create visualizations, mention the visualization IDs in your response\n"
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


def run_analysis(user_question: str, conversation_history: list = None, selected_columns: list = None) -> str:
    # Set selected columns in the tool module so the tool can read it deterministically
    from tools.agentics_generic_tools import set_selected_columns
    set_selected_columns(selected_columns)
    """
    Run a complete analysis for a user question with conversation context.

    Args:
        user_question: The user's question
        conversation_history: List of previous messages for context
        selected_columns: List of column names to include in transduction analysis

    Returns:
        str: The agent's analysis and response
    """
    try:
        # Create agent and task
        agent = create_financial_analyst_agent()
        task = create_analysis_task(agent, user_question, conversation_history, selected_columns)

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
