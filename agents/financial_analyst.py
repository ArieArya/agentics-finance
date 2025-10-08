"""
Financial Analyst Agent Configuration
"""

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
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
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4"),
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Initialize all tools
    tools = [
        # Data query tools
        AvailableIndicatorsTool(),
        DateRangeQueryTool(),
        IndicatorStatsTool(),
        # Analysis tools
        VolatilityAnalysisTool(),
        CorrelationAnalysisTool(),
        FindExtremeValuesTool(),
        # Visualization tools
        TimeSeriesPlotTool(),
        CorrelationHeatmapTool(),
        VolatilityPlotTool(),
        DistributionPlotTool(),
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
            "financial markets, and data analysis. You have access to comprehensive datasets "
            "containing macroeconomic indicators (Fed Funds Rate, CPI, unemployment, etc.) "
            "and market factors (S&P 500, VIX, Bitcoin, oil prices, etc.) spanning from 2008 to present. "
            "You excel at identifying trends, correlations, and anomalies in financial data. "
            "When users ask questions, you use your tools to query the data, perform analysis, "
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


def create_analysis_task(agent: Agent, user_question: str) -> Task:
    """
    Create a task for the financial analyst agent.

    Args:
        agent: The agent to assign the task to
        user_question: The user's question or request

    Returns:
        Task: Configured CrewAI task
    """
    task = Task(
        description=(
            f"User Question: {user_question}\n\n"
            "Instructions:\n"
            "1. Analyze the user's question carefully\n"
            "2. Use available tools to query and analyze the relevant data\n"
            "3. If appropriate, create visualizations to support your analysis\n"
            "4. Provide a comprehensive, well-structured answer with:\n"
            "   - Key findings and insights\n"
            "   - Statistical evidence and data points\n"
            "   - Context and interpretation\n"
            "   - Any relevant visualizations created\n"
            "5. Be specific with dates, values, and indicators\n"
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


def run_analysis(user_question: str) -> str:
    """
    Run a complete analysis for a user question.

    Args:
        user_question: The user's question

    Returns:
        str: The agent's analysis and response
    """
    try:
        # Create agent and task
        agent = create_financial_analyst_agent()
        task = create_analysis_task(agent, user_question)

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

