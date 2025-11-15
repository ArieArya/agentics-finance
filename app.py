"""
Streamlit application for financial data Q&A with CrewAI agents.
"""

import streamlit as st
import os
import json
import glob
import re
import sys
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from agents import run_analysis
from utils import get_data_summary, get_column_descriptions, get_firm_data_summary, get_firm_column_descriptions, get_dj30_data_summary, get_dj30_column_descriptions

from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Financial Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .user-message {
        background-color: var(--secondary-background-color);
        border-left-color: #1f77b4;
        color: var(--text-color);
    }
    .assistant-message {
        background-color: var(--secondary-background-color);
        border-left-color: #2ca02c;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: var(--text-color);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1557a0;
        border: none;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }

    /* Allow sidebar to be resized larger (up to 60% of page width) */
    [data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 60% !important;
    }

    /* Ensure sidebar content is scrollable when extended */
    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto;
    }

    /* Style for example question buttons */
    .example-question-button {
        text-align: left;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        background-color: #f8f9fa;
        color: #333;
        font-size: 0.9rem;
        transition: all 0.2s;
    }

    .example-question-button:hover {
        background-color: #e7f3ff;
        border-color: #1f77b4;
        transform: translateX(4px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_visualizations" not in st.session_state:
    st.session_state.current_visualizations = []

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = ""

if "show_logs" not in st.session_state:
    st.session_state.show_logs = True

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "selected_transduction_columns" not in st.session_state:
    st.session_state.selected_transduction_columns = []

if "transduction_flow" not in st.session_state:
    st.session_state.transduction_flow = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "Chat"


def get_merged_data_columns():
    """Get all column names from merged_data.csv (or split files)."""
    from utils.csv_reader import read_merged_data_header

    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        header = read_merged_data_header(data_dir)
        return header
    except Exception as e:
        st.error(f"Error reading column names: {e}")
        return []


def get_columns_for_question(question: str, all_columns: list) -> list:
    """
    Manually map questions to their required columns.
    Returns a list of exact column names that should be pre-selected.
    """
    # Define question-to-columns mapping
    question_mappings = {
        # Question 1: Explain how AMZN and AAPL's strategy shifted over time, and any major investments they made from 2020 onwards
        "explain how amzn and aapl's strategy shifted over time, and any major investments they made from 2020 onwards": [
            # News and earnings for strategy/investment information
            'news_AAPL', 'news_AMZN', 'Earningcall_AAPL', 'Earningcall_AMZN', 'Headlines',
            # Price data to track performance
            'open_AAPL', 'high_AAPL', 'low_AAPL', 'close_AAPL', 'adj_close_AAPL', 'volume_AAPL',
            # Fundamental metrics for strategy analysis
            'EPS_AAPL_ACTUAL', 'EPS_AAPL_MEDEST', 'EPS_AMZN_ACTUAL', 'EPS_AMZN_MEDEST',
            'ROE_AAPL_ACTUAL', 'ROE_AMZN_ACTUAL',
            'SAL_AAPL_ACTUAL', 'SAL_AMZN_ACTUAL',
            'NET_AAPL_ACTUAL', 'NET_AMZN_ACTUAL'
        ],

        # Question 2: Why did the stock price of AAPL drop in March 2020?
        "why did the stock price of aapl drop in march 2020?": [
            # Price data to see the drop
            'open_AAPL', 'high_AAPL', 'low_AAPL', 'close_AAPL', 'adj_close_AAPL', 'volume_AAPL',
            # News to explain why
            'news_AAPL', 'news_market', 'Headlines',
            # Earnings calls around that time
            'Earningcall_AAPL',
            # Market context (use original CSV column names - tool will sanitize them)
            '^GSPC', '^VIX'
        ],

        # Question 3: What were the key factors behind NVDA's stock price surge in 2023?
        # Note: NVDA doesn't have price columns in this dataset, so we focus on news, earnings, and fundamentals
        "what were the key factors behind nvda's stock price surge in 2023?": [
            # News and earnings
            'news_NVDA', 'Earningcall_NVDA', 'Headlines',
            # Fundamentals
            'EPS_NVDA_ACTUAL', 'EPS_NVDA_MEDEST',
            'ROE_NVDA_ACTUAL',
            'SAL_NVDA_ACTUAL', 'NET_NVDA_ACTUAL',
            # Market context (use original CSV column names - tool will sanitize them)
            '^GSPC', '^VIX'
        ],

        # Question 4: Analyze the relationship between MSFT's earnings announcements and its stock price movements from 2020-2023
        "analyze the relationship between msft's earnings announcements and its stock price movements from 2020-2023": [
            # Price data
            'open_MSFT', 'high_MSFT', 'low_MSFT', 'close_MSFT', 'adj_close_MSFT', 'volume_MSFT',
            # Earnings calls
            'Earningcall_MSFT',
            # News
            'news_MSFT', 'Headlines',
            # Earnings fundamentals
            'EPS_MSFT_ACTUAL', 'EPS_MSFT_MEDEST',
            'ROE_MSFT_ACTUAL',
            'NET_MSFT_ACTUAL'
        ],

        # Question 5: What market events and company-specific news drove JPM's volatility during the 2022-2023 period?
        "what market events and company-specific news drove jpm's volatility during the 2022-2023 period?": [
            # Price data for volatility
            'open_JPM', 'high_JPM', 'low_JPM', 'close_JPM', 'adj_close_JPM', 'volume_JPM',
            # News
            'news_JPM', 'news_market', 'Headlines',
            # Earnings
            'Earningcall_JPM',
            # Market context (use original CSV column names - tool will sanitize them)
            '^GSPC', '^VIX', 'DGS10',
            # Fundamentals (JPM uses returnonequity instead of ROE)
            'returnonequity_JPM'
        ]
    }

    # Normalize question for lookup
    question_normalized = question.lower().strip()

    # Check for exact match
    if question_normalized in question_mappings:
        selected_cols = question_mappings[question_normalized]
        # Filter to only include columns that actually exist in the dataset
        return [col for col in selected_cols if col in all_columns]

    # If no match found, return empty list
    return []


def set_example_question(question: str):
    """Set the example question in the text input and pre-select relevant columns."""
    # Set the question text
    st.session_state.user_input = question

    # Get all columns and determine which ones to select
    all_columns = get_merged_data_columns()
    if all_columns:
        relevant_cols = get_columns_for_question(question, all_columns)
        # Remove duplicates and ensure we have a clean list
        relevant_cols = list(set(relevant_cols))
        # Set session state - this will be reflected in checkboxes on next render
        st.session_state.selected_transduction_columns = relevant_cols

        # Clear all existing checkbox states so they can be reinitialized from selected_transduction_columns
        # This prevents the error of modifying widget state after instantiation
        all_columns_list = get_merged_data_columns()
        if all_columns_list:
            for col in all_columns_list:
                checkbox_key = f"col_{col}"
                # Delete existing checkbox state - it will be reinitialized on next render
                if checkbox_key in st.session_state:
                    del st.session_state[checkbox_key]

        print(f"🔧 Set {len(relevant_cols)} columns for question: {question[:50]}...")
        if relevant_cols:
            print(f"   Selected columns: {relevant_cols}")

    # Rerun to update the UI with new selections
    st.rerun()


def display_transduction_flow():
    """Display the transduction flow visualization."""
    st.markdown('<div class="main-header">Transduction Flow</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visualize how data flows through the transduction process</div>', unsafe_allow_html=True)

    if not st.session_state.transduction_flow:
        st.info("💡 No transduction flow data available. Run a transduction analysis in the Chat page to see the flow here.")
        return

    flow = st.session_state.transduction_flow

    # Helper function to get Pydantic model definition
    def get_model_definition(model_class):
        """Get the Pydantic model definition as a string."""
        try:
            import inspect
            from typing import Union, get_origin, get_args
            # Try to get source code
            try:
                source = inspect.getsource(model_class)
                return source
            except (OSError, TypeError):
                # If source not available, reconstruct from model_fields
                if hasattr(model_class, 'model_fields'):
                    lines = [f"class {model_class.__name__}(BaseModel):"]
                    for field_name, field_info in model_class.model_fields.items():
                        # Get field type annotation
                        field_type = field_info.annotation
                        field_type_str = None

                        # Handle Union types (including Optional)
                        origin = get_origin(field_type)
                        if origin is Union:
                            args = get_args(field_type)
                            # Check if it's Optional (Union with None)
                            non_none_args = [arg for arg in args if arg is not type(None)]
                            if len(non_none_args) == 1 and len(args) == 2:
                                # It's Optional[T]
                                type_name = getattr(non_none_args[0], '__name__', str(non_none_args[0]))
                                field_type_str = f"{type_name} | None"
                            else:
                                # It's a Union of multiple types
                                type_names = [getattr(arg, '__name__', str(arg)) for arg in non_none_args]
                                field_type_str = " | ".join(type_names)
                        else:
                            # Regular type
                            field_type_str = getattr(field_type, '__name__', str(field_type))

                        # Get default value
                        default = field_info.default
                        if default is None:
                            default_str = "None"
                        elif isinstance(default, str):
                            default_str = f'"{default}"'
                        else:
                            default_str = str(default)

                        # Get description if available
                        description = field_info.description
                        if description:
                            # Escape quotes in description
                            description_escaped = description.replace('"', '\\"')
                            lines.append(f"    {field_name}: {field_type_str} = Field(")
                            lines.append(f"        {default_str},")
                            lines.append(f'        description="{description_escaped}"')
                            lines.append(f"    )")
                        else:
                            lines.append(f"    {field_name}: {field_type_str} = {default_str}")

                    return "\n".join(lines)
                else:
                    return f"class {model_class.__name__}(BaseModel):\n    # Model definition not available"
        except Exception as e:
            return f"# Error getting model definition: {e}"

    # Helper function to convert AG to dataframe
    def ag_to_df(ag_obj):
        """Convert an AG object to a pandas DataFrame."""
        try:
            if hasattr(ag_obj, 'to_dataframe'):
                return ag_obj.to_dataframe()
            elif hasattr(ag_obj, 'states'):
                # Convert states directly to dataframe
                data = [state.model_dump() for state in ag_obj.states]
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error converting to dataframe: {e}")
            return pd.DataFrame()

    # Helper function to convert batch results to dataframe
    def batches_to_df(batches):
        """Convert a list of batch results to a pandas DataFrame."""
        try:
            data = []
            for batch in batches:
                if hasattr(batch, 'model_dump'):
                    data.append(batch.model_dump())
                elif isinstance(batch, dict):
                    data.append(batch)
                else:
                    # Try to convert to dict
                    data.append({"result": str(batch)})
            return pd.DataFrame(data)
        except Exception as e:
            st.warning(f"Error converting batches to dataframe: {e}")
            return pd.DataFrame()

    # 1. Initial States
    if "initial_states" in flow:
        st.markdown("---")
        st.markdown(f"### 1️⃣ Initial States")
        st.markdown(f"**Pydantic Class:** `{flow['initial_states']['atype_name']}`")

        # Display model definition
        if hasattr(flow['initial_states']['agentics'], 'atype') and flow['initial_states']['agentics'].atype:
            model_def = get_model_definition(flow['initial_states']['agentics'].atype)
            with st.expander("📋 View Model Definition", expanded=False):
                st.code(model_def, language="python")

        st.markdown(f"**Number of Rows:** {flow['initial_states']['num_rows']}")

        initial_df = ag_to_df(flow['initial_states']['agentics'])
        if not initial_df.empty:
            st.dataframe(initial_df, use_container_width=True, height=800)
        else:
            st.warning("Could not convert initial states to dataframe")

    # 2. Final Intermediate Result
    if "final_intermediate" in flow:
        st.markdown("---")
        st.markdown(f"### 2️⃣ Final Intermediate Result")
        st.markdown(f"**Pydantic Class:** `{flow['final_intermediate']['atype_name']}`")

        # Display model definition
        if hasattr(flow['final_intermediate']['agentics'], 'atype') and flow['final_intermediate']['agentics'].atype:
            model_def = get_model_definition(flow['final_intermediate']['agentics'].atype)
            with st.expander("📋 View Model Definition", expanded=False):
                st.code(model_def, language="python")

        st.markdown(f"**Number of Rows:** {flow['final_intermediate']['num_rows']}")

        final_intermediate_df = ag_to_df(flow['final_intermediate']['agentics'])
        if not final_intermediate_df.empty:
            st.dataframe(final_intermediate_df, use_container_width=True, height=400)
        else:
            st.warning("Could not convert final intermediate result to dataframe")

    # 3. Final Answer
    if "final_answer" in flow:
        st.markdown("---")
        st.markdown(f"### 3️⃣ Final Answer")
        st.markdown(f"**Pydantic Class:** `{flow['final_answer']['atype_name']}`")

        # Display model definition
        if hasattr(flow['final_answer']['agentics'], 'atype') and flow['final_answer']['agentics'].atype:
            model_def = get_model_definition(flow['final_answer']['agentics'].atype)
            with st.expander("📋 View Model Definition", expanded=False):
                st.code(model_def, language="python")

        st.markdown(f"**Number of Rows:** {flow['final_answer']['num_rows']}")

        final_answer_df = ag_to_df(flow['final_answer']['agentics'])
        if not final_answer_df.empty:
            st.dataframe(final_answer_df, use_container_width=True, height=50)
        else:
            st.warning("Could not convert final answer to dataframe")

    st.markdown("---")
    st.caption("💡 This flow shows how your data is transformed through the transduction process. Each stage reduces the data while preserving key insights.")


def clean_old_visualizations():
    """Remove visualization files that are no longer referenced in conversation history."""
    viz_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    if os.path.exists(viz_dir):
        # Collect all visualization IDs from entire conversation history
        all_referenced_viz_ids = set()

        for message in st.session_state.messages:
            if "visualizations" in message and message["visualizations"]:
                all_referenced_viz_ids.update(message["visualizations"])

        # Also include current visualizations
        all_referenced_viz_ids.update(st.session_state.current_visualizations)

        # Get all visualization files
        all_viz_files = glob.glob(os.path.join(viz_dir, "*.json"))

        # Build set of files to keep
        files_to_keep = {
            os.path.join(viz_dir, f"{viz_id}.json")
            for viz_id in all_referenced_viz_ids
        }

        # Delete only files not referenced in conversation history
        for viz_file in all_viz_files:
            if viz_file not in files_to_keep:
                try:
                    os.remove(viz_file)
                except:
                    pass


def load_visualization(viz_id: str):
    """Load and render a visualization from JSON file."""
    viz_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    viz_file = os.path.join(viz_dir, f"{viz_id}.json")

    if not os.path.exists(viz_file):
        st.warning(f"Visualization {viz_id} not found.")
        return

    try:
        with open(viz_file, 'r') as f:
            viz_config = json.load(f)

        viz_type = viz_config.get("type")

        if viz_type == "time_series":
            render_time_series(viz_config)
        elif viz_type == "correlation_heatmap":
            render_correlation_heatmap(viz_config)
        elif viz_type == "volatility_plot":
            render_volatility_plot(viz_config)
        elif viz_type == "distribution":
            render_distribution(viz_config)
        elif viz_type == "scatter":
            render_scatter(viz_config)
        elif viz_type == "comparative_performance":
            render_comparative_performance(viz_config)
        elif viz_type == "moving_average":
            render_moving_average(viz_config)
        elif viz_type == "drawdown":
            render_drawdown(viz_config)
        elif viz_type == "multi_indicator":
            render_multi_indicator(viz_config)
        elif viz_type == "company_comparison":
            render_company_comparison(viz_config)
        elif viz_type == "fundamental_time_series":
            render_fundamental_time_series(viz_config)
        elif viz_type == "valuation_scatter":
            render_valuation_scatter(viz_config)
        elif viz_type == "portfolio_recommendation":
            render_portfolio_recommendation(viz_config)
        elif viz_type == "price_chart":
            render_price_chart(viz_config)
        elif viz_type == "performance_comparison":
            render_performance_comparison(viz_config)
        elif viz_type == "volatility_chart":
            render_volatility_chart(viz_config)
        elif viz_type == "volatility_portfolio":
            render_volatility_portfolio(viz_config)
        elif viz_type == "momentum_portfolio":
            render_momentum_portfolio(viz_config)
        elif viz_type == "sector_portfolio":
            render_sector_portfolio(viz_config)
        else:
            st.warning(f"Unknown visualization type: {viz_type}")
    except Exception as e:
        st.error(f"Error rendering visualization {viz_id}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_time_series(config: dict):
    """Render time series plot with support for dual y-axes when scales differ significantly."""
    df = pd.DataFrame(config["data"])
    indicators = config.get("indicators", df["indicator"].unique().tolist())

    # Indicators that should be on secondary axis (volatility indices, percentages, etc.)
    secondary_axis_indicators = ['^VIX', 'UNRATE', 'FEDFUNDS', 'DGS10', 'DGS2', 'CPIAUCSL']

    # Determine if we need dual axes
    # Check if we have both a secondary axis indicator and a primary axis indicator
    has_secondary = any(ind in secondary_axis_indicators for ind in indicators)
    has_primary = any(ind not in secondary_axis_indicators for ind in indicators)
    use_dual_axes = has_secondary and has_primary and len(indicators) > 1

    if use_dual_axes:
        # Use make_subplots with secondary_y for better visualization of different scales
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, indicator in enumerate(indicators):
            indicator_df = df[df["indicator"] == indicator]
            use_secondary = indicator in secondary_axis_indicators

            fig.add_trace(
                go.Scatter(
                    x=indicator_df["date"],
                    y=indicator_df["value"],
                    name=indicator,
                    line=dict(color=colors[i % len(colors)]),
                    mode='lines'
                ),
                secondary_y=use_secondary
            )

        # Update axes labels
        primary_indicators = [ind for ind in indicators if ind not in secondary_axis_indicators]
        secondary_indicators = [ind for ind in indicators if ind in secondary_axis_indicators]

        fig.update_yaxes(
            title_text=", ".join(primary_indicators) if primary_indicators else "Value",
            secondary_y=False
        )
        fig.update_yaxes(
            title_text=", ".join(secondary_indicators) if secondary_indicators else "Value",
            secondary_y=True
        )

        fig.update_xaxes(title_text="Date")

        fig.update_layout(
            title=config["title"],
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
    else:
        # Use standard plotly express for single scale
        fig = px.line(
            df,
            x="date",
            y="value",
            color="indicator",
            title=config["title"],
            labels={"date": "Date", "value": "Value", "indicator": "Indicator"}
        )

        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )

    st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(config: dict):
    """Render correlation heatmap."""
    df = pd.DataFrame(config["data"])

    # Pivot data for heatmap
    pivot_df = df.pivot(index="y", columns="x", values="correlation")

    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale="RdBu",
        zmid=0,
        text=pivot_df.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=config["title"],
        xaxis_title="",
        yaxis_title="",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


def render_volatility_plot(config: dict):
    """Render volatility plot with dual y-axes."""
    df = pd.DataFrame(config["data"])

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add value trace
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["value"],
            name=config["indicator"],
            line=dict(color="#1f77b4")
        ),
        secondary_y=False,
    )

    # Add volatility trace if available
    if "volatility" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["volatility"],
                name=f"Volatility (rolling {config['window']}d)",
                line=dict(color="#ff7f0e", dash="dash")
            ),
            secondary_y=True,
        )

    # Update layout
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=config["indicator"], secondary_y=False)
    fig.update_yaxes(title_text="Volatility", secondary_y=True)

    fig.update_layout(
        title=config["title"],
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def render_distribution(config: dict):
    """Render distribution histogram."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=config["data"],
        nbinsx=50,
        name=config["indicator"],
        marker_color="#1f77b4"
    ))

    # Add mean line
    mean = config["stats"]["mean"]
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean:.2f}"
    )

    fig.update_layout(
        title=config["title"],
        xaxis_title=config["indicator"],
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_scatter(config: dict):
    """Render scatter plot."""
    df = pd.DataFrame(config["data"])

    fig = px.scatter(
        df,
        x="x",
        y="y",
        title=config["title"],
        labels={"x": config["x_indicator"], "y": config["y_indicator"]},
        hover_data=["date"]
    )

    # Add trendline
    fig.update_traces(marker=dict(size=5, opacity=0.6))

    # Add correlation text
    correlation = config.get("correlation", 0)
    fig.add_annotation(
        text=f"Correlation: {correlation:.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_comparative_performance(config: dict):
    """Render comparative performance chart (normalized to 100)."""
    df = pd.DataFrame(config["data"])

    fig = px.line(
        df,
        x="date",
        y="value",
        color="indicator",
        title=config["title"],
        labels={"date": "Date", "value": "Normalized Value (Base=100)", "indicator": "Indicator"}
    )

    # Add horizontal line at 100
    fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Start")

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def render_moving_average(config: dict):
    """Render moving average chart."""
    df = pd.DataFrame(config["data"])

    fig = go.Figure()

    # Add price line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["price"],
        name=config["indicator"],
        line=dict(color="#1f77b4", width=2)
    ))

    # Add moving averages
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, window in enumerate(config["windows"]):
        ma_col = f"MA_{window}"
        if ma_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df[ma_col],
                name=f"MA {window}",
                line=dict(color=colors[i % len(colors)], width=1.5, dash="dash")
            ))

    fig.update_layout(
        title=config["title"],
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def render_drawdown(config: dict):
    """Render drawdown chart."""
    df = pd.DataFrame(config["data"])

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(config["indicator"], "Drawdown from Peak"),
        row_heights=[0.6, 0.4]
    )

    # Add price
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["price"],
            name=config["indicator"],
            line=dict(color="#1f77b4")
        ),
        row=1, col=1
    )

    # Add drawdown
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["drawdown_pct"],
            name="Drawdown %",
            fill='tozeroy',
            line=dict(color="#d62728")
        ),
        row=2, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    fig.update_layout(
        title=config["title"],
        hovermode="x unified",
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_multi_indicator(config: dict):
    """Render multi-indicator dashboard with subplots."""
    indicators = config["indicators"]
    data = config["data"]

    # Create subplots
    fig = make_subplots(
        rows=len(indicators),
        cols=1,
        shared_xaxes=True,
        subplot_titles=indicators,
        vertical_spacing=0.05
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Add each indicator
    for i, indicator in enumerate(indicators):
        indicator_data = data[indicator]
        df = pd.DataFrame(indicator_data)

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["value"],
                name=indicator,
                line=dict(color=colors[i % len(colors)]),
                showlegend=False
            ),
            row=i+1, col=1
        )

        fig.update_yaxes(title_text=indicator, row=i+1, col=1)

    fig.update_xaxes(title_text="Date", row=len(indicators), col=1)

    fig.update_layout(
        title=config["title"],
        hovermode="x unified",
        height=250 * len(indicators)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_company_comparison(config: dict):
    """Render company comparison bar chart."""
    data = config["data"]
    metrics = config["metrics"]

    # Create subplots for each metric
    num_metrics = len(metrics)
    fig = make_subplots(
        rows=num_metrics,
        cols=1,
        subplot_titles=metrics,
        vertical_spacing=0.08
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, metric in enumerate(metrics):
        tickers = [d["ticker"] for d in data]
        values = [d.get(metric) for d in data]

        # Filter out None values
        filtered_data = [(t, v) for t, v in zip(tickers, values) if v is not None]
        if not filtered_data:
            continue

        tickers_filtered, values_filtered = zip(*filtered_data)

        fig.add_trace(
            go.Bar(
                x=list(tickers_filtered),
                y=list(values_filtered),
                name=metric,
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=i+1, col=1
        )

        fig.update_yaxes(title_text=metric, row=i+1, col=1)

    fig.update_xaxes(title_text="Company", row=num_metrics, col=1)

    fig.update_layout(
        title=config["title"],
        height=300 * num_metrics,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_fundamental_time_series(config: dict):
    """Render fundamental time series plot."""
    df = pd.DataFrame(config["data"])
    metrics = config["metrics"]

    # Check if we need dual axes based on scale differences
    if len(metrics) > 1:
        # Simple heuristic: if ranges differ by more than 10x, use dual axes
        ranges = {}
        for metric in metrics:
            metric_data = df[df["metric"] == metric]["value"]
            if not metric_data.empty:
                ranges[metric] = metric_data.max() - metric_data.min()

        if ranges:
            max_range = max(ranges.values())
            min_range = min(ranges.values())
            use_dual_axes = max_range / min_range > 10 if min_range > 0 else False
        else:
            use_dual_axes = False
    else:
        use_dual_axes = False

    if use_dual_axes and len(metrics) == 2:
        # Use dual y-axes for 2 metrics with different scales
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        colors = ["#1f77b4", "#ff7f0e"]
        for i, metric in enumerate(metrics):
            metric_df = df[df["metric"] == metric]
            fig.add_trace(
                go.Scatter(
                    x=metric_df["date"],
                    y=metric_df["value"],
                    name=metric,
                    line=dict(color=colors[i]),
                    mode='lines+markers'
                ),
                secondary_y=(i == 1)
            )

        fig.update_yaxes(title_text=metrics[0], secondary_y=False)
        fig.update_yaxes(title_text=metrics[1], secondary_y=True)
    else:
        # Standard plot
        fig = px.line(
            df,
            x="date",
            y="value",
            color="metric",
            title=config["title"],
            labels={"date": "Date", "value": "Value", "metric": "Metric"},
            markers=True
        )

    fig.update_layout(
        title=config["title"],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def render_valuation_scatter(config: dict):
    """Render valuation scatter plot."""
    df = pd.DataFrame(config["data"])

    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="ticker",
        title=config["title"],
        labels={"x": config["x_metric"], "y": config["y_metric"]},
        size_max=15
    )

    # Position labels above points
    fig.update_traces(
        textposition='top center',
        marker=dict(size=12, opacity=0.7)
    )

    # Add trendline
    if len(df) >= 2:
        z = np.polyfit(df["x"], df["y"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df["x"].min(), df["x"].max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='Trendline',
                line=dict(dash='dash', color='gray')
            )
        )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_portfolio_recommendation(config: dict):
    """Render portfolio recommendation chart."""
    long_positions = config["long_positions"]
    short_positions = config["short_positions"]

    # Display recommendation summary table
    st.markdown("### Portfolio Recommendations Summary")

    def color_rating(rating):
        """Return colored HTML for rating."""
        colors = {
            "Strong Buy": "#00C851",  # Green
            "Buy": "#4CAF50",         # Light Green
            "Hold": "#FFC107",        # Amber
            "Sell": "#FF5722",        # Deep Orange
            "Strong Sell": "#F44336"  # Red
        }
        color = colors.get(rating, "#757575")
        return f'<span style="color: {color}; font-weight: bold;">{rating}</span>'

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🟢 Long Positions**")
        for i, p in enumerate(long_positions):
            roe_val = f"{p.get('roe', 0):.1f}%" if p.get('roe') else "N/A"
            pe_val = f"{p.get('pe_ratio', 0):.1f}" if p.get('pe_ratio') else "N/A"
            rating_html = color_rating(p.get('rating', 'N/A'))

            st.markdown(f"""
            <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #4CAF50; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> - {rating_html}<br>
                <small>ROE: {roe_val} | P/E: {pe_val}</small>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**🔴 Short Positions**")
        for i, p in enumerate(short_positions):
            roe_val = f"{p.get('roe', 0):.1f}%" if p.get('roe') else "N/A"
            pe_val = f"{p.get('pe_ratio', 0):.1f}" if p.get('pe_ratio') else "N/A"
            rating_html = color_rating(p.get('rating', 'N/A'))

            st.markdown(f"""
            <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #F44336; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> - {rating_html}<br>
                <small>ROE: {roe_val} | P/E: {pe_val}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Create subplots for metrics
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Return on Equity (%)", "P/E Ratio", "EPS Growth (%)"),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Combine data
    all_positions = []
    for pos in long_positions:
        all_positions.append({**pos, "position": "LONG", "color": "#2ca02c"})
    for pos in short_positions:
        all_positions.append({**pos, "position": "SHORT", "color": "#d62728"})

    if not all_positions:
        st.warning("No positions to display")
        return

    tickers = [p["ticker"] for p in all_positions]
    colors = [p["color"] for p in all_positions]

    # ROE
    roe_values = [p.get("roe") for p in all_positions]
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=roe_values,
            marker_color=colors,
            name="ROE",
            showlegend=False
        ),
        row=1, col=1
    )

    # P/E Ratio
    pe_values = [p.get("pe_ratio") for p in all_positions]
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=pe_values,
            marker_color=colors,
            name="P/E",
            showlegend=False
        ),
        row=2, col=1
    )

    # EPS Growth
    eps_growth_values = [p.get("eps_growth") for p in all_positions]
    fig.add_trace(
        go.Bar(
            x=tickers,
            y=eps_growth_values,
            marker_color=colors,
            name="EPS Growth",
            showlegend=False
        ),
        row=3, col=1
    )

    # Add legend
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='#2ca02c'),
            name='LONG'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='#d62728'),
            name='SHORT'
        )
    )

    fig.update_layout(
        title=config["title"],
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_price_chart(config: dict):
    """Render DJ30 price chart (candlestick, line, or OHLC)."""
    df = pd.DataFrame(config["data"])
    df['date'] = pd.to_datetime(df['date'])

    chart_type = config.get("chart_type", "candlestick")
    include_volume = config.get("include_volume", True)

    if include_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Volume")
        )
    else:
        fig = go.Figure()

    # Add price trace
    if chart_type == "candlestick":
        trace = go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        )
    elif chart_type == "ohlc":
        trace = go.Ohlc(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        )
    else:  # line
        trace = go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name="Price",
            line=dict(color="#1f77b4")
        )

    if include_volume:
        fig.add_trace(trace, row=1, col=1)
        # Add volume trace
        fig.add_trace(
            go.Bar(x=df['date'], y=df['volume'], name="Volume", marker_color="#A9A9A9"),
            row=2, col=1
        )
    else:
        fig.add_trace(trace)

    fig.update_layout(
        title=config["title"],
        xaxis_rangeslider_visible=False,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


def render_performance_comparison(config: dict):
    """Render DJ30 performance comparison chart."""
    series = config["series"]

    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (ticker, data) in enumerate(series.items()):
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])

        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['value'],
            mode='lines',
            name=ticker,
            line=dict(color=colors[i % len(colors)])
        ))

    ylabel = "Normalized Price (Base=100)" if config.get("normalized", False) else "Price ($)"

    fig.update_layout(
        title=config["title"],
        xaxis_title="Date",
        yaxis_title=ylabel,
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_volatility_chart(config: dict):
    """Render DJ30 rolling volatility chart."""
    df = pd.DataFrame(config["data"])
    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['volatility'],
        mode='lines',
        name=f"{config['window']}-day Rolling Volatility",
        line=dict(color="#ff7f0e"),
        fill='tozeroy'
    ))

    fig.update_layout(
        title=config["title"],
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def render_volatility_portfolio(config: dict):
    """Render volatility-based portfolio recommendations."""
    long_positions = config.get("long_positions", [])
    short_positions = config.get("short_positions", [])
    portfolio_type = config.get("portfolio_type", "long_short")

    # Render title
    st.markdown(f"### {config.get('title', 'Volatility-Based Portfolio')}")

    # Determine layout based on what positions exist
    has_long = len(long_positions) > 0
    has_short = len(short_positions) > 0

    if has_long and has_short:
        # Two columns for long/short
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🟢 Long Positions**")
            label = "High Volatility" if portfolio_type == "long_short" or portfolio_type == "long_high_vol" else "Low Volatility"
            st.caption(label)
            for i, p in enumerate(long_positions):
                div_info = f" | Div: {p.get('dividend_yield', 0):.2f}%" if p.get('dividend_yield', 0) > 0 else ""
                st.markdown(f"""
                <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #4CAF50; background-color: var(--secondary-background-color);">
                    <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                    <small>Vol: {p.get('volatility', 0):.2f}% | Return: {p.get('annualized_return', 0):+.2f}%{div_info}</small>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("**🔴 Short Positions**")
            st.caption("Low Volatility" if portfolio_type == "long_short" else "High Volatility")
            for i, p in enumerate(short_positions):
                st.markdown(f"""
                <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #F44336; background-color: var(--secondary-background-color);">
                    <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                    <small>Vol: {p.get('volatility', 0):.2f}% | Return: {p.get('annualized_return', 0):+.2f}%</small>
                </div>
                """, unsafe_allow_html=True)

    elif has_long:
        # Only long positions - single column
        st.markdown("**🟢 Long Positions**")
        if portfolio_type == "long_low_vol":
            st.caption("Low Volatility - Defensive Strategy")
        else:
            st.caption("High Volatility - Aggressive Strategy")

        for i, p in enumerate(long_positions):
            div_info = f" | Div: {p.get('dividend_yield', 0):.2f}%" if p.get('dividend_yield', 0) > 0 else ""
            st.markdown(f"""
            <div style="padding: 10px; margin: 6px 0; border-left: 4px solid #4CAF50; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                <small>Vol: {p.get('volatility', 0):.2f}% | Return: {p.get('annualized_return', 0):+.2f}%{div_info}</small><br>
                <small style="color: var(--text-color); opacity: 0.8;">{p.get('rationale', '')}</small>
            </div>
            """, unsafe_allow_html=True)

    elif has_short:
        # Only short positions - single column
        st.markdown("**🔴 Short Positions**")
        st.caption("High Volatility")

        for i, p in enumerate(short_positions):
            st.markdown(f"""
            <div style="padding: 10px; margin: 6px 0; border-left: 4px solid #F44336; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                <small>Vol: {p.get('volatility', 0):.2f}% | Return: {p.get('annualized_return', 0):+.2f}%</small><br>
                <small style="color: var(--text-color); opacity: 0.8;">{p.get('rationale', '')}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Create volatility comparison bar chart
    tickers = [p['ticker'] for p in long_positions] + [p['ticker'] for p in short_positions]
    volatilities = [p['volatility'] for p in long_positions] + [p['volatility'] for p in short_positions]
    colors = ['#4CAF50'] * len(long_positions) + ['#F44336'] * len(short_positions)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers,
        y=volatilities,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in volatilities],
        textposition='outside'
    ))

    fig.update_layout(
        title="Volatility Comparison",
        xaxis_title="Ticker",
        yaxis_title="Annualized Volatility (%)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_momentum_portfolio(config: dict):
    """Render momentum-based portfolio recommendations."""
    long_positions = config.get("long_positions", [])
    short_positions = config.get("short_positions", [])
    portfolio_type = config.get("portfolio_type", "long_short")

    # Render title
    st.markdown(f"### {config.get('title', 'Momentum-Based Portfolio')}")

    # Determine layout based on what positions exist
    has_long = len(long_positions) > 0
    has_short = len(short_positions) > 0

    if has_long and has_short:
        # Two columns for long/short
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🟢 Long Positions (High Momentum)**")
            for i, p in enumerate(long_positions):
                st.markdown(f"""
                <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #4CAF50; background-color: var(--secondary-background-color);">
                    <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                    <small>Momentum: {p.get('momentum', 0):+.2f}% | Vol: {p.get('volatility', 0):.2f}%</small>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("**🔴 Short Positions (Low Momentum)**")
            for i, p in enumerate(short_positions):
                st.markdown(f"""
                <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #F44336; background-color: var(--secondary-background-color);">
                    <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                    <small>Momentum: {p.get('momentum', 0):+.2f}% | Vol: {p.get('volatility', 0):.2f}%</small>
                </div>
                """, unsafe_allow_html=True)

    elif has_long:
        # Only long positions - single column
        st.markdown("**🟢 Long Positions**")
        st.caption("High Momentum - Trend Following Strategy")

        for i, p in enumerate(long_positions):
            st.markdown(f"""
            <div style="padding: 10px; margin: 6px 0; border-left: 4px solid #4CAF50; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                <small>Momentum: {p.get('momentum', 0):+.2f}% | Vol: {p.get('volatility', 0):.2f}%</small><br>
                <small style="color: var(--text-color); opacity: 0.8;">{p.get('rationale', '')}</small>
            </div>
            """, unsafe_allow_html=True)

    elif has_short:
        # Only short positions - single column
        st.markdown("**🔴 Short Positions**")
        st.caption("Low Momentum - Contrarian Strategy")

        for i, p in enumerate(short_positions):
            st.markdown(f"""
            <div style="padding: 10px; margin: 6px 0; border-left: 4px solid #F44336; background-color: var(--secondary-background-color);">
                <strong>#{p.get('rank', i+1)}. {p['ticker']}</strong> ({p.get('sector', 'N/A')})<br>
                <small>Momentum: {p.get('momentum', 0):+.2f}% | Vol: {p.get('volatility', 0):.2f}%</small><br>
                <small style="color: var(--text-color); opacity: 0.8;">{p.get('rationale', '')}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Create momentum comparison bar chart
    tickers = [p['ticker'] for p in long_positions] + [p['ticker'] for p in short_positions]
    momentum = [p['momentum'] for p in long_positions] + [p['momentum'] for p in short_positions]
    colors = ['#4CAF50'] * len(long_positions) + ['#F44336'] * len(short_positions)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers,
        y=momentum,
        marker_color=colors,
        text=[f"{m:+.1f}%" for m in momentum],
        textposition='outside'
    ))

    fig.update_layout(
        title="Momentum Comparison",
        xaxis_title="Ticker",
        yaxis_title="Cumulative Return (%)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def render_sector_portfolio(config: dict):
    """Render sector-diversified portfolio."""
    positions = config["positions"]

    st.markdown("### Sector-Diversified Portfolio")

    # Group by sector
    sectors = {}
    for p in positions:
        sector = p.get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(p)

    # Display by sector
    for sector, stocks in sectors.items():
        st.markdown(f"**{sector}**")
        for p in stocks:
            st.markdown(f"""
            <div style="padding: 8px; margin: 4px 0; border-left: 3px solid #2196F3; background-color: var(--secondary-background-color);">
                <strong>{p['ticker']}</strong><br>
                <small>Return: {p.get('total_return', 0):+.2f}% | Sharpe: {p.get('sharpe_ratio', 0):.2f}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Create sector allocation pie chart
    sector_counts = {sector: len(stocks) for sector, stocks in sectors.items()}

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=list(sector_counts.keys()),
        values=list(sector_counts.values()),
        hole=0.3
    ))

    fig.update_layout(
        title="Sector Allocation",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def run_analysis_with_logs(user_input: str, conversation_history: list, selected_columns: list = None) -> str:
    """Run analysis and capture stdout/stderr to display in logs."""
    # Set selected columns in the tool module before running analysis
    # This ensures the tool reads the deterministic UI selection, not agent-provided values
    from tools.agentics_generic_tools import set_selected_columns
    set_selected_columns(selected_columns)

    # Create a StringIO object to capture output
    captured_output = StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Redirect stdout and stderr to capture output
        sys.stdout = captured_output
        sys.stderr = captured_output

        # Run the analysis (selected columns are already set in tool module)
        response = run_analysis(user_input, conversation_history, selected_columns)

        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Get captured output and clean ANSI escape codes
        raw_logs = captured_output.getvalue()
        # Remove ANSI color codes (e.g., [36m, [0m, [1;36m, etc.)
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        cleaned_logs = ansi_escape.sub('', raw_logs)

        # Store the cleaned logs in session state
        st.session_state.agent_logs = cleaned_logs

        return response

    except Exception as e:
        # Restore original stdout/stderr in case of error
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Clean and store what we captured plus the error
        raw_logs = captured_output.getvalue()
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        cleaned_logs = ansi_escape.sub('', raw_logs)
        st.session_state.agent_logs = cleaned_logs + f"\n\nError: {str(e)}"
        raise e


def extract_visualization_ids(response: str) -> list:
    """Extract visualization IDs from agent response."""
    import re
    import glob
    from datetime import datetime, timedelta

    # First, try to extract from response text
    viz_ids = re.findall(r'viz_\d{8}_\d{6}_[a-f0-9]{8}', response)

    # Fallback: Check for recently created visualization files (within last 30 seconds)
    # This handles cases where the agent doesn't include the exact ID text
    viz_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    if os.path.exists(viz_dir):
        now = datetime.now().timestamp()
        recent_files = []
        for viz_file in glob.glob(os.path.join(viz_dir, "viz_*.json")):
            file_age = now - os.path.getmtime(viz_file)
            if file_age < 30:  # Created within last 30 seconds
                viz_id = os.path.basename(viz_file).replace('.json', '')
                if viz_id not in viz_ids:
                    recent_files.append(viz_id)
        viz_ids.extend(recent_files)

    return list(set(viz_ids))


# Sidebar
with st.sidebar:
    st.markdown("#### 📁 Available Data")

    with st.expander("📈 Dataset Information"):
        data_summary = get_data_summary()
        firm_summary = get_firm_data_summary()
        dj30_summary = get_dj30_data_summary()

        st.markdown("**Macro Factors**")
        st.markdown(f"- **Date Range:** {data_summary['macro_factors']['date_range']['start']} to {data_summary['macro_factors']['date_range']['end']}")
        st.markdown(f"- **Records:** {data_summary['macro_factors']['rows']:,}")
        st.markdown(f"- **Indicators:** {len(data_summary['macro_factors']['columns'])}")

        st.markdown("**Market Factors**")
        st.markdown(f"- **Date Range:** {data_summary['market_factors']['date_range']['start']} to {data_summary['market_factors']['date_range']['end']}")
        st.markdown(f"- **Records:** {data_summary['market_factors']['rows']:,}")
        st.markdown(f"- **Indicators:** {len(data_summary['market_factors']['columns'])}")

        st.markdown("**Company Fundamentals**")
        st.markdown(f"- **Date Range:** {firm_summary['date_range']['start']} to {firm_summary['date_range']['end']}")
        st.markdown(f"- **Records:** {firm_summary['total_records']:,}")
        st.markdown(f"- **Companies:** {firm_summary['unique_tickers']}")
        st.markdown(f"- **Metrics:** EPS, ROE, ROA, P/E, Margins, Growth")

        st.markdown(dj30_summary)

    with st.expander("📋 Available Indicators"):
        descriptions = get_column_descriptions()
        firm_descriptions = get_firm_column_descriptions()
        dj30_descriptions = get_dj30_column_descriptions()

        st.markdown("**Macroeconomic Indicators:**")
        for indicator, desc in descriptions["macro_factors"].items():
            st.markdown(f"- **{indicator}**: {desc}")

        st.markdown("\n**Market Indicators:**")
        for indicator, desc in descriptions["market_factors"].items():
            if indicator != "Headlines":
                st.markdown(f"- **{indicator}**: {desc}")

        st.markdown("\n**Company Fundamental Metrics:**")
        # Show key metrics only (not all the forward growth/volatility variants)
        key_metrics = ["TICKER", "STATPERS", "PRICE", "EBS", "EPS", "DPS", "ROA", "ROE", "NAV", "GRM"]
        for metric in key_metrics:
            if metric in firm_descriptions:
                st.markdown(f"- **{metric}**: {firm_descriptions[metric]}")
        st.markdown("- Plus forward 1-year growth and volatility estimates for all metrics")

        st.markdown("\n**DJ30 Stock Price Data:**")
        for category, metrics in dj30_descriptions.items():
            st.markdown(f"\n*{category}:*")
            for metric in metrics:
                st.markdown(f"  {metric}")

    st.markdown("---")

    # Transduction Column Selection
    with st.expander("🔧 Transduction Column Selection", expanded=False):
        st.markdown("**Select columns to include in transduction analysis:**")
        st.caption("Date column is always included. Select additional columns to analyze.")

        all_columns = get_merged_data_columns()
        if all_columns:
            # Search box to filter columns
            search_term = st.text_input("🔍 Search columns:", placeholder="Type to filter columns...", key="column_search")

            # Filter columns based on search
            if search_term:
                search_lower = search_term.lower()
                all_columns = [col for col in all_columns if search_lower in col.lower()]
                if not all_columns:
                    st.info("No columns match your search.")
                    st.stop()
            # Separate Date from other columns
            date_col = "Date"
            other_columns = [col for col in all_columns if col != date_col]

            # Group columns by category for better UX
            macro_cols = [col for col in other_columns if col in ['FEDFUNDS', 'TB3MS', 'T10Y3M', 'CPIAUCSL', 'CPILFESL', 'PCEPI', 'PCEPILFE', 'UNRATE', 'PAYEMS', 'INDPRO', 'RSAFS']]
            market_cols = [col for col in other_columns if col.startswith('^') or col in ['BTC-USD', 'GSG', 'DGS2', 'DGS10', 'DTWEXBGS', 'DCOILBRENTEU', 'GLD', 'US10Y2Y', 'Headlines']]
            dj30_price_cols = [col for col in other_columns if any(col.startswith(prefix) for prefix in ['open_', 'high_', 'low_', 'close_', 'adj_close_', 'volume_', 'dividend_'])]
            fundamental_cols = [col for col in other_columns if any(col.endswith(suffix) for suffix in ['_MEDEST', '_MEANEST', '_ACTUAL'])]
            news_cols = [col for col in other_columns if col.startswith('news_') or col.startswith('Earningcall_')]
            other_cols = [col for col in other_columns if col not in macro_cols + market_cols + dj30_price_cols + fundamental_cols + news_cols]

            # Initialize selected columns if empty
            if "selected_transduction_columns" not in st.session_state:
                st.session_state.selected_transduction_columns = []

            # Build selected list from session state first (to ensure it reflects programmatic changes)
            # This ensures programmatically set columns are included
            # Make a copy to avoid reference issues
            selected = list(st.session_state.selected_transduction_columns) if st.session_state.selected_transduction_columns else []

            # Category selection with checkboxes
            # Macro columns
            if macro_cols:
                with st.expander(f"📊 Macroeconomic Indicators ({len(macro_cols)})", expanded=False):
                    for col in sorted(macro_cols):
                        # Use session state key for checkbox to maintain state
                        checkbox_key = f"col_{col}"
                        # Initialize from session state if not set
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                        is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                        # Update selected list based on checkbox state
                        if is_checked:
                            if col not in selected:
                                selected.append(col)
                        else:
                            if col in selected:
                                selected.remove(col)

            # Market columns
            if market_cols:
                with st.expander(f"📈 Market Factors ({len(market_cols)})", expanded=False):
                    for col in sorted(market_cols):
                        # Use session state key for checkbox to maintain state
                        checkbox_key = f"col_{col}"
                        # Initialize from session state if not set
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                        is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                        # Update selected list based on checkbox state
                        if is_checked:
                            if col not in selected:
                                selected.append(col)
                        else:
                            if col in selected:
                                selected.remove(col)

            # DJ30 Price columns
            if dj30_price_cols:
                with st.expander(f"💹 DJ30 Stock Prices ({len(dj30_price_cols)})", expanded=False):
                    # Group by ticker for better organization
                    ticker_groups = {}
                    for col in dj30_price_cols:
                        parts = col.split('_')
                        if len(parts) >= 2:
                            ticker = parts[-1]  # Last part is usually ticker
                            if ticker not in ticker_groups:
                                ticker_groups[ticker] = []
                            ticker_groups[ticker].append(col)

                    for ticker in sorted(ticker_groups.keys()):
                        with st.expander(f"  {ticker} ({len(ticker_groups[ticker])} columns)", expanded=False):
                            for col in sorted(ticker_groups[ticker]):
                                # Use session state key for checkbox to maintain state
                                checkbox_key = f"col_{col}"
                                # Initialize from session state if not set
                                if checkbox_key not in st.session_state:
                                    st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                                is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                                # Update selected list based on checkbox state
                                if is_checked:
                                    if col not in selected:
                                        selected.append(col)
                                else:
                                    if col in selected:
                                        selected.remove(col)

            # Fundamental columns
            if fundamental_cols:
                with st.expander(f"🏢 Company Fundamentals ({len(fundamental_cols)})", expanded=False):
                    # Group by metric type
                    metric_groups = {}
                    for col in fundamental_cols:
                        parts = col.split('_')
                        if len(parts) >= 2:
                            metric = parts[0]  # First part is metric
                            if metric not in metric_groups:
                                metric_groups[metric] = []
                            metric_groups[metric].append(col)

                    for metric in sorted(metric_groups.keys()):
                        with st.expander(f"  {metric} ({len(metric_groups[metric])} columns)", expanded=False):
                            for col in sorted(metric_groups[metric]):
                                # Use session state key for checkbox to maintain state
                                checkbox_key = f"col_{col}"
                                # Initialize from session state if not set
                                if checkbox_key not in st.session_state:
                                    st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                                is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                                # Update selected list based on checkbox state
                                if is_checked:
                                    if col not in selected:
                                        selected.append(col)
                                else:
                                    if col in selected:
                                        selected.remove(col)

            # News columns
            if news_cols:
                with st.expander(f"📰 News & Earnings Calls ({len(news_cols)})", expanded=False):
                    for col in sorted(news_cols):
                        # Use session state key for checkbox to maintain state
                        checkbox_key = f"col_{col}"
                        # Initialize from session state if not set
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                        is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                        # Update selected list based on checkbox state
                        if is_checked:
                            if col not in selected:
                                selected.append(col)
                        else:
                            if col in selected:
                                selected.remove(col)

            # Other columns
            if other_cols:
                with st.expander(f"🔹 Other Columns ({len(other_cols)})", expanded=False):
                    for col in sorted(other_cols):
                        # Use session state key for checkbox to maintain state
                        checkbox_key = f"col_{col}"
                        # Initialize from session state if not set
                        if checkbox_key not in st.session_state:
                            st.session_state[checkbox_key] = col in st.session_state.selected_transduction_columns

                        is_checked = st.checkbox(col, value=st.session_state[checkbox_key], key=checkbox_key)
                        # Update selected list based on checkbox state
                        if is_checked:
                            if col not in selected:
                                selected.append(col)
                        else:
                            if col in selected:
                                selected.remove(col)

            # Always update session state with current selection (removes duplicates)
            # This ensures the session state reflects both programmatic changes and user interactions
            final_selected = list(set(selected))
            st.session_state.selected_transduction_columns = final_selected


            # Show summary - use the final selected count
            # Use session state for display to ensure consistency
            total_selected = len(st.session_state.selected_transduction_columns)
            st.caption(f"📊 **{total_selected} columns selected** (Date always included)")

            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Select All", use_container_width=True):
                    st.session_state.selected_transduction_columns = other_columns
                    st.rerun()
            with col2:
                if st.button("❌ Clear All", use_container_width=True):
                    st.session_state.selected_transduction_columns = []
                    st.rerun()
        else:
            st.warning("Could not load column names from merged_data.csv")

    st.markdown("---")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.current_visualizations = []
        st.session_state.agent_logs = ""
        clean_old_visualizations()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 💡 Example Questions")
    st.caption("Click any question to load it and auto-select relevant columns")

    # 5 carefully chosen diagnostic questions with manually pre-selected columns
    questions = [
        "Explain how AMZN and AAPL's strategy shifted over time, and any major investments they made from 2020 onwards",
        "Why did the stock price of AAPL drop in March 2020?",
        "What were the key factors behind NVDA's stock price surge in 2023?",
        "Analyze the relationship between MSFT's earnings announcements and its stock price movements from 2020-2023",
        "What market events and company-specific news drove JPM's volatility during the 2022-2023 period?"
    ]

    for i, q in enumerate(questions, 1):
        if st.button(f"{i}. {q}", key=f"q_{i}", use_container_width=True):
            set_example_question(q)

# Page selector in sidebar
with st.sidebar:
    st.markdown("### 📑 Navigation")
    page = st.radio(
        "Select Page",
        ["Chat", "Transduction Flow"],
        index=0 if st.session_state.current_page == "Chat" else 1,
        key="page_selector"
    )
    st.session_state.current_page = page
    st.markdown("---")

# Agent Logs Sidebar (Left) - Always visible
with st.sidebar:
    st.markdown("#### 🔍 Agent Thought Process")
    st.caption("View the agent's reasoning and tool calls in real-time")

    if st.session_state.agent_logs:
        # Show logs in expandable section
        with st.expander("📜 View Agent Logs", expanded=st.session_state.show_logs):
            st.code(st.session_state.agent_logs, language="text")

    else:
        st.info("💡 Agent logs will appear here once you ask a question. You'll be able to see the agent's tool usage, reasoning process, and decision-making in real-time!")

# Main content - Show different pages based on selection
if st.session_state.current_page == "Transduction Flow":
    display_transduction_flow()
else:
    # Chat page
    st.markdown('<div class="main-header">Financial Data Analyst</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about macroeconomic data, market factors, and company fundamentals from 2018 to present</div>', unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{content}</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>🤖 Analyst:</strong><br>{content}</div>',
                       unsafe_allow_html=True)

            # Display visualizations for this message
            if "visualizations" in message:
                for viz_id in message["visualizations"]:
                    load_visualization(viz_id)

    # User input
    with st.container():
        # If clear_input flag is set, reset the text area
        if st.session_state.clear_input:
            st.session_state.user_input = ""
            st.session_state.clear_input = False

        user_input = st.text_area(
            "Ask your question:",
            placeholder="E.g., How did market volatility change during the 2020 pandemic?",
            height=100,
            key="user_input"
        )

        col1, col2 = st.columns([8, 1])
        with col2:
            submit_button = st.button("Analyze", type="primary")

    # Process user input
    if submit_button and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Clean old visualizations from previous turn
        clean_old_visualizations()
        st.session_state.current_visualizations = []

        # Show loading state
        with st.spinner("Thinking..."):
            try:
                # Run analysis with conversation history for context and capture logs
                # Pass selected columns for transduction filtering
                selected_cols = st.session_state.get("selected_transduction_columns", [])
                response = run_analysis_with_logs(
                    user_input,
                    st.session_state.messages,
                    selected_columns=selected_cols if selected_cols else None
                )

                # Extract visualization IDs from response
                viz_ids = extract_visualization_ids(response)
                st.session_state.current_visualizations = viz_ids

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "visualizations": viz_ids
                })

                # Clear previous transduction flow when new analysis starts
                # The new flow will be set by the transduction tool
                # This ensures we only show the latest flow

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}. Please try rephrasing your question.",
                    "visualizations": []
                })

        # Set flag to clear input on next run
        st.session_state.clear_input = True

        # Rerun to display new messages
        st.rerun()

