"""
Streamlit application for financial data Q&A with CrewAI agents.
"""

import streamlit as st
import os
import json
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from agents import run_analysis
from utils import get_data_summary, get_column_descriptions

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
        background-color: #1a2332;
        border-left-color: #1f77b4;
        color: #ffffff;
    }
    .assistant-message {
        background-color: #1e1e1e;
        border-left-color: #2ca02c;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #ffffff;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_visualizations" not in st.session_state:
    st.session_state.current_visualizations = []

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False


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
    else:
        st.warning(f"Unknown visualization type: {viz_type}")


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


def extract_visualization_ids(response: str) -> list:
    """Extract visualization IDs from agent response."""
    import re
    viz_ids = re.findall(r'viz_\d{8}_\d{6}_[a-f0-9]{8}', response)
    return list(set(viz_ids))


# Sidebar
with st.sidebar:
    st.markdown("#### 📁 Available Data")

    with st.expander("📈 Dataset Information"):
        data_summary = get_data_summary()

        st.markdown("**Macro Factors**")
        st.markdown(f"- **Date Range:** {data_summary['macro_factors']['date_range']['start']} to {data_summary['macro_factors']['date_range']['end']}")
        st.markdown(f"- **Records:** {data_summary['macro_factors']['rows']:,}")
        st.markdown(f"- **Indicators:** {len(data_summary['macro_factors']['columns'])}")

        st.markdown("**Market Factors**")
        st.markdown(f"- **Date Range:** {data_summary['market_factors']['date_range']['start']} to {data_summary['market_factors']['date_range']['end']}")
        st.markdown(f"- **Records:** {data_summary['market_factors']['rows']:,}")
        st.markdown(f"- **Indicators:** {len(data_summary['market_factors']['columns'])}")

    with st.expander("📋 Available Indicators"):
        descriptions = get_column_descriptions()

        st.markdown("**Macroeconomic Indicators:**")
        for indicator, desc in descriptions["macro_factors"].items():
            st.markdown(f"- **{indicator}**: {desc}")

        st.markdown("\n**Market Indicators:**")
        for indicator, desc in descriptions["market_factors"].items():
            if indicator != "Headlines":
                st.markdown(f"- **{indicator}**: {desc}")

    st.markdown("---")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.current_visualizations = []
        clean_old_visualizations()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 💡 Example Questions")

    with st.expander("📊 Market Analysis"):
        st.markdown("""
        - Compare performance of S&P 500, Gold, and Bitcoin from 2020 to 2023
        - What was the maximum drawdown during the 2008 financial crisis?
        - Show me S&P 500 with 50 and 200-day moving averages
        - Calculate monthly returns for Bitcoin in 2021
        """)

    with st.expander("📈 Economic Analysis"):
        st.markdown("""
        - What are the year-over-year inflation trends from 2020 to 2023?
        - Create a dashboard showing unemployment, inflation, and retail sales during COVID
        - How much did the unemployment rate change from 2019 to 2021?
        - Show the relationship between oil prices and inflation
        """)

    with st.expander("⚠️ Risk & Volatility"):
        st.markdown("""
        - Why was the S&P 500 so volatile in March 2020?
        - Explain the volatility spike in oil prices during 2008
        - What caused Bitcoin's extreme volatility in 2021?
        - What indicators moved together during the 2008 crisis?
        - Identify correlated movements on March 11, 2020
        - What was the volatility of the S&P 500 during March 2020?
        - Show me the drawdown chart for Bitcoin from 2021 to 2022
        - Find the most volatile periods for oil prices
        - Analyze drawdowns and recovery time for the stock market
        """)

    with st.expander("🔗 Correlations & Relationships"):
        st.markdown("""
        - Show me the correlation between VIX and S&P 500
        - Create a scatter plot of unemployment vs stock market performance
        - What's the correlation between gold, Bitcoin, and stocks?
        - Analyze the relationship between interest rates and inflation
        """)

    with st.expander("📰 News & Events"):
        st.markdown("""
        - What were the most popular news on January 22nd, 2012?
        - Show me headlines from the 2008 financial crisis
        - What major events affected the market during the COVID pandemic?
        - Create a timeline of significant market events from 2020-2022
        """)

# Main content
st.markdown('<div class="main-header">Financial Data Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about macroeconomic and market data from 2008 to present</div>', unsafe_allow_html=True)

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
        placeholder="E.g., What was the S&P 500 volatility during the 2008 crisis?",
        height=100,
        key="user_input"
    )

    col1, col2 = st.columns([10, 1])
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
            # Run analysis with conversation history for context
            response = run_analysis(user_input, st.session_state.messages)

            # Extract visualization IDs from response
            viz_ids = extract_visualization_ids(response)
            st.session_state.current_visualizations = viz_ids

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "visualizations": viz_ids
            })

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

