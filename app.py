"""
Streamlit application for financial data Q&A using Transduction.
Simplified version focused purely on transduction without CrewAI or tools.
"""

import streamlit as st
import os
import json
from datetime import datetime, date
from typing import List, Dict, Any
from transduction_pipeline import TransductionPipeline
import asyncio

from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Financial Data Analyst - Transduction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.75rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1557a0;
    }
    .metadata-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    /* Allow sidebar to be resized larger */
    [data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 50% !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = []


def get_merged_data_columns() -> List[str]:
    """Get all column names from merged_data.csv."""
    from utils.csv_reader import read_merged_data_header

    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        header = read_merged_data_header(data_dir)
        return header
    except Exception as e:
        st.error(f"Error reading column names: {e}")
        return []


def format_answer(result: Dict[str, Any]) -> str:
    """Format the transduction result into a readable answer."""
    if not result.get("success"):
        return f"❌ Error: {result.get('error', 'Unknown error occurred')}"

    answer_parts = []

    # Main answer
    if result.get("detailed_answer"):
        answer_parts.append(result["detailed_answer"])

    # Explanation
    if result.get("explanation"):
        answer_parts.append(f"\n\n**Explanation:**\n{result['explanation']}")

    # Metadata
    date_range = result.get("date_range", {})
    if date_range:
        metadata = f"""
<div class="metadata-box">
<b>Analysis Details:</b><br>
• Date Range: {date_range.get('start')} to {date_range.get('end')}<br>
• Rows in Range: {date_range.get('total_rows', 0):,}<br>
• Rows Analyzed: {date_range.get('analyzed_rows', 0):,}
{f"(sampled from {date_range.get('total_rows', 0):,})" if date_range.get('sampling_applied') else ""}<br>
• Batches Processed: {date_range.get('num_batches', 0)}
</div>
"""
        answer_parts.append(metadata)

    return "\n\n".join(answer_parts)


def clear_conversation():
    """Clear conversation history."""
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.session_state.pipeline = None


# Sidebar - Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Date Range Selector
    st.markdown("#### 📅 Analysis Date Range")
    st.caption("Select the time period for your analysis (2018-2025)")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date(2020, 1, 1),
            min_value=date(2018, 1, 1),
            max_value=date(2025, 12, 31),
            key="start_date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date(2023, 12, 31),
            min_value=date(2018, 1, 1),
            max_value=date(2025, 12, 31),
            key="end_date"
        )

    # Validate date range
    if start_date >= end_date:
        st.error("⚠️ Start date must be before end date!")

    st.markdown("---")

    # Column Selection (Optional)
    st.markdown("#### 📊 Column Selection (Optional)")
    st.caption("Select specific columns to focus the analysis, or leave empty to use all columns")

    all_columns = get_merged_data_columns()

    if all_columns:
        # Categorize columns for easier selection
        price_cols = [c for c in all_columns if any(x in c.lower() for x in ['open', 'high', 'low', 'close', 'volume', 'adj'])]
        news_cols = [c for c in all_columns if 'news' in c.lower() or 'headline' in c.lower() or 'earningcall' in c.lower()]
        fundamental_cols = [c for c in all_columns if any(x in c.upper() for x in ['EPS', 'ROE', 'ROA', 'SAL', 'NET', 'EBITDA'])]
        macro_cols = [c for c in all_columns if c.startswith('^') or any(x in c.lower() for x in ['gdp', 'inflation', 'rate'])]
        other_cols = [c for c in all_columns if c not in price_cols + news_cols + fundamental_cols + macro_cols and c != 'Date']

        # Summary and clear button at the top
        if st.session_state.selected_columns:
            st.info(f"✅ {len(st.session_state.selected_columns)} columns selected")
        else:
            st.info("ℹ️ No columns selected - will use all columns")
        if st.button("🔄 Clear", use_container_width=True, key="btn_clear"):
            st.session_state.selected_columns = []
            st.rerun()

        # Category-based selection
        if price_cols:
            with st.expander(f"📈 Price Data ({len(price_cols)} columns)", expanded=False):
                selected_price = st.multiselect(
                    "Select price data columns:",
                    options=price_cols,
                    default=[c for c in price_cols if c in st.session_state.selected_columns],
                    key="price_multiselect",
                    help="Stock price data including Open, High, Low, Close, Volume, Adjusted Close"
                )
                # Update session state
                for col in price_cols:
                    if col in selected_price and col not in st.session_state.selected_columns:
                        st.session_state.selected_columns.append(col)
                    elif col not in selected_price and col in st.session_state.selected_columns:
                        st.session_state.selected_columns.remove(col)

        if news_cols:
            with st.expander(f"📰 News & Earnings ({len(news_cols)} columns)", expanded=False):
                selected_news = st.multiselect(
                    "Select news and earnings columns:",
                    options=news_cols,
                    default=[c for c in news_cols if c in st.session_state.selected_columns],
                    key="news_multiselect",
                    help="Company news headlines and earnings call summaries"
                )
                # Update session state
                for col in news_cols:
                    if col in selected_news and col not in st.session_state.selected_columns:
                        st.session_state.selected_columns.append(col)
                    elif col not in selected_news and col in st.session_state.selected_columns:
                        st.session_state.selected_columns.remove(col)

        if fundamental_cols:
            with st.expander(f"💼 Fundamentals ({len(fundamental_cols)} columns)", expanded=False):
                selected_fundamentals = st.multiselect(
                    "Select fundamental data columns:",
                    options=fundamental_cols,
                    default=[c for c in fundamental_cols if c in st.session_state.selected_columns],
                    key="fundamentals_multiselect",
                    help="Company fundamentals like EPS, ROE, ROA, sales, margins, etc."
                )
                # Update session state
                for col in fundamental_cols:
                    if col in selected_fundamentals and col not in st.session_state.selected_columns:
                        st.session_state.selected_columns.append(col)
                    elif col not in selected_fundamentals and col in st.session_state.selected_columns:
                        st.session_state.selected_columns.remove(col)

        if macro_cols:
            with st.expander(f"🌍 Macro & Market Data ({len(macro_cols)} columns)", expanded=False):
                selected_macro = st.multiselect(
                    "Select macroeconomic and market data columns:",
                    options=macro_cols,
                    default=[c for c in macro_cols if c in st.session_state.selected_columns],
                    key="macro_multiselect",
                    help="Macroeconomic indicators, market indices, sector performance"
                )
                # Update session state
                for col in macro_cols:
                    if col in selected_macro and col not in st.session_state.selected_columns:
                        st.session_state.selected_columns.append(col)
                    elif col not in selected_macro and col in st.session_state.selected_columns:
                        st.session_state.selected_columns.remove(col)

        if other_cols:
            with st.expander(f"📋 Other Columns ({len(other_cols)} columns)", expanded=False):
                selected_other = st.multiselect(
                    "Select other columns:",
                    options=other_cols,
                    default=[c for c in other_cols if c in st.session_state.selected_columns],
                    key="other_multiselect",
                    help="Other data columns not categorized above"
                )
                # Update session state
                for col in other_cols:
                    if col in selected_other and col not in st.session_state.selected_columns:
                        st.session_state.selected_columns.append(col)
                    elif col not in selected_other and col in st.session_state.selected_columns:
                        st.session_state.selected_columns.remove(col)

    st.markdown("---")

    # Example Questions
    st.markdown("#### 💡 Example Questions")
    st.caption("Click to load a question")

    example_questions = [
        ("How did AAPL and AMZN's strategies shift over time, and what major investments did they make?", None),
        ("What market events and company news drove JPM's performance?", None),
        ("Analyze the relationship between market volatility and tech stock performance", None),
        ("How did macroeconomic factors influence the financial sector?", None),
        ("What were the key drivers of NVDA's growth during this period?", None)
    ]

    for i, (question, cols) in enumerate(example_questions, 1):
        display_text = question[:50] + "..." if len(question) > 50 else question
        if st.button(f"{i}. {display_text}", key=f"example_{i}", use_container_width=True):
            st.session_state.example_question = question
            if cols:
                st.session_state.selected_columns = cols
            st.rerun()

    st.markdown("---")

    # Actions
    st.markdown("#### 🔧 Actions")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        clear_conversation()
        st.rerun()

    # Dataset Info
    with st.expander("📁 Dataset Information"):
        st.markdown("""
**Financial Dataset (2018-2025)**

- **Macroeconomic Indicators**: GDP, inflation, interest rates, etc.
- **Market Factors**: S&P 500, VIX, sector indices
- **DJ30 Stock Prices**: Daily OHLCV data
- **Company Fundamentals**: EPS, ROE, ROA, margins, growth metrics
- **News & Earnings**: Company news and earnings call summaries

All analysis uses **transduction** to reduce large datasets into meaningful insights.
        """)

# Main content area
st.markdown('<div class="main-header">📊 Financial Data Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about financial data using Transduction (2018-2025)</div>', unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(f'<div class="chat-message user-message"><b>You:</b><br>{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant-message"><b>Assistant:</b><br>{content}</div>', unsafe_allow_html=True)

# Input area
with st.form(key="question_form", clear_on_submit=True):
    st.markdown("#### Ask a Question")

    # Check if an example question was selected
    default_value = st.session_state.pop("example_question", "")

    user_input = st.text_area(
        "Your question:",
        value=default_value,
        placeholder="E.g., How did market volatility change during the 2020 pandemic?",
        height=100,
        key="question_input"
    )

    submit_button = st.form_submit_button("🔍 Analyze", use_container_width=True)

if submit_button and user_input:
    # Validate date range
    if start_date >= end_date:
        st.error("⚠️ Please select a valid date range (start date must be before end date)")
    else:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Show processing message
        with st.spinner("🔄 Analyzing data using transduction... This may take a moment..."):
            try:
                # Initialize pipeline if needed
                if st.session_state.pipeline is None:
                    st.session_state.pipeline = TransductionPipeline()

                # Run transduction analysis
                result = asyncio.run(st.session_state.pipeline.answer_question(
                    question=user_input,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    selected_columns=st.session_state.selected_columns if st.session_state.selected_columns else None,
                    conversation_history=st.session_state.conversation_history
                ))

                # Format and display answer
                formatted_answer = format_answer(result)

                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_answer
                })

                # Update conversation history (for context in future questions)
                if result.get("success"):
                    st.session_state.conversation_history.append({
                        "question": user_input,
                        "answer": result.get("detailed_answer", ""),
                        "date_range": f"{start_date} to {end_date}"
                    })

            except Exception as e:
                error_msg = f"❌ Error during analysis: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.error(error_msg)

        # Rerun to show new messages
        st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by Agentics Transduction Framework | Data: 2018-2025")

