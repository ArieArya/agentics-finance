# Financial Data Analyst with CrewAI

An intelligent financial data analysis application powered by CrewAI agents, capable of analyzing macroeconomic and market data through natural language Q&A conversations.

## 🌟 Features

- **Multi-turn Q&A Interface**: Interactive Streamlit chat interface for natural conversations about financial data
- **Comprehensive Data Analysis**: Access to macro factors (Fed Funds Rate, CPI, Unemployment, etc.) and market factors (S&P 500, VIX, Bitcoin, Oil, Gold, etc.)
- **Intelligent Agent**: GPT-4 powered CrewAI agent with specialized financial analysis tools
- **Beautiful Visualizations**: Clean, interactive Plotly charts including time series, correlation heatmaps, volatility plots, and distributions
- **Multi-turn Visualization Management**: Automatic cleanup of visualizations between conversation turns

## 📊 Available Data

### Macroeconomic Indicators
- **FEDFUNDS**: Federal Funds Effective Rate
- **TB3MS**: 3-Month Treasury Bill Rate
- **T10Y3M**: 10-Year minus 3-Month Treasury Spread
- **CPIAUCSL**: Consumer Price Index
- **CPILFESL**: Core CPI (excl. Food & Energy)
- **PCEPI**: Personal Consumption Expenditures Price Index
- **PCEPILFE**: Core PCE Price Index
- **UNRATE**: Unemployment Rate
- **PAYEMS**: Total Nonfarm Payroll Employment
- **INDPRO**: Industrial Production Index
- **RSAFS**: Retail Sales

### Market Indicators
- **^GSPC**: S&P 500 Index
- **^STOXX50E**: Euro Stoxx 50 Index
- **BTC-USD**: Bitcoin Price
- **^VIX**: CBOE Volatility Index
- **GSG**: Commodity Index
- **DGS2**: 2-Year Treasury Rate
- **DGS10**: 10-Year Treasury Rate
- **DTWEXBGS**: Trade Weighted U.S. Dollar Index
- **DCOILBRENTEU**: Brent Crude Oil Price
- **GLD**: Gold ETF
- **US10Y2Y**: 10Y-2Y Treasury Yield Spread
- **Headlines**: Daily news headlines

**Data Coverage**: July 2008 - Present (Daily frequency)

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Gemini API key (from https://aistudio.google.com/app/apikey)

### Installation

1. Clone the repository:
```bash
cd agentics-finance
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

> **Note**: The Agentics framework is automatically installed from the local `Agentics/` directory in editable mode via requirements.txt.

4. Set up environment variables:
```bash
# Copy the example file
cp env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_gemini_api_key_here
```

### Running the Application

Start the Streamlit application:
```bash
uv run streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`