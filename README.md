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
pip install -r requirements.txt
```

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
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🛠️ Project Structure

```
agentics-finance/
├── agents/                  # CrewAI agent configurations
│   ├── __init__.py
│   └── financial_analyst.py # Main financial analyst agent
├── tools/                   # Custom CrewAI tools
│   ├── __init__.py
│   ├── data_query_tools.py  # Data querying tools
│   ├── analysis_tools.py    # Analysis tools (volatility, correlation)
│   └── visualization_tools.py # Visualization generation tools
├── utils/                   # Utility modules
│   ├── __init__.py
│   └── data_loader.py       # Data loading and caching
├── data/                    # CSV data files
│   ├── macro_factors_new.csv
│   └── market_factors_new.csv
├── visualizations/          # Generated visualization JSON files (temporary)
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── env.example             # Environment variables template
└── README.md               # This file
```

## 🔧 Available Tools

The CrewAI agent has access to 10 specialized tools:

### Data Query Tools
1. **List Available Indicators**: Discover all available indicators
2. **Query Data by Date Range**: Fetch data for specific indicators and time periods
3. **Get Indicator Statistics**: Calculate summary statistics for any indicator

### Analysis Tools
4. **Analyze Volatility**: Calculate rolling volatility and identify high-volatility periods
5. **Analyze Correlation**: Compute correlation matrices between indicators
6. **Find Extreme Values**: Identify highest and lowest values for any indicator

### Visualization Tools
7. **Create Time Series Plot**: Line charts showing trends over time
8. **Create Correlation Heatmap**: Visual correlation matrices
9. **Create Volatility Plot**: Dual-axis plots showing value and volatility
10. **Create Distribution Plot**: Histograms showing value distributions

## 💡 Example Questions

Try asking the agent questions like:

- "What was the volatility of the S&P 500 during the 2008 financial crisis?"
- "Show me the correlation between VIX and S&P 500 from 2020 to 2022"
- "Plot the Federal Funds Rate and unemployment rate over time"
- "What were the highest values of oil prices in 2022?"
- "Analyze the relationship between Bitcoin and gold"
- "Find periods of extreme volatility in the stock market"
- "What happened to inflation during COVID-19?"
- "Show me the distribution of unemployment rates since 2008"

## 🎨 Visualization Management

The application intelligently manages visualizations across multi-turn conversations:

- Each visualization is saved as a JSON file with a unique ID
- Visualizations are displayed inline with agent responses
- Old visualizations are automatically cleaned up between conversation turns
- Only visualizations from the current conversation turn are retained

## 🤖 Agent Configuration

The financial analyst agent is configured with:

- **Model**: Google Gemini 2.0 Flash (via CrewAI's LLM integration)
- **Temperature**: 0.1 (for consistent, factual responses)
- **Max Iterations**: 15 (allows complex multi-step analyses)
- **Tools**: All 10 custom tools for comprehensive analysis

## 📝 Notes on Code Execution

Currently, the agent uses specialized tools for data analysis. While CrewAI supports code execution tools, the current implementation prioritizes:

1. **Reliability**: Pre-built tools provide consistent, tested functionality
2. **Safety**: No arbitrary code execution reduces security risks
3. **Performance**: Optimized tools are faster than dynamic code generation
4. **Maintainability**: Clear tool definitions are easier to debug and extend

**Future Enhancement**: A sandboxed code execution tool could be added as a fallback for edge cases not covered by existing tools.

## 🔒 Security

- API keys are stored in `.env` files (never committed to git)
- `.gitignore` includes `.env` and other sensitive files
- Temporary visualization files are cleaned up automatically

## 🤝 Contributing

To extend the project:

1. **Add new tools**: Create new tool classes in the `tools/` directory
2. **Enhance visualizations**: Add new visualization types in `visualization_tools.py`
3. **Improve agent prompts**: Modify agent configuration in `agents/financial_analyst.py`
4. **Add new data sources**: Update `utils/data_loader.py` with new datasets

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **CrewAI**: For the agent orchestration framework
- **Google Gemini**: For the Gemini 2.0 Flash language model
- **Plotly**: For beautiful, interactive visualizations
- **Streamlit**: For the rapid web application framework

