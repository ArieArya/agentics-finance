"""
Quick test script to verify the setup is working correctly.
Run this before starting the Streamlit app.
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import pandas
        print("✓ pandas")
    except ImportError:
        print("✗ pandas not installed")
        return False

    try:
        import plotly
        print("✓ plotly")
    except ImportError:
        print("✗ plotly not installed")
        return False

    try:
        import streamlit
        print("✓ streamlit")
    except ImportError:
        print("✗ streamlit not installed")
        return False

    try:
        import crewai
        print("✓ crewai")
    except ImportError:
        print("✗ crewai not installed")
        return False

    try:
        import openai
        print("✓ openai")
    except ImportError:
        print("✗ openai not installed")
        return False

    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv")
    except ImportError:
        print("✗ python-dotenv not installed")
        return False

    return True


def test_data_files():
    """Test that data files exist and can be loaded."""
    print("\nTesting data files...")

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    macro_file = os.path.join(data_dir, "macro_factors_new.csv")
    market_file = os.path.join(data_dir, "market_factors_new.csv")

    if not os.path.exists(macro_file):
        print(f"✗ Macro data file not found: {macro_file}")
        return False
    print(f"✓ Macro data file found")

    if not os.path.exists(market_file):
        print(f"✗ Market data file not found: {market_file}")
        return False
    print(f"✓ Market data file found")

    # Try loading the data
    try:
        from utils.data_loader import load_macro_factors, load_market_factors, get_data_summary

        macro_df = load_macro_factors()
        print(f"✓ Macro data loaded: {len(macro_df)} rows, {len(macro_df.columns)} columns")

        market_df = load_market_factors()
        print(f"✓ Market data loaded: {len(market_df)} rows, {len(market_df.columns)} columns")

        summary = get_data_summary()
        print(f"✓ Data summary retrieved")

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

    return True


def test_environment():
    """Test that environment variables are set."""
    print("\nTesting environment variables...")

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("✗ OPENAI_API_KEY not set in .env file")
        print("  Please create a .env file with your OpenAI API key")
        print("  See env.example for template")
        return False

    if api_key == "your_openai_api_key_here":
        print("✗ OPENAI_API_KEY is still set to default value")
        print("  Please update .env with your actual API key")
        return False

    print("✓ OPENAI_API_KEY is set")

    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    print(f"✓ Model: {model}")

    return True


def test_tools():
    """Test that tools can be imported."""
    print("\nTesting tools...")

    try:
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
        print("✓ All tools imported successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing tools: {e}")
        return False


def test_agents():
    """Test that agents can be imported."""
    print("\nTesting agents...")

    try:
        from agents import create_financial_analyst_agent
        print("✓ Agent module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing agents: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Financial Data Analyst - Setup Test")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Data Files", test_data_files()))
    results.append(("Environment", test_environment()))
    results.append(("Tools", test_tools()))
    results.append(("Agents", test_agents()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! You're ready to run the application.")
        print("\nRun the following command to start:")
        print("  streamlit run app.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Create .env file with your OpenAI API key")
        print("  - Ensure data files are in the data/ directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())

