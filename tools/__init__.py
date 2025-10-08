"""Custom tools for financial data analysis."""

from .data_query_tools import (
    DateRangeQueryTool,
    IndicatorStatsTool,
    AvailableIndicatorsTool
)
from .analysis_tools import (
    VolatilityAnalysisTool,
    CorrelationAnalysisTool,
    FindExtremeValuesTool
)
from .visualization_tools import (
    TimeSeriesPlotTool,
    CorrelationHeatmapTool,
    VolatilityPlotTool,
    DistributionPlotTool
)
from .advanced_analysis_tools import (
    ReturnsAnalysisTool,
    DrawdownAnalysisTool,
    MovingAverageTool,
    PercentageChangeTool,
    YearOverYearTool
)
from .advanced_visualization_tools import (
    ScatterPlotTool,
    ComparativePerformanceTool,
    MovingAveragePlotTool,
    DrawdownChartTool,
    MultiIndicatorPlotTool
)
from .news_tools import (
    HeadlinesFetcherTool,
    VolatilityNewsCorrelationTool,
    EventTimelineTool
)
from .volatility_explanation_tools import (
    ComprehensiveVolatilityExplanationTool,
    IdentifyCorrelatedMovementsTool
)

__all__ = [
    # Data query tools
    'DateRangeQueryTool',
    'IndicatorStatsTool',
    'AvailableIndicatorsTool',
    # Analysis tools
    'VolatilityAnalysisTool',
    'CorrelationAnalysisTool',
    'FindExtremeValuesTool',
    # Visualization tools
    'TimeSeriesPlotTool',
    'CorrelationHeatmapTool',
    'VolatilityPlotTool',
    'DistributionPlotTool',
    # Advanced analysis tools
    'ReturnsAnalysisTool',
    'DrawdownAnalysisTool',
    'MovingAverageTool',
    'PercentageChangeTool',
    'YearOverYearTool',
    # Advanced visualization tools
    'ScatterPlotTool',
    'ComparativePerformanceTool',
    'MovingAveragePlotTool',
    'DrawdownChartTool',
    'MultiIndicatorPlotTool',
    # News and event tools
    'HeadlinesFetcherTool',
    'VolatilityNewsCorrelationTool',
    'EventTimelineTool',
    # Volatility explanation tools
    'ComprehensiveVolatilityExplanationTool',
    'IdentifyCorrelatedMovementsTool',
]

