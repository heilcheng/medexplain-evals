"""MEQ-Bench Analysis Module.

This module provides analysis and visualization tools for
evaluating benchmark results.

Components:
    - analyzer: Core analysis functions
    - visualizations: Chart generation (matplotlib/plotly)
    - error_analysis: Failure case identification
    - report_generator: HTML/Markdown report generation
    - statistical_tests: Significance testing utilities
"""

from .analyzer import ScoreAnalyzer, AnalysisResults
from .visualizations import MEQBenchVisualizer
from .error_analysis import ErrorAnalyzer
from .report_generator import ReportGenerator
from .statistical_tests import StatisticalTests

__all__ = [
    "ScoreAnalyzer",
    "AnalysisResults",
    "MEQBenchVisualizer",
    "ErrorAnalyzer",
    "ReportGenerator",
    "StatisticalTests",
]

