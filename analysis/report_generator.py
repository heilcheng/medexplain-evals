"""Report generation for MEQ-Bench evaluation results.

This module generates comprehensive HTML and Markdown reports
from evaluation results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "MEQ-Bench 2.0 Evaluation Report"
    include_figures: bool = True
    include_error_analysis: bool = True
    include_statistical_tests: bool = True
    include_raw_data: bool = False
    max_error_cases: int = 20


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --secondary: #7c3aed;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-600: #4b5563;
            --gray-900: #111827;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--gray-900);
            background: var(--gray-50);
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            padding: 2rem;
        }}
        
        h1 {{
            color: var(--primary);
            margin-bottom: 0.5rem;
            font-size: 2rem;
        }}
        
        h2 {{
            color: var(--gray-900);
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--gray-200);
        }}
        
        h3 {{
            color: var(--gray-600);
            margin: 1.5rem 0 0.75rem;
        }}
        
        .metadata {{
            color: var(--gray-600);
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .stat-card {{
            background: var(--gray-100);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}
        
        .stat-label {{
            font-size: 0.85rem;
            color: var(--gray-600);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }}
        
        th {{
            background: var(--gray-100);
            font-weight: 600;
        }}
        
        tr:hover {{
            background: var(--gray-50);
        }}
        
        .rank-1 {{ color: var(--success); font-weight: bold; }}
        .rank-2 {{ color: var(--primary); }}
        .rank-3 {{ color: var(--secondary); }}
        
        .figure {{
            margin: 1.5rem 0;
            text-align: center;
        }}
        
        .figure img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .figure-caption {{
            font-size: 0.85rem;
            color: var(--gray-600);
            margin-top: 0.5rem;
        }}
        
        .error-card {{
            background: #fef2f2;
            border-left: 4px solid var(--danger);
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
        }}
        
        .warning-card {{
            background: #fffbeb;
            border-left: 4px solid var(--warning);
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
        }}
        
        .recommendation {{
            background: #ecfdf5;
            border-left: 4px solid var(--success);
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .badge-success {{ background: #d1fae5; color: #065f46; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
        .badge-danger {{ background: #fee2e2; color: #991b1b; }}
        
        footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--gray-200);
            color: var(--gray-600);
            font-size: 0.85rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
        <footer>
            Generated by MEQ-Bench 2.0 | {timestamp}
        </footer>
    </div>
</body>
</html>
"""


class ReportGenerator:
    """Generate HTML and Markdown reports from evaluation results."""
    
    def __init__(
        self,
        output_dir: str = "reports",
        config: Optional[ReportConfig] = None,
    ):
        """Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
            config: Report configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ReportConfig()
    
    def generate_html_report(
        self,
        analysis_results: Dict[str, Any],
        error_report: Optional[Dict[str, Any]] = None,
        statistical_results: Optional[Dict[str, Any]] = None,
        figures_dir: Optional[str] = None,
        filename: str = "summary_report.html",
    ) -> Path:
        """Generate comprehensive HTML report.
        
        Args:
            analysis_results: Results from ScoreAnalyzer
            error_report: Optional error analysis report
            statistical_results: Optional statistical test results
            figures_dir: Directory containing figures
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        content_parts = []
        
        # Header
        content_parts.append(f"<h1>{self.config.title}</h1>")
        content_parts.append(f'<p class="metadata">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        
        # Summary statistics
        content_parts.append(self._generate_summary_section(analysis_results))
        
        # Rankings table
        content_parts.append(self._generate_rankings_section(analysis_results))
        
        # Dimension breakdown
        content_parts.append(self._generate_dimension_section(analysis_results))
        
        # Audience breakdown
        content_parts.append(self._generate_audience_section(analysis_results))
        
        # Figures
        if self.config.include_figures and figures_dir:
            content_parts.append(self._generate_figures_section(figures_dir))
        
        # Error analysis
        if self.config.include_error_analysis and error_report:
            content_parts.append(self._generate_error_section(error_report))
        
        # Statistical tests
        if self.config.include_statistical_tests and statistical_results:
            content_parts.append(self._generate_stats_section(statistical_results))
        
        # Combine content
        content = "\n".join(content_parts)
        html = HTML_TEMPLATE.format(
            title=self.config.title,
            content=content,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        logger.info(f"Generated HTML report: {output_path}")
        return output_path
    
    def _generate_summary_section(self, results: Dict[str, Any]) -> str:
        """Generate summary statistics section."""
        rankings = results.get("rankings", [])
        
        if not rankings:
            return "<h2>Summary</h2><p>No results available.</p>"
        
        best_model = rankings[0] if rankings else {}
        avg_score = sum(r.get("overall_mean", 0) for r in rankings) / len(rankings) if rankings else 0
        
        return f"""
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-value">{len(rankings)}</div>
                <div class="stat-label">Models Evaluated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{best_model.get('model', 'N/A')}</div>
                <div class="stat-label">Top Model</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{best_model.get('overall_mean', 0):.2f}</div>
                <div class="stat-label">Top Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_score:.2f}</div>
                <div class="stat-label">Average Score</div>
            </div>
        </div>
        """
    
    def _generate_rankings_section(self, results: Dict[str, Any]) -> str:
        """Generate rankings table section."""
        rankings = results.get("rankings", [])
        
        rows = []
        for r in rankings:
            rank_class = f"rank-{r['rank']}" if r['rank'] <= 3 else ""
            safety_badge = "badge-success" if r.get('safety_pass_rate', 1) > 0.9 else "badge-warning"
            
            rows.append(f"""
                <tr>
                    <td class="{rank_class}">{r['rank']}</td>
                    <td class="{rank_class}">{r['model']}</td>
                    <td>{r['overall_mean']:.3f} ± {r.get('overall_std', 0):.3f}</td>
                    <td><span class="badge {safety_badge}">{r.get('safety_pass_rate', 1)*100:.0f}%</span></td>
                    <td>{r.get('item_count', 'N/A')}</td>
                </tr>
            """)
        
        return f"""
        <h2>Model Rankings</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Overall Score</th>
                    <th>Safety Rate</th>
                    <th>Items</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
    
    def _generate_dimension_section(self, results: Dict[str, Any]) -> str:
        """Generate dimension breakdown section."""
        dimension_comparison = results.get("dimension_comparison", {})
        
        if not dimension_comparison:
            return ""
        
        models = list(dimension_comparison.keys())
        dimensions = list(next(iter(dimension_comparison.values())).keys()) if models else []
        
        header = "<th>Model</th>" + "".join(f"<th>{d.replace('_', ' ').title()}</th>" for d in dimensions)
        
        rows = []
        for model in models:
            scores = dimension_comparison[model]
            cells = "".join(f"<td>{scores.get(d, 0):.2f}</td>" for d in dimensions)
            rows.append(f"<tr><td>{model}</td>{cells}</tr>")
        
        return f"""
        <h2>Performance by Dimension</h2>
        <table>
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        """
    
    def _generate_audience_section(self, results: Dict[str, Any]) -> str:
        """Generate audience breakdown section."""
        audience_comparison = results.get("audience_comparison", {})
        
        if not audience_comparison:
            return ""
        
        models = list(audience_comparison.keys())
        audiences = ["physician", "nurse", "patient", "caregiver"]
        
        header = "<th>Model</th>" + "".join(f"<th>{a.title()}</th>" for a in audiences)
        
        rows = []
        for model in models:
            scores = audience_comparison[model]
            cells = "".join(f"<td>{scores.get(a, 0):.2f}</td>" for a in audiences)
            rows.append(f"<tr><td>{model}</td>{cells}</tr>")
        
        return f"""
        <h2>Performance by Audience</h2>
        <table>
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        """
    
    def _generate_figures_section(self, figures_dir: str) -> str:
        """Generate figures section."""
        figures_path = Path(figures_dir)
        if not figures_path.exists():
            return ""
        
        figures = []
        for img_path in sorted(figures_path.glob("*.png")):
            rel_path = img_path.name
            caption = img_path.stem.replace("_", " ").title()
            figures.append(f"""
                <div class="figure">
                    <img src="figures/{rel_path}" alt="{caption}">
                    <p class="figure-caption">{caption}</p>
                </div>
            """)
        
        if not figures:
            return ""
        
        return f"""
        <h2>Visualizations</h2>
        {''.join(figures)}
        """
    
    def _generate_error_section(self, error_report: Dict[str, Any]) -> str:
        """Generate error analysis section."""
        total = error_report.get("total_errors", 0)
        by_type = error_report.get("errors_by_type", {})
        recommendations = error_report.get("recommendations", [])
        
        type_items = "".join(f"<li>{t}: {c}</li>" for t, c in by_type.items())
        rec_items = "".join(f'<div class="recommendation">{r}</div>' for r in recommendations)
        
        return f"""
        <h2>Error Analysis</h2>
        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Errors</div>
            </div>
        </div>
        
        <h3>Errors by Type</h3>
        <ul>{type_items}</ul>
        
        <h3>Recommendations</h3>
        {rec_items}
        """
    
    def _generate_stats_section(self, stats_results: Dict[str, Any]) -> str:
        """Generate statistical tests section."""
        tests = stats_results.get("tests", [])
        
        rows = []
        for test in tests:
            sig = "Yes" if test.get("significant", False) else "No"
            badge_class = "badge-success" if test.get("significant") else "badge-warning"
            
            rows.append(f"""
                <tr>
                    <td>{test.get('comparison', 'N/A')}</td>
                    <td>{test.get('test_type', 'N/A')}</td>
                    <td>{test.get('statistic', 0):.3f}</td>
                    <td>{test.get('p_value', 1):.4f}</td>
                    <td><span class="badge {badge_class}">{sig}</span></td>
                </tr>
            """)
        
        return f"""
        <h2>Statistical Tests</h2>
        <table>
            <thead>
                <tr>
                    <th>Comparison</th>
                    <th>Test</th>
                    <th>Statistic</th>
                    <th>p-value</th>
                    <th>Significant</th>
                </tr>
            </thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        """
    
    def generate_markdown_report(
        self,
        analysis_results: Dict[str, Any],
        filename: str = "summary_report.md",
    ) -> Path:
        """Generate Markdown report for GitHub.
        
        Args:
            analysis_results: Results from ScoreAnalyzer
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        lines = []
        
        # Header
        lines.append(f"# {self.config.title}")
        lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Rankings
        lines.append("## Model Rankings\n")
        lines.append("| Rank | Model | Overall Score | Safety Rate |")
        lines.append("|------|-------|---------------|-------------|")
        
        for r in analysis_results.get("rankings", []):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r["rank"], "")
            lines.append(f"| {r['rank']} {medal} | {r['model']} | {r['overall_mean']:.3f} | {r.get('safety_pass_rate', 1)*100:.0f}% |")
        
        lines.append("")
        
        # Dimension comparison
        dimension_comparison = analysis_results.get("dimension_comparison", {})
        if dimension_comparison:
            lines.append("## Performance by Dimension\n")
            
            models = list(dimension_comparison.keys())
            dims = list(next(iter(dimension_comparison.values())).keys()) if models else []
            
            header = "| Model | " + " | ".join(d.replace("_", " ").title() for d in dims) + " |"
            separator = "|-------|" + "|".join("------" for _ in dims) + "|"
            
            lines.append(header)
            lines.append(separator)
            
            for model in models:
                scores = dimension_comparison[model]
                row = f"| {model} | " + " | ".join(f"{scores.get(d, 0):.2f}" for d in dims) + " |"
                lines.append(row)
            
            lines.append("")
        
        # Footer
        lines.append("---\n")
        lines.append("*MEQ-Bench 2.0 - Benchmark for evaluating audience-adaptive explanation quality in medical LLMs*")
        
        content = "\n".join(lines)
        
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Generated Markdown report: {output_path}")
        return output_path
    
    def generate_all_reports(
        self,
        analysis_results: Dict[str, Any],
        error_report: Optional[Dict[str, Any]] = None,
        statistical_results: Optional[Dict[str, Any]] = None,
        figures_dir: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Generate all report formats.
        
        Args:
            analysis_results: Results from ScoreAnalyzer
            error_report: Optional error analysis report
            statistical_results: Optional statistical test results
            figures_dir: Directory containing figures
            
        Returns:
            Dict of format -> path
        """
        reports = {}
        
        reports["html"] = self.generate_html_report(
            analysis_results=analysis_results,
            error_report=error_report,
            statistical_results=statistical_results,
            figures_dir=figures_dir,
        )
        
        reports["markdown"] = self.generate_markdown_report(analysis_results)
        
        # Save raw JSON
        json_path = self.output_dir / "analysis_results.json"
        with open(json_path, "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)
        reports["json"] = json_path
        
        logger.info(f"Generated {len(reports)} reports")
        return reports

