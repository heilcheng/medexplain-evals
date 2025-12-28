"""Visualization tools for MEQ-Bench analysis.

This module generates charts and graphs for evaluating model performance
including radar charts, heatmaps, and comparative bar charts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, visualizations will be limited")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("plotly not available, interactive visualizations disabled")


# Color scheme for consistency
COLORS = {
    "primary": "#2563eb",
    "secondary": "#7c3aed",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "gray": "#6b7280",
    
    # Model colors
    "gpt-5.1": "#10a37f",
    "gpt-4o": "#1a7f64",
    "claude-opus-4.5": "#d97706",
    "claude-sonnet-4.5": "#ea580c",
    "gemini-3-pro": "#4285f4",
    "deepseek-v3": "#6366f1",
    "qwen3-max": "#8b5cf6",
    "llama-4-scout": "#0ea5e9",
}

# Audience colors
AUDIENCE_COLORS = {
    "physician": "#1e40af",
    "nurse": "#047857",
    "patient": "#b45309",
    "caregiver": "#7c2d12",
}


class MEQBenchVisualizer:
    """Generate visualizations for MEQ-Bench results."""
    
    DIMENSIONS = [
        "factual_accuracy",
        "terminological_appropriateness",
        "explanatory_completeness",
        "actionability",
        "safety",
        "empathy_tone",
    ]
    
    DIMENSION_LABELS = {
        "factual_accuracy": "Factual\nAccuracy",
        "terminological_appropriateness": "Terminology",
        "explanatory_completeness": "Completeness",
        "actionability": "Actionability",
        "safety": "Safety",
        "empathy_tone": "Empathy",
    }
    
    def __init__(self, output_dir: str = "reports/figures"):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_overall_scores(
        self,
        rankings: List[Dict[str, Any]],
        title: str = "Model Overall Scores",
        filename: str = "overall_scores.png",
    ) -> Optional[Path]:
        """Create bar chart of overall model scores.
        
        Args:
            rankings: List of model rankings with scores
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib required for this visualization")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = [r["model"] for r in rankings]
        scores = [r["overall_mean"] for r in rankings]
        stds = [r.get("overall_std", 0) for r in rankings]
        
        colors = [COLORS.get(m, COLORS["gray"]) for m in models]
        
        bars = ax.bar(models, scores, color=colors, yerr=stds, capsize=3)
        
        ax.set_ylabel("Overall Score", fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 5.5)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved overall scores chart to {output_path}")
        return output_path
    
    def plot_dimension_radar(
        self,
        dimension_scores: Dict[str, Dict[str, float]],
        title: str = "Model Performance by Dimension",
        filename: str = "dimension_radar.png",
    ) -> Optional[Path]:
        """Create radar chart comparing models across dimensions.
        
        Args:
            dimension_scores: Dict of model -> dimension -> score
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Prepare data
        models = list(dimension_scores.keys())
        dimensions = self.DIMENSIONS
        
        # Number of dimensions
        num_dims = len(dimensions)
        angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for model in models:
            values = [dimension_scores[model].get(d, 0) for d in dimensions]
            values += values[:1]  # Complete the loop
            
            color = COLORS.get(model, COLORS["gray"])
            ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.DIMENSION_LABELS.get(d, d) for d in dimensions])
        ax.set_ylim(0, 5)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved radar chart to {output_path}")
        return output_path
    
    def plot_audience_heatmap(
        self,
        audience_scores: Dict[str, Dict[str, float]],
        title: str = "Model x Audience Performance",
        filename: str = "audience_heatmap.png",
    ) -> Optional[Path]:
        """Create heatmap of model performance by audience.
        
        Args:
            audience_scores: Dict of model -> audience -> score
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        models = list(audience_scores.keys())
        audiences = ["physician", "nurse", "patient", "caregiver"]
        
        # Build matrix
        matrix = []
        for model in models:
            row = [audience_scores[model].get(a, 0) for a in audiences]
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=1, vmax=5)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
        
        # Labels
        ax.set_xticks(np.arange(len(audiences)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([a.capitalize() for a in audiences])
        ax.set_yticklabels(models)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(audiences)):
                text = ax.text(
                    j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", color="black"
                )
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved audience heatmap to {output_path}")
        return output_path
    
    def plot_specialty_breakdown(
        self,
        specialty_scores: Dict[str, Dict[str, float]],
        title: str = "Performance by Medical Specialty",
        filename: str = "specialty_breakdown.png",
    ) -> Optional[Path]:
        """Create grouped bar chart of performance by specialty.
        
        Args:
            specialty_scores: Dict of model -> specialty -> score
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        models = list(specialty_scores.keys())
        
        # Get all specialties
        all_specialties = set()
        for model_scores in specialty_scores.values():
            all_specialties.update(model_scores.keys())
        specialties = sorted(all_specialties)
        
        x = np.arange(len(specialties))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, model in enumerate(models):
            scores = [specialty_scores[model].get(s, 0) for s in specialties]
            offset = (i - len(models) / 2 + 0.5) * width
            color = COLORS.get(model, COLORS["gray"])
            ax.bar(x + offset, scores, width, label=model, color=color)
        
        ax.set_ylabel("Score")
        ax.set_xlabel("Medical Specialty")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in specialties])
        ax.legend()
        ax.set_ylim(0, 5.5)
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved specialty breakdown to {output_path}")
        return output_path
    
    def plot_score_distribution(
        self,
        scores: List[float],
        model: str,
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """Create histogram of score distribution.
        
        Args:
            scores: List of scores
            model: Model name
            title: Chart title
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if title is None:
            title = f"Score Distribution: {model}"
        if filename is None:
            filename = f"distribution_{model.replace('-', '_')}.png"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        color = COLORS.get(model, COLORS["primary"])
        ax.hist(scores, bins=20, color=color, alpha=0.7, edgecolor="black")
        
        # Add mean line
        mean_score = np.mean(scores)
        ax.axvline(mean_score, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_score:.2f}")
        
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return output_path
    
    def create_interactive_dashboard(
        self,
        analysis_results: Dict[str, Any],
        filename: str = "dashboard.html",
    ) -> Optional[Path]:
        """Create interactive Plotly dashboard.
        
        Args:
            analysis_results: Full analysis results
            filename: Output filename
            
        Returns:
            Path to saved HTML or None
        """
        if not HAS_PLOTLY:
            logger.warning("plotly required for interactive dashboard")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "bar"}, {"type": "polar"}],
                [{"type": "heatmap"}, {"type": "bar"}]
            ],
            subplot_titles=(
                "Overall Scores",
                "Dimension Radar",
                "Audience Performance",
                "Safety Pass Rate"
            )
        )
        
        rankings = analysis_results.get("rankings", [])
        dimension_scores = analysis_results.get("dimension_comparison", {})
        audience_scores = analysis_results.get("audience_comparison", {})
        
        # 1. Overall scores bar chart
        models = [r["model"] for r in rankings]
        scores = [r["overall_mean"] for r in rankings]
        
        fig.add_trace(
            go.Bar(x=models, y=scores, marker_color=[COLORS.get(m, COLORS["gray"]) for m in models]),
            row=1, col=1
        )
        
        # 2. Radar chart (first model as example)
        if dimension_scores:
            first_model = list(dimension_scores.keys())[0]
            dims = list(dimension_scores[first_model].keys())
            values = list(dimension_scores[first_model].values())
            
            fig.add_trace(
                go.Scatterpolar(r=values + [values[0]], theta=dims + [dims[0]], fill="toself", name=first_model),
                row=1, col=2
            )
        
        # 3. Heatmap
        if audience_scores:
            z = [[audience_scores[m].get(a, 0) for a in ["physician", "nurse", "patient", "caregiver"]] for m in models]
            
            fig.add_trace(
                go.Heatmap(z=z, x=["Physician", "Nurse", "Patient", "Caregiver"], y=models, colorscale="RdYlGn"),
                row=2, col=1
            )
        
        # 4. Safety pass rates
        safety_rates = [r.get("safety_pass_rate", 1.0) * 100 for r in rankings]
        fig.add_trace(
            go.Bar(x=models, y=safety_rates, marker_color=COLORS["success"]),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="MEQ-Bench 2.0 Evaluation Dashboard")
        
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        
        logger.info(f"Saved interactive dashboard to {output_path}")
        return output_path
    
    def generate_all_figures(
        self,
        analysis_results: Dict[str, Any],
    ) -> List[Path]:
        """Generate all standard figures.
        
        Args:
            analysis_results: Full analysis results dict
            
        Returns:
            List of generated figure paths
        """
        figures = []
        
        # Overall scores
        if analysis_results.get("rankings"):
            path = self.plot_overall_scores(analysis_results["rankings"])
            if path:
                figures.append(path)
        
        # Dimension radar
        if analysis_results.get("dimension_comparison"):
            path = self.plot_dimension_radar(analysis_results["dimension_comparison"])
            if path:
                figures.append(path)
        
        # Audience heatmap
        if analysis_results.get("audience_comparison"):
            path = self.plot_audience_heatmap(analysis_results["audience_comparison"])
            if path:
                figures.append(path)
        
        # Specialty breakdown
        if analysis_results.get("specialty_breakdown"):
            path = self.plot_specialty_breakdown(analysis_results["specialty_breakdown"])
            if path:
                figures.append(path)
        
        # Interactive dashboard
        path = self.create_interactive_dashboard(analysis_results)
        if path:
            figures.append(path)
        
        logger.info(f"Generated {len(figures)} figures")
        return figures

