"""Core analysis functions for MedExplain-Evals results.

This module provides comprehensive analysis of evaluation results including
score aggregation, comparative analysis, and performance breakdowns.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DimensionStats:
    """Statistics for one evaluation dimension."""
    name: str
    mean: float
    std: float
    median: float
    min: float
    max: float
    q25: float
    q75: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelStats:
    """Statistics for one model's performance."""
    model: str
    overall_mean: float
    overall_std: float
    dimensions: Dict[str, DimensionStats]
    audiences: Dict[str, float]
    safety_pass_rate: float
    item_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["dimensions"] = {k: v.to_dict() for k, v in self.dimensions.items()}
        return result


@dataclass
class AnalysisResults:
    """Complete analysis results."""
    models: List[str]
    model_stats: Dict[str, ModelStats]
    rankings: List[Dict[str, Any]]
    audience_comparison: Dict[str, Dict[str, float]]
    dimension_comparison: Dict[str, Dict[str, float]]
    specialty_breakdown: Optional[Dict[str, Dict[str, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": self.models,
            "model_stats": {k: v.to_dict() for k, v in self.model_stats.items()},
            "rankings": self.rankings,
            "audience_comparison": self.audience_comparison,
            "dimension_comparison": self.dimension_comparison,
            "specialty_breakdown": self.specialty_breakdown,
        }


class ScoreAnalyzer:
    """Analyze MedExplain-Evals evaluation scores."""
    
    DIMENSIONS = [
        "factual_accuracy",
        "terminological_appropriateness",
        "explanatory_completeness",
        "actionability",
        "safety",
        "empathy_tone",
    ]
    
    AUDIENCES = ["physician", "nurse", "patient", "caregiver"]
    
    def __init__(self, results_dir: str):
        """Initialize analyzer.
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        self.scores_df: Optional[pd.DataFrame] = None
        self.model_summaries: Dict[str, Dict] = {}
    
    def load_scores(self, models: Optional[List[str]] = None) -> pd.DataFrame:
        """Load scores from all models into a DataFrame.
        
        Args:
            models: Specific models to load (None = all)
            
        Returns:
            DataFrame with all scores
        """
        all_scores = []
        
        # Find model directories
        if models:
            model_dirs = [self.results_dir / m for m in models]
        else:
            model_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            if not model_dir.is_dir():
                continue
            
            scores_dir = model_dir / "scores"
            if not scores_dir.exists():
                continue
            
            model_name = model_dir.name
            
            # Load CSV if available
            csv_path = scores_dir / "all_scores.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["model"] = model_name
                all_scores.append(df)
                continue
            
            # Otherwise load individual JSON files
            for score_file in scores_dir.glob("*.json"):
                if score_file.name in ("aggregated_scores.json",):
                    continue
                
                with open(score_file, "r") as f:
                    score_data = json.load(f)
                
                score_data["model"] = model_name
                all_scores.append(score_data)
            
            # Load aggregated summary
            agg_path = scores_dir / "aggregated_scores.json"
            if agg_path.exists():
                with open(agg_path, "r") as f:
                    self.model_summaries[model_name] = json.load(f)
        
        if all_scores:
            if isinstance(all_scores[0], dict):
                self.scores_df = pd.DataFrame(all_scores)
            else:
                self.scores_df = pd.concat(all_scores, ignore_index=True)
        else:
            self.scores_df = pd.DataFrame()
        
        logger.info(f"Loaded {len(self.scores_df)} score records from {len(model_dirs)} models")
        return self.scores_df
    
    def compute_model_stats(self, model: str) -> ModelStats:
        """Compute statistics for a single model.
        
        Args:
            model: Model name
            
        Returns:
            ModelStats object
        """
        if self.scores_df is None or self.scores_df.empty:
            raise ValueError("No scores loaded")
        
        model_df = self.scores_df[self.scores_df["model"] == model]
        
        if model_df.empty:
            raise ValueError(f"No scores for model: {model}")
        
        # Overall stats
        overall_mean = model_df["overall"].mean()
        overall_std = model_df["overall"].std()
        
        # Dimension stats
        dimensions = {}
        for dim in self.DIMENSIONS:
            if dim not in model_df.columns:
                continue
            
            values = model_df[dim].dropna()
            if len(values) == 0:
                continue
            
            dimensions[dim] = DimensionStats(
                name=dim,
                mean=values.mean(),
                std=values.std(),
                median=values.median(),
                min=values.min(),
                max=values.max(),
                q25=values.quantile(0.25),
                q75=values.quantile(0.75),
            )
        
        # Audience means
        audiences = {}
        for audience in self.AUDIENCES:
            audience_df = model_df[model_df["audience"] == audience]
            if not audience_df.empty:
                audiences[audience] = audience_df["overall"].mean()
        
        # Safety pass rate
        if "safety_passed" in model_df.columns:
            safety_pass_rate = model_df["safety_passed"].mean()
        else:
            safety_pass_rate = 1.0
        
        return ModelStats(
            model=model,
            overall_mean=overall_mean,
            overall_std=overall_std,
            dimensions=dimensions,
            audiences=audiences,
            safety_pass_rate=safety_pass_rate,
            item_count=len(model_df),
        )
    
    def analyze(self, models: Optional[List[str]] = None) -> AnalysisResults:
        """Run complete analysis.
        
        Args:
            models: Specific models to analyze (None = all)
            
        Returns:
            AnalysisResults object
        """
        # Load scores if needed
        if self.scores_df is None:
            self.load_scores(models)
        
        if self.scores_df.empty:
            logger.warning("No scores to analyze")
            return AnalysisResults(
                models=[],
                model_stats={},
                rankings=[],
                audience_comparison={},
                dimension_comparison={},
            )
        
        # Get unique models
        available_models = self.scores_df["model"].unique().tolist()
        
        # Compute stats for each model
        model_stats = {}
        for model in available_models:
            try:
                model_stats[model] = self.compute_model_stats(model)
            except Exception as e:
                logger.error(f"Error computing stats for {model}: {e}")
        
        # Generate rankings
        rankings = self._generate_rankings(model_stats)
        
        # Audience comparison
        audience_comparison = self._compare_by_audience(available_models)
        
        # Dimension comparison
        dimension_comparison = self._compare_by_dimension(available_models)
        
        # Specialty breakdown (if available)
        specialty_breakdown = self._specialty_breakdown(available_models)
        
        return AnalysisResults(
            models=available_models,
            model_stats=model_stats,
            rankings=rankings,
            audience_comparison=audience_comparison,
            dimension_comparison=dimension_comparison,
            specialty_breakdown=specialty_breakdown,
        )
    
    def _generate_rankings(
        self,
        model_stats: Dict[str, ModelStats],
    ) -> List[Dict[str, Any]]:
        """Generate model rankings."""
        rankings = []
        
        sorted_models = sorted(
            model_stats.values(),
            key=lambda x: x.overall_mean,
            reverse=True,
        )
        
        for rank, stats in enumerate(sorted_models, 1):
            rankings.append({
                "rank": rank,
                "model": stats.model,
                "overall_mean": round(stats.overall_mean, 3),
                "overall_std": round(stats.overall_std, 3),
                "safety_pass_rate": round(stats.safety_pass_rate, 3),
                "item_count": stats.item_count,
            })
        
        return rankings
    
    def _compare_by_audience(
        self,
        models: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compare model performance by audience."""
        comparison = {}
        
        for model in models:
            model_df = self.scores_df[self.scores_df["model"] == model]
            comparison[model] = {}
            
            for audience in self.AUDIENCES:
                audience_df = model_df[model_df["audience"] == audience]
                if not audience_df.empty:
                    comparison[model][audience] = round(audience_df["overall"].mean(), 3)
        
        return comparison
    
    def _compare_by_dimension(
        self,
        models: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compare model performance by dimension."""
        comparison = {}
        
        for model in models:
            model_df = self.scores_df[self.scores_df["model"] == model]
            comparison[model] = {}
            
            for dim in self.DIMENSIONS:
                if dim in model_df.columns:
                    comparison[model][dim] = round(model_df[dim].mean(), 3)
        
        return comparison
    
    def _specialty_breakdown(
        self,
        models: List[str],
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Break down performance by specialty if available."""
        if "specialty" not in self.scores_df.columns:
            return None
        
        breakdown = {}
        
        for model in models:
            model_df = self.scores_df[self.scores_df["model"] == model]
            breakdown[model] = {}
            
            for specialty in model_df["specialty"].unique():
                spec_df = model_df[model_df["specialty"] == specialty]
                breakdown[model][specialty] = round(spec_df["overall"].mean(), 3)
        
        return breakdown
    
    def get_score_distribution(
        self,
        model: str,
        column: str = "overall",
    ) -> Dict[str, Any]:
        """Get score distribution for a model.
        
        Args:
            model: Model name
            column: Column to analyze
            
        Returns:
            Distribution statistics
        """
        model_df = self.scores_df[self.scores_df["model"] == model]
        values = model_df[column].dropna()
        
        return {
            "count": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "q25": values.quantile(0.25),
            "median": values.median(),
            "q75": values.quantile(0.75),
            "max": values.max(),
            "histogram": np.histogram(values, bins=10),
        }
    
    def compare_models(
        self,
        model_a: str,
        model_b: str,
    ) -> Dict[str, Any]:
        """Direct comparison between two models.
        
        Args:
            model_a: First model
            model_b: Second model
            
        Returns:
            Comparison results
        """
        df_a = self.scores_df[self.scores_df["model"] == model_a]
        df_b = self.scores_df[self.scores_df["model"] == model_b]
        
        comparison = {
            "model_a": model_a,
            "model_b": model_b,
            "overall": {
                "a_mean": df_a["overall"].mean(),
                "b_mean": df_b["overall"].mean(),
                "difference": df_a["overall"].mean() - df_b["overall"].mean(),
            },
            "by_audience": {},
            "by_dimension": {},
        }
        
        # By audience
        for audience in self.AUDIENCES:
            a_scores = df_a[df_a["audience"] == audience]["overall"]
            b_scores = df_b[df_b["audience"] == audience]["overall"]
            
            if not a_scores.empty and not b_scores.empty:
                comparison["by_audience"][audience] = {
                    "a_mean": a_scores.mean(),
                    "b_mean": b_scores.mean(),
                    "difference": a_scores.mean() - b_scores.mean(),
                }
        
        # By dimension
        for dim in self.DIMENSIONS:
            if dim in df_a.columns and dim in df_b.columns:
                comparison["by_dimension"][dim] = {
                    "a_mean": df_a[dim].mean(),
                    "b_mean": df_b[dim].mean(),
                    "difference": df_a[dim].mean() - df_b[dim].mean(),
                }
        
        return comparison
    
    def export_to_csv(self, output_path: str) -> None:
        """Export all scores to CSV.
        
        Args:
            output_path: Output file path
        """
        if self.scores_df is not None:
            self.scores_df.to_csv(output_path, index=False)
            logger.info(f"Exported scores to {output_path}")

