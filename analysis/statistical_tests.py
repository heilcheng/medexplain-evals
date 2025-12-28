"""Statistical testing utilities for MEQ-Bench analysis.

This module provides statistical significance testing for
comparing model performance.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import scipy for statistical tests
try:
    from scipy import stats
    from scipy.stats import ttest_rel, ttest_ind, wilcoxon, mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available, statistical tests will be limited")


@dataclass
class TestResult:
    """Result from a statistical test."""
    comparison: str  # e.g., "model_a vs model_b"
    test_type: str  # e.g., "paired_t_test"
    statistic: float
    p_value: float
    significant: bool  # at alpha = 0.05
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    n_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result["confidence_interval"]:
            result["confidence_interval"] = list(result["confidence_interval"])
        return result


@dataclass
class BootstrapResult:
    """Result from bootstrap confidence interval estimation."""
    statistic: str  # e.g., "mean"
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float  # e.g., 0.95
    n_bootstrap: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatisticalTests:
    """Statistical testing utilities."""
    
    EFFECT_SIZE_THRESHOLDS = {
        "small": 0.2,
        "medium": 0.5,
        "large": 0.8,
    }
    
    def __init__(self, alpha: float = 0.05):
        """Initialize statistical tests.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
    
    def paired_t_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        model_a: str = "Model A",
        model_b: str = "Model B",
    ) -> TestResult:
        """Perform paired t-test between two models.
        
        Requires matched pairs (same items evaluated by both models).
        
        Args:
            scores_a: Scores from first model
            scores_b: Scores from second model
            model_a: First model name
            model_b: Second model name
            
        Returns:
            TestResult object
        """
        if not HAS_SCIPY:
            logger.warning("scipy required for t-test")
            return TestResult(
                comparison=f"{model_a} vs {model_b}",
                test_type="paired_t_test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
            )
        
        if len(scores_a) != len(scores_b):
            raise ValueError("Paired test requires equal-length score arrays")
        
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        # Perform test
        statistic, p_value = ttest_rel(scores_a, scores_b)
        
        # Effect size (Cohen's d for paired samples)
        diff = scores_a - scores_b
        effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        
        return TestResult(
            comparison=f"{model_a} vs {model_b}",
            test_type="paired_t_test",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            effect_size=float(effect_size),
            effect_size_interpretation=self._interpret_effect_size(abs(effect_size)),
            n_samples=len(scores_a),
        )
    
    def independent_t_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        model_a: str = "Model A",
        model_b: str = "Model B",
    ) -> TestResult:
        """Perform independent samples t-test.
        
        Args:
            scores_a: Scores from first model
            scores_b: Scores from second model
            model_a: First model name
            model_b: Second model name
            
        Returns:
            TestResult object
        """
        if not HAS_SCIPY:
            return TestResult(
                comparison=f"{model_a} vs {model_b}",
                test_type="independent_t_test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
            )
        
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        statistic, p_value = ttest_ind(scores_a, scores_b)
        
        # Cohen's d for independent samples
        pooled_std = np.sqrt(
            ((len(scores_a) - 1) * np.var(scores_a, ddof=1) + 
             (len(scores_b) - 1) * np.var(scores_b, ddof=1)) /
            (len(scores_a) + len(scores_b) - 2)
        )
        effect_size = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std if pooled_std > 0 else 0
        
        return TestResult(
            comparison=f"{model_a} vs {model_b}",
            test_type="independent_t_test",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            effect_size=float(effect_size),
            effect_size_interpretation=self._interpret_effect_size(abs(effect_size)),
            n_samples=len(scores_a) + len(scores_b),
        )
    
    def wilcoxon_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        model_a: str = "Model A",
        model_b: str = "Model B",
    ) -> TestResult:
        """Perform Wilcoxon signed-rank test (non-parametric paired test).
        
        Args:
            scores_a: Scores from first model
            scores_b: Scores from second model
            model_a: First model name
            model_b: Second model name
            
        Returns:
            TestResult object
        """
        if not HAS_SCIPY:
            return TestResult(
                comparison=f"{model_a} vs {model_b}",
                test_type="wilcoxon",
                statistic=0.0,
                p_value=1.0,
                significant=False,
            )
        
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, p_value = wilcoxon(scores_a, scores_b)
        except ValueError:
            # All differences are zero
            return TestResult(
                comparison=f"{model_a} vs {model_b}",
                test_type="wilcoxon",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                n_samples=len(scores_a),
            )
        
        return TestResult(
            comparison=f"{model_a} vs {model_b}",
            test_type="wilcoxon",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            n_samples=len(scores_a),
        )
    
    def mann_whitney_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        model_a: str = "Model A",
        model_b: str = "Model B",
    ) -> TestResult:
        """Perform Mann-Whitney U test (non-parametric independent test).
        
        Args:
            scores_a: Scores from first model
            scores_b: Scores from second model
            model_a: First model name
            model_b: Second model name
            
        Returns:
            TestResult object
        """
        if not HAS_SCIPY:
            return TestResult(
                comparison=f"{model_a} vs {model_b}",
                test_type="mann_whitney",
                statistic=0.0,
                p_value=1.0,
                significant=False,
            )
        
        statistic, p_value = mannwhitneyu(scores_a, scores_b, alternative="two-sided")
        
        return TestResult(
            comparison=f"{model_a} vs {model_b}",
            test_type="mann_whitney",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < self.alpha,
            n_samples=len(scores_a) + len(scores_b),
        )
    
    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        statistic_func: str = "mean",
        ci_level: float = 0.95,
        n_bootstrap: int = 10000,
        random_seed: int = 42,
    ) -> BootstrapResult:
        """Calculate bootstrap confidence interval.
        
        Args:
            scores: Score values
            statistic_func: Statistic to compute ("mean", "median", "std")
            ci_level: Confidence level
            n_bootstrap: Number of bootstrap samples
            random_seed: Random seed
            
        Returns:
            BootstrapResult object
        """
        np.random.seed(random_seed)
        scores = np.array(scores)
        
        stat_funcs = {
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
        }
        
        func = stat_funcs.get(statistic_func, np.mean)
        
        # Point estimate
        point_estimate = func(scores)
        
        # Bootstrap
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_stats.append(func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Percentile confidence interval
        lower_pct = (1 - ci_level) / 2 * 100
        upper_pct = (1 + ci_level) / 2 * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_pct)
        ci_upper = np.percentile(bootstrap_stats, upper_pct)
        
        return BootstrapResult(
            statistic=statistic_func,
            point_estimate=float(point_estimate),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            ci_level=ci_level,
            n_bootstrap=n_bootstrap,
        )
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < self.EFFECT_SIZE_THRESHOLDS["small"]:
            return "negligible"
        elif d < self.EFFECT_SIZE_THRESHOLDS["medium"]:
            return "small"
        elif d < self.EFFECT_SIZE_THRESHOLDS["large"]:
            return "medium"
        else:
            return "large"
    
    def compare_all_models(
        self,
        scores_df: pd.DataFrame,
        score_column: str = "overall",
        test_type: str = "paired",
    ) -> Dict[str, Any]:
        """Compare all pairs of models.
        
        Args:
            scores_df: DataFrame with model scores
            score_column: Column containing scores
            test_type: "paired" or "independent"
            
        Returns:
            Dict with all test results
        """
        models = scores_df["model"].unique().tolist()
        results = []
        
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                scores_a = scores_df[scores_df["model"] == model_a][score_column].tolist()
                scores_b = scores_df[scores_df["model"] == model_b][score_column].tolist()
                
                if test_type == "paired":
                    # For paired tests, we need matched items
                    # This is a simplification - in practice, would need to match by item_id
                    min_len = min(len(scores_a), len(scores_b))
                    scores_a = scores_a[:min_len]
                    scores_b = scores_b[:min_len]
                    
                    result = self.paired_t_test(scores_a, scores_b, model_a, model_b)
                else:
                    result = self.independent_t_test(scores_a, scores_b, model_a, model_b)
                
                results.append(result)
        
        # Sort by p-value
        results.sort(key=lambda x: x.p_value)
        
        # Apply Bonferroni correction
        n_tests = len(results)
        corrected_alpha = self.alpha / n_tests if n_tests > 0 else self.alpha
        
        return {
            "tests": [r.to_dict() for r in results],
            "n_tests": n_tests,
            "alpha": self.alpha,
            "corrected_alpha": corrected_alpha,
            "significant_comparisons": sum(1 for r in results if r.p_value < corrected_alpha),
        }
    
    def compute_model_confidence_intervals(
        self,
        scores_df: pd.DataFrame,
        score_column: str = "overall",
        ci_level: float = 0.95,
    ) -> Dict[str, BootstrapResult]:
        """Compute bootstrap CIs for all models.
        
        Args:
            scores_df: DataFrame with model scores
            score_column: Column containing scores
            ci_level: Confidence level
            
        Returns:
            Dict of model -> BootstrapResult
        """
        models = scores_df["model"].unique().tolist()
        results = {}
        
        for model in models:
            scores = scores_df[scores_df["model"] == model][score_column].tolist()
            result = self.bootstrap_confidence_interval(
                scores,
                statistic_func="mean",
                ci_level=ci_level,
            )
            results[model] = result
        
        return results
    
    def run_comprehensive_analysis(
        self,
        scores_df: pd.DataFrame,
        score_column: str = "overall",
    ) -> Dict[str, Any]:
        """Run comprehensive statistical analysis.
        
        Args:
            scores_df: DataFrame with model scores
            score_column: Column containing scores
            
        Returns:
            Complete analysis results
        """
        # Pairwise comparisons
        comparisons = self.compare_all_models(scores_df, score_column)
        
        # Confidence intervals
        cis = self.compute_model_confidence_intervals(scores_df, score_column)
        
        # Overall statistics
        models = scores_df["model"].unique().tolist()
        model_stats = {}
        
        for model in models:
            scores = scores_df[scores_df["model"] == model][score_column].tolist()
            model_stats[model] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "median": np.median(scores),
                "n": len(scores),
                "ci_lower": cis[model].ci_lower,
                "ci_upper": cis[model].ci_upper,
            }
        
        return {
            "pairwise_comparisons": comparisons,
            "confidence_intervals": {k: v.to_dict() for k, v in cis.items()},
            "model_statistics": model_stats,
        }

