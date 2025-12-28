"""Error analysis tools for identifying failure cases.

This module identifies and categorizes low-performing explanations,
safety failures, and audience mismatches.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ErrorCase:
    """Individual error case."""
    item_id: str
    model: str
    audience: str
    overall_score: float
    error_type: str  # low_score, safety_failure, audience_mismatch, judge_disagreement
    details: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None
    original_content: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorAnalysisReport:
    """Complete error analysis report."""
    total_errors: int
    errors_by_type: Dict[str, int]
    errors_by_model: Dict[str, int]
    errors_by_audience: Dict[str, int]
    error_cases: List[ErrorCase]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_errors": self.total_errors,
            "errors_by_type": self.errors_by_type,
            "errors_by_model": self.errors_by_model,
            "errors_by_audience": self.errors_by_audience,
            "error_cases": [e.to_dict() for e in self.error_cases],
            "recommendations": self.recommendations,
        }


class ErrorAnalyzer:
    """Analyze failure cases in evaluation results."""
    
    # Thresholds for error detection
    LOW_SCORE_THRESHOLD = 2.5  # Below this is considered low
    SAFETY_FAIL_THRESHOLD = 0.5  # Safety score below this
    DISAGREEMENT_THRESHOLD = 1.5  # Score std dev above this
    
    ERROR_TYPES = [
        "low_score",
        "safety_failure",
        "audience_mismatch",
        "judge_disagreement",
        "dimension_failure",
    ]
    
    def __init__(
        self,
        low_score_threshold: float = 2.5,
        safety_threshold: float = 0.5,
    ):
        """Initialize error analyzer.
        
        Args:
            low_score_threshold: Threshold for low score detection
            safety_threshold: Threshold for safety failure
        """
        self.low_score_threshold = low_score_threshold
        self.safety_threshold = safety_threshold
        self.scores_df: Optional[pd.DataFrame] = None
        self.explanations: Dict[str, Dict] = {}
    
    def load_data(
        self,
        scores_df: pd.DataFrame,
        explanations_dir: Optional[str] = None,
    ) -> None:
        """Load scores and optionally explanations.
        
        Args:
            scores_df: DataFrame with scores
            explanations_dir: Optional directory with explanations
        """
        self.scores_df = scores_df
        
        if explanations_dir:
            exp_dir = Path(explanations_dir)
            for exp_file in exp_dir.glob("*.json"):
                if exp_file.name in ("checkpoint.json", "summary.json"):
                    continue
                with open(exp_file, "r") as f:
                    data = json.load(f)
                item_id = data.get("item_id", exp_file.stem)
                self.explanations[item_id] = data
    
    def find_low_scores(self) -> List[ErrorCase]:
        """Find items with overall scores below threshold.
        
        Returns:
            List of low-score error cases
        """
        errors = []
        
        if self.scores_df is None:
            return errors
        
        low_df = self.scores_df[self.scores_df["overall"] < self.low_score_threshold]
        
        for _, row in low_df.iterrows():
            errors.append(ErrorCase(
                item_id=row.get("item_id", "unknown"),
                model=row.get("model", "unknown"),
                audience=row.get("audience", "unknown"),
                overall_score=row.get("overall", 0),
                error_type="low_score",
                details={
                    "threshold": self.low_score_threshold,
                    "dimensions": {
                        col: row[col] for col in row.index
                        if col not in ["item_id", "model", "audience", "overall"]
                    }
                },
            ))
        
        return errors
    
    def find_safety_failures(self) -> List[ErrorCase]:
        """Find items that failed safety checks.
        
        Returns:
            List of safety failure cases
        """
        errors = []
        
        if self.scores_df is None:
            return errors
        
        # Check safety_passed column
        if "safety_passed" in self.scores_df.columns:
            failed_df = self.scores_df[self.scores_df["safety_passed"] == False]
        elif "safety" in self.scores_df.columns:
            failed_df = self.scores_df[self.scores_df["safety"] < self.safety_threshold * 5]
        else:
            return errors
        
        for _, row in failed_df.iterrows():
            errors.append(ErrorCase(
                item_id=row.get("item_id", "unknown"),
                model=row.get("model", "unknown"),
                audience=row.get("audience", "unknown"),
                overall_score=row.get("overall", 0),
                error_type="safety_failure",
                details={
                    "safety_score": row.get("safety", row.get("safety_eval_score", 0)),
                    "safety_passed": row.get("safety_passed", False),
                },
            ))
        
        return errors
    
    def find_audience_mismatches(self) -> List[ErrorCase]:
        """Find explanations inappropriate for target audience.
        
        Detection based on terminological_appropriateness score.
        
        Returns:
            List of audience mismatch cases
        """
        errors = []
        
        if self.scores_df is None or "terminological_appropriateness" not in self.scores_df.columns:
            return errors
        
        # Low terminology appropriateness indicates mismatch
        mismatch_df = self.scores_df[self.scores_df["terminological_appropriateness"] < 2.5]
        
        for _, row in mismatch_df.iterrows():
            errors.append(ErrorCase(
                item_id=row.get("item_id", "unknown"),
                model=row.get("model", "unknown"),
                audience=row.get("audience", "unknown"),
                overall_score=row.get("overall", 0),
                error_type="audience_mismatch",
                details={
                    "terminology_score": row.get("terminological_appropriateness", 0),
                    "target_audience": row.get("audience", "unknown"),
                },
            ))
        
        return errors
    
    def find_judge_disagreements(self) -> List[ErrorCase]:
        """Find items where judges significantly disagreed.
        
        Returns:
            List of disagreement cases
        """
        errors = []
        
        if self.scores_df is None or "judge_agreement" not in self.scores_df.columns:
            return errors
        
        # Low agreement score indicates disagreement
        disagree_df = self.scores_df[self.scores_df["judge_agreement"] < 0.7]
        
        for _, row in disagree_df.iterrows():
            errors.append(ErrorCase(
                item_id=row.get("item_id", "unknown"),
                model=row.get("model", "unknown"),
                audience=row.get("audience", "unknown"),
                overall_score=row.get("overall", 0),
                error_type="judge_disagreement",
                details={
                    "agreement_score": row.get("judge_agreement", 0),
                },
            ))
        
        return errors
    
    def find_dimension_failures(
        self,
        dimension: str,
        threshold: float = 2.0,
    ) -> List[ErrorCase]:
        """Find items failing on a specific dimension.
        
        Args:
            dimension: Dimension to check
            threshold: Score threshold
            
        Returns:
            List of dimension failure cases
        """
        errors = []
        
        if self.scores_df is None or dimension not in self.scores_df.columns:
            return errors
        
        failed_df = self.scores_df[self.scores_df[dimension] < threshold]
        
        for _, row in failed_df.iterrows():
            errors.append(ErrorCase(
                item_id=row.get("item_id", "unknown"),
                model=row.get("model", "unknown"),
                audience=row.get("audience", "unknown"),
                overall_score=row.get("overall", 0),
                error_type="dimension_failure",
                details={
                    "dimension": dimension,
                    "score": row[dimension],
                    "threshold": threshold,
                },
            ))
        
        return errors
    
    def analyze_errors_by_category(
        self,
        error_cases: List[ErrorCase],
    ) -> Dict[str, Any]:
        """Categorize errors by various dimensions.
        
        Args:
            error_cases: List of error cases
            
        Returns:
            Error breakdown by category
        """
        by_type: Dict[str, int] = defaultdict(int)
        by_model: Dict[str, int] = defaultdict(int)
        by_audience: Dict[str, int] = defaultdict(int)
        
        for error in error_cases:
            by_type[error.error_type] += 1
            by_model[error.model] += 1
            by_audience[error.audience] += 1
        
        return {
            "by_type": dict(by_type),
            "by_model": dict(by_model),
            "by_audience": dict(by_audience),
        }
    
    def generate_recommendations(
        self,
        error_cases: List[ErrorCase],
        categories: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations based on errors.
        
        Args:
            error_cases: List of error cases
            categories: Error categorization
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        by_type = categories.get("by_type", {})
        by_model = categories.get("by_model", {})
        by_audience = categories.get("by_audience", {})
        
        # Type-based recommendations
        if by_type.get("safety_failure", 0) > len(error_cases) * 0.1:
            recommendations.append(
                "High safety failure rate detected. Consider reviewing prompt templates "
                "to include stronger safety guidelines and contraindication warnings."
            )
        
        if by_type.get("audience_mismatch", 0) > len(error_cases) * 0.15:
            recommendations.append(
                "Significant audience mismatch issues. Recommend enhancing audience "
                "persona descriptions in prompts and adding reading level targeting."
            )
        
        if by_type.get("judge_disagreement", 0) > len(error_cases) * 0.2:
            recommendations.append(
                "High judge disagreement rate. Consider calibrating the ensemble "
                "with additional human annotations or adjusting judge weights."
            )
        
        # Model-based recommendations
        worst_model = max(by_model.items(), key=lambda x: x[1], default=(None, 0))
        if worst_model[0] and worst_model[1] > len(error_cases) * 0.3:
            recommendations.append(
                f"Model '{worst_model[0]}' has disproportionately high error rate. "
                "Consider model-specific prompt optimization or excluding from benchmark."
            )
        
        # Audience-based recommendations
        worst_audience = max(by_audience.items(), key=lambda x: x[1], default=(None, 0))
        if worst_audience[0] and worst_audience[1] > len(error_cases) * 0.35:
            recommendations.append(
                f"'{worst_audience[0]}' audience has highest error rate. "
                "Review persona definition and consider adding more specific guidance."
            )
        
        if not recommendations:
            recommendations.append(
                "Error distribution appears balanced. Focus on individual case remediation."
            )
        
        return recommendations
    
    def run_full_analysis(self) -> ErrorAnalysisReport:
        """Run complete error analysis.
        
        Returns:
            ErrorAnalysisReport with all findings
        """
        all_errors = []
        
        # Collect all error types
        all_errors.extend(self.find_low_scores())
        all_errors.extend(self.find_safety_failures())
        all_errors.extend(self.find_audience_mismatches())
        all_errors.extend(self.find_judge_disagreements())
        
        # Categorize
        categories = self.analyze_errors_by_category(all_errors)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(all_errors, categories)
        
        # Deduplicate errors (same item-model-audience can have multiple error types)
        unique_errors: Dict[str, ErrorCase] = {}
        for error in all_errors:
            key = f"{error.item_id}_{error.model}_{error.audience}_{error.error_type}"
            if key not in unique_errors:
                unique_errors[key] = error
        
        report = ErrorAnalysisReport(
            total_errors=len(unique_errors),
            errors_by_type=categories["by_type"],
            errors_by_model=categories["by_model"],
            errors_by_audience=categories["by_audience"],
            error_cases=list(unique_errors.values()),
            recommendations=recommendations,
        )
        
        logger.info(f"Error analysis complete: {report.total_errors} errors found")
        
        return report
    
    def export_error_cases(
        self,
        error_cases: List[ErrorCase],
        output_path: str,
        format: str = "json",
    ) -> None:
        """Export error cases to file.
        
        Args:
            error_cases: List of error cases
            output_path: Output file path
            format: Output format (json or csv)
        """
        if format == "json":
            with open(output_path, "w") as f:
                json.dump([e.to_dict() for e in error_cases], f, indent=2)
        elif format == "csv":
            df = pd.DataFrame([e.to_dict() for e in error_cases])
            df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(error_cases)} error cases to {output_path}")
    
    def get_worst_cases(
        self,
        n: int = 10,
        error_type: Optional[str] = None,
    ) -> List[ErrorCase]:
        """Get the n worst error cases.
        
        Args:
            n: Number of cases to return
            error_type: Optional filter by error type
            
        Returns:
            List of worst cases
        """
        all_errors = []
        all_errors.extend(self.find_low_scores())
        all_errors.extend(self.find_safety_failures())
        
        if error_type:
            all_errors = [e for e in all_errors if e.error_type == error_type]
        
        # Sort by score (lowest first)
        sorted_errors = sorted(all_errors, key=lambda x: x.overall_score)
        
        return sorted_errors[:n]

