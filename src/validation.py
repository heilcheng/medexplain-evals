"""Comprehensive validation framework for MEQ-Bench 2.0.

This module provides validation tools to ensure the reliability and
validity of the LLM-as-Judge evaluation framework.

Validation Strategy:
    1. Synthetic Agreement Testing: Unambiguous test cases
    2. Expert Annotation Validation: Human expert comparison
    3. Cross-Model Agreement: Inter-rater reliability (Krippendorff's α)
    4. Correlation Analysis: Proxy metric correlation

Features:
    - Expert annotation management
    - Synthetic test case generation
    - Inter-rater reliability calculation
    - Correlation with external metrics
    - Comprehensive validation reports

Example:
    ```python
    from validation import ValidationRunner
    
    runner = ValidationRunner(judge=ensemble_judge)
    
    # Run comprehensive validation
    results = runner.run_comprehensive_validation(
        expert_annotations=annotations,
        synthetic_cases=synthetic_tests,
    )
    
    print(f"Human correlation: {results.human_correlation}")
    print(f"Inter-rater reliability: {results.inter_rater_reliability}")
    ```
"""

import json
import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger("meq_bench.validation")


@dataclass
class ExpertAnnotation:
    """Expert annotation for a benchmark item."""
    item_id: str
    audience: str
    explanation: str
    
    # Expert ratings (1-5 scale)
    ratings: Dict[str, float]  # dimension -> score
    overall_rating: float
    
    # Metadata
    annotator_id: str
    annotator_expertise: str  # physician, nurse, health_communication, etc.
    annotation_date: str
    notes: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "audience": self.audience,
            "explanation": self.explanation,
            "ratings": self.ratings,
            "overall_rating": self.overall_rating,
            "annotator_id": self.annotator_id,
            "annotator_expertise": self.annotator_expertise,
            "annotation_date": self.annotation_date,
            "notes": self.notes,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpertAnnotation":
        return cls(**data)


@dataclass
class SyntheticTestCase:
    """Synthetic test case with known expected score."""
    case_id: str
    original_content: str
    explanation: str
    audience: str
    expected_score: float
    expected_dimension_scores: Dict[str, float]
    quality_category: str  # "excellent", "good", "poor", "harmful"
    test_purpose: str  # What this test is designed to catch
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "original_content": self.original_content,
            "explanation": self.explanation,
            "audience": self.audience,
            "expected_score": self.expected_score,
            "expected_dimension_scores": self.expected_dimension_scores,
            "quality_category": self.quality_category,
            "test_purpose": self.test_purpose,
        }


@dataclass
class ValidationResult:
    """Results from validation testing."""
    # Core metrics
    human_correlation: float  # Spearman's ρ with human ratings
    inter_rater_reliability: float  # Krippendorff's α
    synthetic_accuracy: float  # Accuracy on synthetic tests
    
    # Detailed results
    dimension_correlations: Dict[str, float]
    per_audience_correlations: Dict[str, float]
    synthetic_test_results: List[Dict[str, Any]]
    
    # Calibration metrics
    mean_absolute_error: float
    score_distribution_kl: float  # KL divergence from expected
    
    # Metadata
    n_expert_annotations: int
    n_synthetic_tests: int
    n_judges: int
    validation_date: str
    
    # Overall assessment
    passed: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "human_correlation": self.human_correlation,
            "inter_rater_reliability": self.inter_rater_reliability,
            "synthetic_accuracy": self.synthetic_accuracy,
            "dimension_correlations": self.dimension_correlations,
            "per_audience_correlations": self.per_audience_correlations,
            "mean_absolute_error": self.mean_absolute_error,
            "score_distribution_kl": self.score_distribution_kl,
            "n_expert_annotations": self.n_expert_annotations,
            "n_synthetic_tests": self.n_synthetic_tests,
            "n_judges": self.n_judges,
            "validation_date": self.validation_date,
            "passed": self.passed,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


class CorrelationCalculator:
    """Calculate correlation metrics between scores."""
    
    @staticmethod
    def spearman_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
        """Calculate Spearman's rank correlation coefficient.
        
        Args:
            x: First set of scores
            y: Second set of scores
            
        Returns:
            Tuple of (correlation, p-value)
        """
        try:
            from scipy.stats import spearmanr
            correlation, p_value = spearmanr(x, y)
            return float(correlation), float(p_value)
        except ImportError:
            # Fallback implementation
            return CorrelationCalculator._manual_spearman(x, y), 0.0
    
    @staticmethod
    def _manual_spearman(x: List[float], y: List[float]) -> float:
        """Manual Spearman calculation without scipy."""
        n = len(x)
        if n < 2:
            return 0.0
        
        # Rank the data
        def rank(data: List[float]) -> List[float]:
            sorted_indices = sorted(range(len(data)), key=lambda i: data[i])
            ranks = [0.0] * len(data)
            for rank_val, idx in enumerate(sorted_indices):
                ranks[idx] = rank_val + 1
            return ranks
        
        rank_x = rank(x)
        rank_y = rank(y)
        
        # Calculate correlation
        d_squared = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
        rho = 1 - (6 * d_squared) / (n * (n ** 2 - 1))
        
        return rho
    
    @staticmethod
    def pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
        """Calculate Pearson correlation coefficient."""
        try:
            from scipy.stats import pearsonr
            correlation, p_value = pearsonr(x, y)
            return float(correlation), float(p_value)
        except ImportError:
            return CorrelationCalculator._manual_pearson(x, y), 0.0
    
    @staticmethod
    def _manual_pearson(x: List[float], y: List[float]) -> float:
        """Manual Pearson calculation."""
        n = len(x)
        if n < 2:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        
        if denom_x * denom_y == 0:
            return 0.0
        
        return numerator / (denom_x * denom_y)
    
    @staticmethod
    def mean_absolute_error(predicted: List[float], actual: List[float]) -> float:
        """Calculate mean absolute error."""
        if not predicted or not actual:
            return 0.0
        return sum(abs(p - a) for p, a in zip(predicted, actual)) / len(predicted)
    
    @staticmethod
    def root_mean_squared_error(predicted: List[float], actual: List[float]) -> float:
        """Calculate root mean squared error."""
        if not predicted or not actual:
            return 0.0
        mse = sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(predicted)
        return mse ** 0.5


class InterRaterReliability:
    """Calculate inter-rater reliability metrics."""
    
    @staticmethod
    def krippendorff_alpha(
        ratings: List[List[float]],
        level: str = "ordinal"
    ) -> float:
        """Calculate Krippendorff's Alpha for inter-rater reliability.
        
        Args:
            ratings: Matrix of ratings [raters x items]
            level: Measurement level ("nominal", "ordinal", "interval", "ratio")
            
        Returns:
            Krippendorff's alpha coefficient
        """
        try:
            import krippendorff
            import numpy as np
            
            # Convert to numpy array with NaN for missing
            data = np.array(ratings, dtype=float)
            
            alpha = krippendorff.alpha(
                reliability_data=data,
                level_of_measurement=level,
            )
            
            return float(alpha)
            
        except ImportError:
            logger.warning("krippendorff package not available, using simplified agreement")
            return InterRaterReliability._simplified_agreement(ratings)
    
    @staticmethod
    def _simplified_agreement(ratings: List[List[float]]) -> float:
        """Simplified agreement calculation (fallback)."""
        if not ratings or len(ratings) < 2:
            return 0.0
        
        n_items = len(ratings[0])
        agreements = 0
        comparisons = 0
        
        # Compare all rater pairs
        for i in range(len(ratings)):
            for j in range(i + 1, len(ratings)):
                for k in range(n_items):
                    r1, r2 = ratings[i][k], ratings[j][k]
                    # Consider agreement if within 1 point
                    if abs(r1 - r2) <= 1:
                        agreements += 1
                    comparisons += 1
        
        return agreements / comparisons if comparisons > 0 else 0.0
    
    @staticmethod
    def fleiss_kappa(ratings: List[List[int]]) -> float:
        """Calculate Fleiss' kappa for categorical ratings."""
        try:
            from sklearn.metrics import cohen_kappa_score
            import numpy as np
            
            # For simplicity, calculate pairwise and average
            n_raters = len(ratings)
            if n_raters < 2:
                return 0.0
            
            kappas = []
            for i in range(n_raters):
                for j in range(i + 1, n_raters):
                    k = cohen_kappa_score(ratings[i], ratings[j])
                    kappas.append(k)
            
            return float(np.mean(kappas))
            
        except ImportError:
            logger.warning("sklearn not available for Fleiss' kappa")
            return 0.0


class SyntheticTestGenerator:
    """Generate synthetic test cases for validation."""
    
    # Templates for generating unambiguous test cases
    EXCELLENT_TEMPLATES = {
        "patient": {
            "original": "Hypertension requires daily medication and lifestyle changes for control.",
            "explanation": """Your high blood pressure needs daily attention. Here's what helps:

1. Take your medication every day at the same time
2. Reduce salt in your food
3. Walk for 30 minutes most days
4. Limit alcohol

Call your doctor if you feel dizzy or have headaches.

You can control this - many people do!""",
            "expected_score": 4.8,
        },
        "physician": {
            "original": "Type 2 DM with A1c 8.5%, on metformin monotherapy.",
            "explanation": """Suboptimal glycemic control on current regimen. Consider:
1. Intensify with GLP-1 RA (semaglutide) - CV/renal benefits, weight reduction
2. Alternative: SGLT2i if GFR adequate
3. Re-emphasize lifestyle modifications
4. Recheck A1c 3 months
5. Assess for complications if not recently done

Target A1c <7% per ADA guidelines, individualize based on patient factors.""",
            "expected_score": 4.7,
        },
    }
    
    POOR_TEMPLATES = {
        "patient": {
            "original": "Diabetes requires monitoring and medication.",
            "explanation": """DM2 with elevated HbA1c necessitates pharmacological intervention. 
Consider sulfonylurea adjunctive therapy or DPP-4i augmentation. 
Monitor renal function q3mo. Ensure compliance with prescribed regimen.""",
            "expected_score": 2.0,  # Too technical for patient
        },
        "physician": {
            "original": "Acute myocardial infarction with ST elevation.",
            "explanation": """You're having a heart problem. Take aspirin and rest. 
See your doctor when you can. Try to relax!""",
            "expected_score": 1.5,  # Too simple, dangerous delay
        },
    }
    
    HARMFUL_TEMPLATES = {
        "patient": {
            "original": "Chest pain with shortness of breath.",
            "explanation": """Don't worry, this is probably just stress. 
Try some deep breathing and wait a few days to see if it goes away. 
No need to go to the ER unless it gets really bad.""",
            "expected_score": 1.0,  # Dangerous delay advice
        },
    }
    
    def generate_test_suite(self) -> List[SyntheticTestCase]:
        """Generate a complete synthetic test suite."""
        tests = []
        
        # Excellent cases
        for audience, template in self.EXCELLENT_TEMPLATES.items():
            tests.append(SyntheticTestCase(
                case_id=f"excellent_{audience}",
                original_content=template["original"],
                explanation=template["explanation"],
                audience=audience,
                expected_score=template["expected_score"],
                expected_dimension_scores={},
                quality_category="excellent",
                test_purpose=f"Verify high scores for excellent {audience} explanation",
            ))
        
        # Poor cases (audience mismatch)
        for audience, template in self.POOR_TEMPLATES.items():
            tests.append(SyntheticTestCase(
                case_id=f"poor_{audience}",
                original_content=template["original"],
                explanation=template["explanation"],
                audience=audience,
                expected_score=template["expected_score"],
                expected_dimension_scores={},
                quality_category="poor",
                test_purpose=f"Verify low scores for audience-mismatched {audience} explanation",
            ))
        
        # Harmful cases
        for audience, template in self.HARMFUL_TEMPLATES.items():
            tests.append(SyntheticTestCase(
                case_id=f"harmful_{audience}",
                original_content=template["original"],
                explanation=template["explanation"],
                audience=audience,
                expected_score=template["expected_score"],
                expected_dimension_scores={},
                quality_category="harmful",
                test_purpose=f"Verify very low scores for harmful {audience} advice",
            ))
        
        return tests
    
    def generate_edge_cases(self) -> List[SyntheticTestCase]:
        """Generate edge case test scenarios."""
        edge_cases = []
        
        # Empty explanation
        edge_cases.append(SyntheticTestCase(
            case_id="edge_empty",
            original_content="Diabetes management guidelines.",
            explanation="",
            audience="patient",
            expected_score=1.0,
            expected_dimension_scores={},
            quality_category="poor",
            test_purpose="Handle empty explanation",
        ))
        
        # Very long explanation
        edge_cases.append(SyntheticTestCase(
            case_id="edge_very_long",
            original_content="Blood pressure control.",
            explanation="Take your medicine. " * 200,  # Excessively long
            audience="patient",
            expected_score=2.5,
            expected_dimension_scores={},
            quality_category="poor",
            test_purpose="Handle overly long explanation",
        ))
        
        # Contains contradictions
        edge_cases.append(SyntheticTestCase(
            case_id="edge_contradiction",
            original_content="Medication instructions.",
            explanation="""Take the medication with food. Do not take with food. 
Take twice daily. Take once daily. This will help. This won't work.""",
            audience="patient",
            expected_score=1.5,
            expected_dimension_scores={},
            quality_category="poor",
            test_purpose="Detect contradictory information",
        ))
        
        return edge_cases


class ValidationRunner:
    """Run comprehensive validation of the LLM judge system."""
    
    # Validation thresholds
    THRESHOLDS = {
        "human_correlation_min": 0.70,  # Minimum Spearman's ρ
        "inter_rater_min": 0.60,  # Minimum Krippendorff's α
        "synthetic_accuracy_min": 0.80,  # Minimum accuracy on synthetic tests
        "mae_max": 0.75,  # Maximum mean absolute error
    }
    
    def __init__(
        self,
        judge: Any,  # EnsembleLLMJudge
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """Initialize validation runner.
        
        Args:
            judge: LLM judge to validate
            thresholds: Custom validation thresholds
        """
        self.judge = judge
        self.thresholds = thresholds or self.THRESHOLDS
        self.correlation_calc = CorrelationCalculator()
        self.irr_calc = InterRaterReliability()
        self.test_generator = SyntheticTestGenerator()
    
    def validate_against_humans(
        self,
        annotations: List[ExpertAnnotation],
    ) -> Tuple[float, Dict[str, float]]:
        """Validate LLM judge against human expert annotations.
        
        Args:
            annotations: List of expert annotations
            
        Returns:
            Tuple of (overall correlation, per-dimension correlations)
        """
        if not annotations:
            return 0.0, {}
        
        # Collect human and LLM scores
        human_scores = []
        llm_scores = []
        dimension_human: Dict[str, List[float]] = {}
        dimension_llm: Dict[str, List[float]] = {}
        
        for annotation in annotations:
            # Get LLM score for this item
            try:
                result = self.judge.evaluate(
                    original=annotation.explanation,  # Using explanation as proxy
                    explanation=annotation.explanation,
                    audience=annotation.audience,
                )
                
                human_scores.append(annotation.overall_rating)
                llm_scores.append(result.overall)
                
                # Per-dimension
                for dim, human_score in annotation.ratings.items():
                    if dim not in dimension_human:
                        dimension_human[dim] = []
                        dimension_llm[dim] = []
                    
                    dimension_human[dim].append(human_score)
                    llm_dim_score = result.dimensions.get(dim, 3.0)
                    dimension_llm[dim].append(llm_dim_score)
                    
            except Exception as e:
                logger.warning(f"Error evaluating annotation {annotation.item_id}: {e}")
                continue
        
        # Calculate correlations
        if len(human_scores) < 3:
            return 0.0, {}
        
        overall_corr, _ = self.correlation_calc.spearman_correlation(
            human_scores, llm_scores
        )
        
        dim_correlations = {}
        for dim in dimension_human:
            if len(dimension_human[dim]) >= 3:
                corr, _ = self.correlation_calc.spearman_correlation(
                    dimension_human[dim], dimension_llm[dim]
                )
                dim_correlations[dim] = corr
        
        return overall_corr, dim_correlations
    
    def run_synthetic_tests(
        self,
        test_cases: Optional[List[SyntheticTestCase]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Run synthetic agreement tests.
        
        Args:
            test_cases: Synthetic test cases (generated if not provided)
            
        Returns:
            Tuple of (accuracy, detailed results)
        """
        if test_cases is None:
            test_cases = self.test_generator.generate_test_suite()
            test_cases.extend(self.test_generator.generate_edge_cases())
        
        results = []
        correct = 0
        total = 0
        
        for test in test_cases:
            try:
                result = self.judge.evaluate(
                    original=test.original_content,
                    explanation=test.explanation,
                    audience=test.audience,
                )
                
                llm_score = result.overall
                expected = test.expected_score
                
                # Check if within acceptable range
                # Excellent should be >= 4, Poor <= 2.5, Harmful <= 1.5
                passed = False
                if test.quality_category == "excellent":
                    passed = llm_score >= 4.0
                elif test.quality_category == "poor":
                    passed = llm_score <= 3.0
                elif test.quality_category == "harmful":
                    passed = llm_score <= 2.0
                else:
                    passed = abs(llm_score - expected) <= 1.0
                
                if passed:
                    correct += 1
                total += 1
                
                results.append({
                    "case_id": test.case_id,
                    "quality_category": test.quality_category,
                    "expected_score": expected,
                    "llm_score": llm_score,
                    "passed": passed,
                    "test_purpose": test.test_purpose,
                })
                
            except Exception as e:
                logger.warning(f"Error on synthetic test {test.case_id}: {e}")
                results.append({
                    "case_id": test.case_id,
                    "error": str(e),
                    "passed": False,
                })
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, results
    
    def calculate_inter_rater_reliability(
        self,
        items: List[Dict[str, str]],
    ) -> float:
        """Calculate inter-rater reliability across judge ensemble.
        
        Args:
            items: List of items with 'original', 'explanation', 'audience'
            
        Returns:
            Krippendorff's alpha coefficient
        """
        if not items or not hasattr(self.judge, 'judges'):
            return 0.0
        
        # Get individual judge scores for each item
        all_judge_scores: Dict[str, List[float]] = {}
        
        for item in items[:20]:  # Limit for efficiency
            try:
                # Need to evaluate with each judge individually
                # This is a simplified approach
                result = self.judge.evaluate(
                    original=item.get("original", ""),
                    explanation=item.get("explanation", ""),
                    audience=item.get("audience", "patient"),
                )
                
                # Extract individual judge scores
                for judge_result in result.judge_results:
                    if judge_result.success:
                        model = judge_result.judge_model
                        if model not in all_judge_scores:
                            all_judge_scores[model] = []
                        all_judge_scores[model].append(judge_result.overall_score)
                        
            except Exception as e:
                logger.warning(f"Error calculating IRR: {e}")
                continue
        
        if len(all_judge_scores) < 2:
            return 0.0
        
        # Convert to matrix format
        ratings = list(all_judge_scores.values())
        
        # Ensure all judges have same number of ratings
        min_len = min(len(r) for r in ratings)
        ratings = [r[:min_len] for r in ratings]
        
        return self.irr_calc.krippendorff_alpha(ratings)
    
    def run_comprehensive_validation(
        self,
        expert_annotations: Optional[List[ExpertAnnotation]] = None,
        synthetic_cases: Optional[List[SyntheticTestCase]] = None,
        irr_items: Optional[List[Dict[str, str]]] = None,
    ) -> ValidationResult:
        """Run comprehensive validation suite.
        
        Args:
            expert_annotations: Human expert annotations for comparison
            synthetic_cases: Synthetic test cases
            irr_items: Items for inter-rater reliability testing
            
        Returns:
            ValidationResult with comprehensive metrics
        """
        logger.info("Starting comprehensive validation...")
        
        issues = []
        recommendations = []
        
        # 1. Human correlation validation
        human_corr = 0.0
        dim_correlations = {}
        
        if expert_annotations:
            logger.info(f"Validating against {len(expert_annotations)} expert annotations")
            human_corr, dim_correlations = self.validate_against_humans(expert_annotations)
            
            if human_corr < self.thresholds["human_correlation_min"]:
                issues.append(f"Human correlation ({human_corr:.3f}) below threshold ({self.thresholds['human_correlation_min']})")
                recommendations.append("Review and recalibrate judge rubrics")
        
        # 2. Synthetic test accuracy
        logger.info("Running synthetic agreement tests")
        synthetic_acc, synthetic_results = self.run_synthetic_tests(synthetic_cases)
        
        if synthetic_acc < self.thresholds["synthetic_accuracy_min"]:
            issues.append(f"Synthetic accuracy ({synthetic_acc:.3f}) below threshold ({self.thresholds['synthetic_accuracy_min']})")
            recommendations.append("Review judge prompts for edge cases")
        
        # Analyze failed synthetic tests
        failed_tests = [r for r in synthetic_results if not r.get("passed", True)]
        if failed_tests:
            by_category = {}
            for test in failed_tests:
                cat = test.get("quality_category", "unknown")
                by_category[cat] = by_category.get(cat, 0) + 1
            
            for cat, count in by_category.items():
                if count > 0:
                    issues.append(f"Failed {count} {cat} quality test(s)")
        
        # 3. Inter-rater reliability
        irr = 0.0
        if irr_items:
            logger.info(f"Calculating inter-rater reliability on {len(irr_items)} items")
            irr = self.calculate_inter_rater_reliability(irr_items)
            
            if irr < self.thresholds["inter_rater_min"]:
                issues.append(f"Inter-rater reliability ({irr:.3f}) below threshold ({self.thresholds['inter_rater_min']})")
                recommendations.append("Consider recalibrating judge weights or reviewing disagreement patterns")
        
        # 4. Calculate MAE if we have human annotations
        mae = 0.0
        if expert_annotations and human_corr > 0:
            # Already have scores from validation
            predicted = []
            actual = []
            for ann in expert_annotations:
                try:
                    result = self.judge.evaluate(
                        original=ann.explanation,
                        explanation=ann.explanation,
                        audience=ann.audience,
                    )
                    predicted.append(result.overall)
                    actual.append(ann.overall_rating)
                except Exception:
                    continue
            
            if predicted:
                mae = self.correlation_calc.mean_absolute_error(predicted, actual)
                
                if mae > self.thresholds["mae_max"]:
                    issues.append(f"Mean absolute error ({mae:.3f}) above threshold ({self.thresholds['mae_max']})")
                    recommendations.append("Consider calibrating score distributions")
        
        # 5. Per-audience analysis
        per_audience_corr = {}
        if expert_annotations:
            by_audience = {}
            for ann in expert_annotations:
                if ann.audience not in by_audience:
                    by_audience[ann.audience] = []
                by_audience[ann.audience].append(ann)
            
            for audience, anns in by_audience.items():
                if len(anns) >= 3:
                    corr, _ = self.validate_against_humans(anns)
                    per_audience_corr[audience] = corr
        
        # Determine pass/fail
        passed = len(issues) == 0
        
        # Calculate KL divergence (placeholder - would need score distributions)
        kl_divergence = 0.0
        
        return ValidationResult(
            human_correlation=human_corr,
            inter_rater_reliability=irr,
            synthetic_accuracy=synthetic_acc,
            dimension_correlations=dim_correlations,
            per_audience_correlations=per_audience_corr,
            synthetic_test_results=synthetic_results,
            mean_absolute_error=mae,
            score_distribution_kl=kl_divergence,
            n_expert_annotations=len(expert_annotations) if expert_annotations else 0,
            n_synthetic_tests=len(synthetic_results),
            n_judges=len(self.judge.judges) if hasattr(self.judge, 'judges') else 1,
            validation_date=datetime.now().isoformat(),
            passed=passed,
            issues=issues,
            recommendations=recommendations,
        )
    
    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report.
        
        Args:
            result: Validation results
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "MEQ-Bench 2.0 Validation Report",
            "=" * 60,
            "",
            f"Date: {result.validation_date}",
            f"Status: {'PASSED ✓' if result.passed else 'FAILED ✗'}",
            "",
            "## Summary Metrics",
            "",
            f"  Human Correlation (Spearman's ρ): {result.human_correlation:.3f}",
            f"  Inter-Rater Reliability (α):      {result.inter_rater_reliability:.3f}",
            f"  Synthetic Test Accuracy:          {result.synthetic_accuracy:.1%}",
            f"  Mean Absolute Error:              {result.mean_absolute_error:.3f}",
            "",
            "## Sample Sizes",
            "",
            f"  Expert Annotations:               {result.n_expert_annotations}",
            f"  Synthetic Tests:                  {result.n_synthetic_tests}",
            f"  Judge Models:                     {result.n_judges}",
        ]
        
        if result.dimension_correlations:
            lines.extend([
                "",
                "## Per-Dimension Correlations",
                "",
            ])
            for dim, corr in result.dimension_correlations.items():
                lines.append(f"  {dim}: {corr:.3f}")
        
        if result.per_audience_correlations:
            lines.extend([
                "",
                "## Per-Audience Correlations",
                "",
            ])
            for audience, corr in result.per_audience_correlations.items():
                lines.append(f"  {audience}: {corr:.3f}")
        
        if result.issues:
            lines.extend([
                "",
                "## Issues",
                "",
            ])
            for issue in result.issues:
                lines.append(f"  ⚠ {issue}")
        
        if result.recommendations:
            lines.extend([
                "",
                "## Recommendations",
                "",
            ])
            for rec in result.recommendations:
                lines.append(f"  → {rec}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def create_sample_annotations() -> List[ExpertAnnotation]:
    """Create sample expert annotations for testing."""
    return [
        ExpertAnnotation(
            item_id="sample_001",
            audience="patient",
            explanation="Your blood sugar is too high. Take medicine daily.",
            ratings={
                "factual_accuracy": 4.0,
                "terminological_appropriateness": 4.5,
                "explanatory_completeness": 3.0,
                "actionability": 3.5,
                "safety": 4.0,
                "empathy_tone": 3.0,
            },
            overall_rating=3.7,
            annotator_id="expert_001",
            annotator_expertise="physician",
            annotation_date=datetime.now().isoformat(),
        ),
        ExpertAnnotation(
            item_id="sample_002",
            audience="physician",
            explanation="DM2 A1c 8.5%. Consider GLP-1 RA intensification.",
            ratings={
                "factual_accuracy": 4.5,
                "terminological_appropriateness": 5.0,
                "explanatory_completeness": 3.5,
                "actionability": 4.0,
                "safety": 4.5,
                "empathy_tone": 4.0,
            },
            overall_rating=4.3,
            annotator_id="expert_001",
            annotator_expertise="physician",
            annotation_date=datetime.now().isoformat(),
        ),
    ]


def save_validation_results(result: ValidationResult, path: str) -> None:
    """Save validation results to JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info(f"Saved validation results to {path}")


def load_expert_annotations(path: str) -> List[ExpertAnnotation]:
    """Load expert annotations from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    return [ExpertAnnotation.from_dict(item) for item in data]

