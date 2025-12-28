"""Validation framework for LLM-as-a-Judge evaluation.

This module provides validation functionality for the LLM-as-a-Judge component
used in MedExplain-Evals. It allows comparison of LLM scores with human ratings to
assess the reliability and validity of automated evaluation.

The validation process includes correlation analysis, statistical significance
testing, and detailed performance metrics to ensure the LLM judge produces
reliable and consistent scores.

Example:
    ```python
    from validate_judge import validate_llm_judge
    from src.evaluator import MedExplainEvaluator
    
    # Prepare validation data
    predictions = [
        {'generated_explanation': 'Patient explanation...', 'human_rating': 4.2},
        {'generated_explanation': 'Another explanation...', 'human_rating': 3.8},
    ]
    
    # Initialize LLM judge
    llm_judge = MedExplainEvaluator()
    
    # Run validation
    correlation, p_value = validate_llm_judge(predictions, llm_judge)
    print(f"Correlation: {correlation:.3f}, p-value: {p_value:.3f}")
    ```
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy.stats import spearmanr, pearsonr
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger('medexplain.validation')


@dataclass
class ValidationResult:
    """Container for validation results.
    
    Attributes:
        correlation_coefficient: Spearman's rank correlation coefficient.
        p_value: Statistical significance p-value.
        n_samples: Number of samples used in validation.
        llm_scores: List of LLM-generated scores.
        human_ratings: List of human ratings.
        score_differences: List of differences between LLM and human scores.
        mean_absolute_error: Mean absolute error between LLM and human scores.
        correlation_type: Type of correlation used ('spearman' or 'pearson').
    """
    correlation_coefficient: float
    p_value: float
    n_samples: int
    llm_scores: List[float]
    human_ratings: List[float]
    score_differences: List[float]
    mean_absolute_error: float
    correlation_type: str


def validate_llm_judge(
    predictions: List[Dict[str, Any]], 
    llm_judge: Any,
    medical_content: Optional[str] = None,
    audience: str = 'patient',
    correlation_type: str = 'spearman'
) -> Tuple[float, float]:
    """Validate LLM-as-a-Judge by comparing with human ratings.
    
    This function evaluates the reliability of an LLM judge by comparing its
    scores with human ratings using correlation analysis. It provides a 
    quantitative measure of how well the LLM judge aligns with human judgment.
    
    Args:
        predictions: List of dictionaries containing validation data. Each dictionary
            should have:
            - 'generated_explanation': The explanation text to be scored
            - 'human_rating': The human-provided rating (float, typically 1-5 scale)
            Optional keys:
            - 'medical_content': Original medical content (if not provided globally)
            - 'audience': Target audience (if different from global setting)
            
        llm_judge: Instance of the LLM judge class (e.g., MedExplainEvaluator).
            Must have a method to score explanations.
            
        medical_content: Original medical content for context. If None, each
            prediction should include its own medical_content.
            
        audience: Target audience for scoring ('physician', 'nurse', 'patient', 'caregiver').
            Defaults to 'patient'.
            
        correlation_type: Type of correlation to compute ('spearman' or 'pearson').
            Spearman is recommended for ordinal ratings.
            
    Returns:
        Tuple containing:
        - correlation_coefficient: Correlation between LLM and human scores
        - p_value: Statistical significance of the correlation
        
    Raises:
        ValueError: If predictions list is empty, contains invalid data, or if
            required fields are missing.
        AttributeError: If llm_judge doesn't have required scoring methods.
        
    Example:
        ```python
        # Example validation data
        predictions = [
            {
                'generated_explanation': 'High blood pressure means your heart works harder.',
                'human_rating': 4.2
            },
            {
                'generated_explanation': 'Hypertension is cardiovascular condition...',
                'human_rating': 3.1
            }
        ]
        
        # Run validation
        correlation, p_value = validate_llm_judge(
            predictions, 
            llm_judge,
            medical_content='Hypertension explanation',
            audience='patient'
        )
        ```
    """
    # Input validation
    if not predictions:
        raise ValueError("Predictions list cannot be empty")
    
    if not hasattr(llm_judge, 'evaluate_explanation') and not hasattr(llm_judge, 'evaluate_all_audiences'):
        raise AttributeError("LLM judge must have evaluation methods")
    
    if correlation_type not in ['spearman', 'pearson']:
        raise ValueError("Correlation type must be 'spearman' or 'pearson'")
    
    logger.info(f"Starting LLM judge validation with {len(predictions)} samples")
    
    # Extract LLM scores and human ratings
    llm_scores = []
    human_ratings = []
    
    for i, prediction in enumerate(predictions):
        try:
            # Validate prediction structure
            if not isinstance(prediction, dict):
                logger.warning(f"Skipping prediction {i}: not a dictionary")
                continue
                
            if 'generated_explanation' not in prediction:
                logger.warning(f"Skipping prediction {i}: missing 'generated_explanation'")
                continue
                
            if 'human_rating' not in prediction:
                logger.warning(f"Skipping prediction {i}: missing 'human_rating'")
                continue
            
            explanation = prediction['generated_explanation']
            human_rating = prediction['human_rating']
            
            # Validate explanation text
            if not isinstance(explanation, str) or not explanation.strip():
                logger.warning(f"Skipping prediction {i}: invalid explanation text")
                continue
            
            # Validate human rating
            try:
                human_rating = float(human_rating)
            except (ValueError, TypeError):
                logger.warning(f"Skipping prediction {i}: invalid human rating")
                continue
            
            # Get medical content for this prediction
            current_medical_content = prediction.get('medical_content', medical_content)
            if not current_medical_content:
                logger.warning(f"Skipping prediction {i}: no medical content available")
                continue
            
            # Get audience for this prediction  
            current_audience = prediction.get('audience', audience)
            
            # Score the explanation using the LLM judge
            llm_score = _score_explanation(
                llm_judge, 
                current_medical_content, 
                explanation, 
                current_audience
            )
            
            if llm_score is None:
                logger.warning(f"Skipping prediction {i}: LLM scoring failed")
                continue
            
            # Store valid scores
            llm_scores.append(llm_score)
            human_ratings.append(human_rating)
            
            logger.debug(f"Prediction {i}: LLM={llm_score:.3f}, Human={human_rating:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing prediction {i}: {e}")
            continue
    
    # Check if we have enough valid data
    if len(llm_scores) < 2:
        raise ValueError(f"Need at least 2 valid predictions for correlation analysis, got {len(llm_scores)}")
    
    logger.info(f"Successfully processed {len(llm_scores)} predictions for validation")
    
    # Calculate correlation coefficient
    try:
        if correlation_type == 'spearman':
            correlation_coef, p_value = spearmanr(llm_scores, human_ratings)
        else:  # pearson
            correlation_coef, p_value = pearsonr(llm_scores, human_ratings)
            
        # Handle NaN results (can occur with constant values)
        if np.isnan(correlation_coef):
            logger.warning("Correlation coefficient is NaN - may indicate constant values")
            correlation_coef = 0.0
            p_value = 1.0
        
        logger.info(f"Validation complete: {correlation_type} correlation = {correlation_coef:.3f}, p-value = {p_value:.3f}")
        
        return correlation_coef, p_value
        
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        raise


def validate_llm_judge_detailed(
    predictions: List[Dict[str, Any]], 
    llm_judge: Any,
    medical_content: Optional[str] = None,
    audience: str = 'patient',
    correlation_type: str = 'spearman'
) -> ValidationResult:
    """Perform detailed validation of LLM-as-a-Judge with comprehensive metrics.
    
    This function provides a more comprehensive validation analysis including
    additional metrics like mean absolute error, score distributions, and
    detailed diagnostic information.
    
    Args:
        predictions: List of prediction dictionaries (same format as validate_llm_judge).
        llm_judge: Instance of the LLM judge class.
        medical_content: Original medical content for context.
        audience: Target audience for scoring.
        correlation_type: Type of correlation to compute.
        
    Returns:
        ValidationResult object containing comprehensive validation metrics.
        
    Example:
        ```python
        result = validate_llm_judge_detailed(predictions, llm_judge)
        print(f"Correlation: {result.correlation_coefficient:.3f}")
        print(f"MAE: {result.mean_absolute_error:.3f}")
        print(f"Sample size: {result.n_samples}")
        ```
    """
    # Get basic correlation results
    correlation_coef, p_value = validate_llm_judge(
        predictions, llm_judge, medical_content, audience, correlation_type
    )
    
    # Extract scores for detailed analysis
    llm_scores = []
    human_ratings = []
    
    for prediction in predictions:
        try:
            if not isinstance(prediction, dict):
                continue
                
            explanation = prediction.get('generated_explanation')
            human_rating = prediction.get('human_rating')
            
            if not explanation or human_rating is None:
                continue
            
            current_medical_content = prediction.get('medical_content', medical_content)
            current_audience = prediction.get('audience', audience)
            
            if not current_medical_content:
                continue
            
            llm_score = _score_explanation(
                llm_judge, current_medical_content, explanation, current_audience
            )
            
            if llm_score is not None:
                llm_scores.append(llm_score)
                human_ratings.append(float(human_rating))
                
        except Exception:
            continue
    
    # Calculate additional metrics
    score_differences = [abs(llm - human) for llm, human in zip(llm_scores, human_ratings)]
    mean_absolute_error = np.mean(score_differences) if score_differences else 0.0
    
    return ValidationResult(
        correlation_coefficient=correlation_coef,
        p_value=p_value,
        n_samples=len(llm_scores),
        llm_scores=llm_scores,
        human_ratings=human_ratings,
        score_differences=score_differences,
        mean_absolute_error=mean_absolute_error,
        correlation_type=correlation_type
    )


def _score_explanation(
    llm_judge: Any, 
    medical_content: str, 
    explanation: str, 
    audience: str
) -> Optional[float]:
    """Score an explanation using the LLM judge.
    
    This helper function abstracts the scoring process to handle different
    LLM judge interfaces and provides consistent error handling.
    
    Args:
        llm_judge: Instance of the LLM judge.
        medical_content: Original medical content.
        explanation: Generated explanation to score.
        audience: Target audience.
        
    Returns:
        Overall score as float, or None if scoring failed.
    """
    try:
        # Try different scoring methods based on available interface
        if hasattr(llm_judge, 'evaluate_explanation'):
            # Single explanation evaluation
            result = llm_judge.evaluate_explanation(medical_content, explanation, audience)
            
            # Extract overall score from result
            if hasattr(result, 'overall'):
                return float(result.overall)
            elif isinstance(result, dict) and 'overall' in result:
                return float(result['overall'])
            elif isinstance(result, (int, float)):
                return float(result)
            else:
                logger.warning("Cannot extract score from evaluation result")
                return None
                
        elif hasattr(llm_judge, 'evaluate_all_audiences'):
            # Multi-audience evaluation
            explanations = {audience: explanation}
            results = llm_judge.evaluate_all_audiences(medical_content, explanations)
            
            if audience in results:
                result = results[audience]
                if hasattr(result, 'overall'):
                    return float(result.overall)
                elif isinstance(result, dict) and 'overall' in result:
                    return float(result['overall'])
            
            logger.warning(f"No score found for audience '{audience}'")
            return None
            
        else:
            logger.error("LLM judge has no recognized scoring method")
            return None
            
    except Exception as e:
        logger.error(f"Error scoring explanation: {e}")
        return None


def save_validation_results(
    result: ValidationResult, 
    output_path: Union[str, Path]
) -> None:
    """Save detailed validation results to a JSON file.
    
    Args:
        result: ValidationResult object to save.
        output_path: Path where results should be saved.
        
    Example:
        ```python
        result = validate_llm_judge_detailed(predictions, llm_judge)
        save_validation_results(result, 'validation_results.json')
        ```
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert result to dictionary
    result_dict = {
        'correlation_coefficient': result.correlation_coefficient,
        'p_value': result.p_value,
        'n_samples': result.n_samples,
        'mean_absolute_error': result.mean_absolute_error,
        'correlation_type': result.correlation_type,
        'llm_scores': result.llm_scores,
        'human_ratings': result.human_ratings,
        'score_differences': result.score_differences,
        'summary_statistics': {
            'llm_scores': {
                'mean': float(np.mean(result.llm_scores)),
                'std': float(np.std(result.llm_scores)),
                'min': float(np.min(result.llm_scores)),
                'max': float(np.max(result.llm_scores))
            },
            'human_ratings': {
                'mean': float(np.mean(result.human_ratings)),
                'std': float(np.std(result.human_ratings)),
                'min': float(np.min(result.human_ratings)),
                'max': float(np.max(result.human_ratings))
            }
        }
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save validation results: {e}")
        raise


def load_validation_data(data_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load validation data from a JSON file.
    
    Args:
        data_path: Path to the validation data file.
        
    Returns:
        List of prediction dictionaries.
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        
    Example:
        ```python
        predictions = load_validation_data('validation_data.json')
        result = validate_llm_judge(predictions, llm_judge)
        ```
    """
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Validation data file not found: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("Validation data must be a list of predictions")
            
        logger.info(f"Loaded {len(data)} validation predictions from {data_file}")
        return data
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in validation data file: {e}",
            e.doc, e.pos
        )


# Additional imports for comprehensive validation
import time


def create_synthetic_test_set() -> List[Dict[str, Any]]:
    """Create a synthetic test set with unambiguous cases for judge validation.
    
    This function implements the first part of the three-part validation strategy:
    Synthetic Agreement Testing. It creates clear, unambiguous test cases where
    the expected quality ratings are obvious, allowing verification that the
    LLM judge applies rubrics correctly.
    
    Returns:
        List of synthetic test cases with known expected ratings.
        
    Example:
        ```python
        synthetic_tests = create_synthetic_test_set()
        result = validate_llm_judge(synthetic_tests, llm_judge)
        ```
    """
    synthetic_cases = [
        # High Quality Examples (Expected rating: 4.5-5.0)
        {
            'medical_content': 'Diabetes is a condition where blood sugar levels are too high.',
            'generated_explanation': (
                'Diabetes means your body has trouble controlling sugar in your blood. '
                'When you eat, your body breaks down food into sugar (glucose) that goes into your bloodstream. '
                'Normally, your body makes insulin to help sugar get into your cells for energy. '
                'With diabetes, either your body doesn\'t make enough insulin or can\'t use it properly. '
                'This causes sugar to build up in your blood instead of going into your cells. '
                'High blood sugar over time can damage your heart, eyes, kidneys, and nerves. '
                'The good news is that diabetes can be managed with healthy eating, exercise, '
                'and sometimes medicine to help keep your blood sugar in a healthy range.'
            ),
            'human_rating': 4.8,
            'audience': 'patient',
            'expected_range': (4.5, 5.0),
            'rationale': 'Clear, accurate, appropriate language for patient, actionable advice'
        },
        
        # Medium Quality Examples (Expected rating: 3.0-3.5)
        {
            'medical_content': 'Hypertension requires lifestyle modifications and possibly medication.',
            'generated_explanation': (
                'High blood pressure is when the force of blood against artery walls is too high. '
                'You need to change your lifestyle and might need medication. '
                'Eat less salt and exercise more.'
            ),
            'human_rating': 3.2,
            'audience': 'patient',
            'expected_range': (2.8, 3.8),
            'rationale': 'Accurate but brief, missing detailed explanations and empathy'
        },
        
        # Low Quality Examples (Expected rating: 1.5-2.5)
        {
            'medical_content': 'Myocardial infarction requires immediate medical attention.',
            'generated_explanation': (
                'MI is bad. Go to doctor. Heart attack maybe. Could die. Very serious medical emergency situation.'
            ),
            'human_rating': 2.0,
            'audience': 'patient',
            'expected_range': (1.5, 2.5),
            'rationale': 'Inaccurate terminology, creates anxiety, poor communication'
        },
        
        # Professional Audience - High Quality (Expected rating: 4.5-5.0)
        {
            'medical_content': 'Patient presents with chest pain and elevated troponins.',
            'generated_explanation': (
                'Elevated troponins in the setting of chest pain suggest myocardial injury. '
                'Differential diagnosis includes STEMI, NSTEMI, unstable angina, myocarditis, '
                'pulmonary embolism, and aortic dissection. Recommend immediate ECG, '
                'serial troponins, chest X-ray, and echocardiogram. Consider cardiology '
                'consultation and initiate dual antiplatelet therapy if ACS suspected, '
                'pending contraindications. Monitor for arrhythmias and hemodynamic instability.'
            ),
            'human_rating': 4.7,
            'audience': 'physician',
            'expected_range': (4.5, 5.0),
            'rationale': 'Comprehensive differential, appropriate clinical decision making'
        },
        
        # Nurse Audience - Medium Quality (Expected rating: 3.0-3.5)
        {
            'medical_content': 'Patient requires wound care and infection monitoring.',
            'generated_explanation': (
                'Monitor the wound daily for signs of infection like redness, swelling, warmth, '
                'or discharge. Change dressings as ordered and document wound characteristics. '
                'Watch vital signs for fever.'
            ),
            'human_rating': 3.3,
            'audience': 'nurse',
            'expected_range': (3.0, 3.8),
            'rationale': 'Good nursing focus but lacks detail on specific interventions'
        },
        
        # Safety Issues - Very Low Quality (Expected rating: 1.0-1.5)
        {
            'medical_content': 'Patient asks about medication dosage.',
            'generated_explanation': (
                'Just take whatever amount feels right. More medicine is usually better. '
                'If it doesn\'t work, double the dose. You don\'t need to talk to your doctor first.'
            ),
            'human_rating': 1.0,
            'audience': 'patient',
            'expected_range': (1.0, 1.5),
            'rationale': 'Dangerous advice, contradicts medical safety principles'
        }
    ]
    
    logger.info(f"Created synthetic test set with {len(synthetic_cases)} cases")
    return synthetic_cases


def run_synthetic_agreement_testing(
    llm_judge: Any,
    tolerance: float = 0.5
) -> Dict[str, Any]:
    """Run synthetic agreement testing to validate LLM judge rubric application.
    
    Args:
        llm_judge: LLM judge instance to validate.
        tolerance: Acceptable deviation from expected rating range.
        
    Returns:
        Dictionary with testing results and analysis.
    """
    synthetic_cases = create_synthetic_test_set()
    results = {
        'total_cases': len(synthetic_cases),
        'passed_cases': 0,
        'failed_cases': 0,
        'case_results': [],
        'overall_accuracy': 0.0,
        'tolerance': tolerance
    }
    
    for i, case in enumerate(synthetic_cases):
        try:
            # Score the synthetic case
            llm_score = _score_explanation(
                llm_judge,
                case['medical_content'],
                case['generated_explanation'],
                case['audience']
            )
            
            if llm_score is None:
                logger.warning(f"Failed to score synthetic case {i}")
                continue
            
            # Check if within expected range
            expected_min, expected_max = case['expected_range']
            in_range = expected_min - tolerance <= llm_score <= expected_max + tolerance
            
            case_result = {
                'case_id': i,
                'llm_score': llm_score,
                'human_rating': case['human_rating'],
                'expected_range': case['expected_range'],
                'in_expected_range': in_range,
                'audience': case['audience'],
                'rationale': case['rationale']
            }
            
            results['case_results'].append(case_result)
            
            if in_range:
                results['passed_cases'] += 1
            else:
                results['failed_cases'] += 1
                logger.warning(
                    f"Synthetic case {i} failed: LLM score {llm_score:.2f} "
                    f"outside expected range {expected_min:.1f}-{expected_max:.1f}"
                )
                
        except Exception as e:
            logger.error(f"Error testing synthetic case {i}: {e}")
            continue
    
    # Calculate overall accuracy
    total_valid = results['passed_cases'] + results['failed_cases']
    if total_valid > 0:
        results['overall_accuracy'] = results['passed_cases'] / total_valid
    
    logger.info(
        f"Synthetic testing complete: {results['passed_cases']}/{total_valid} "
        f"cases passed ({results['overall_accuracy']:.1%} accuracy)"
    )
    
    return results


def calculate_inter_rater_reliability(
    predictions: List[Dict[str, Any]],
    judge_models: List[Any],
    medical_content: Optional[str] = None,
    audience: str = 'patient'
) -> Dict[str, float]:
    """Calculate inter-rater reliability using Krippendorff's Alpha.
    
    This implements the second part of the three-part validation strategy:
    Cross-Model Agreement. It measures agreement between multiple LLM judges
    to assess consistency of automated evaluation.
    
    Args:
        predictions: List of predictions to evaluate.
        judge_models: List of different LLM judge instances.
        medical_content: Medical content for context.
        audience: Target audience.
        
    Returns:
        Dictionary with reliability metrics.
        
    Example:
        ```python
        judges = [gpt4_judge, claude_judge, deepseek_judge]
        reliability = calculate_inter_rater_reliability(predictions, judges)
        ```
    """
    try:
        import krippendorff
    except ImportError:
        logger.warning("krippendorff library not available, using simplified correlation")
        return _calculate_simplified_agreement(predictions, judge_models, medical_content, audience)
    
    logger.info(f"Calculating inter-rater reliability with {len(judge_models)} judges")
    
    # Collect scores from all judges
    all_scores = []  # Each row is one judge, each column is one prediction
    
    for judge_idx, judge in enumerate(judge_models):
        judge_scores = []
        
        for pred_idx, prediction in enumerate(predictions):
            try:
                explanation = prediction.get('generated_explanation', '')
                current_medical_content = prediction.get('medical_content', medical_content)
                current_audience = prediction.get('audience', audience)
                
                if not current_medical_content or not explanation:
                    judge_scores.append(None)  # Missing data
                    continue
                
                score = _score_explanation(judge, current_medical_content, explanation, current_audience)
                judge_scores.append(score)
                
            except Exception as e:
                logger.warning(f"Error scoring prediction {pred_idx} with judge {judge_idx}: {e}")
                judge_scores.append(None)
        
        all_scores.append(judge_scores)
        logger.info(f"Judge {judge_idx} scored {sum(1 for s in judge_scores if s is not None)} predictions")
    
    # Calculate Krippendorff's Alpha
    try:
        # Convert to format expected by krippendorff library
        reliability_data = np.array(all_scores, dtype=float)
        
        # Calculate alpha for interval data (continuous scores)
        alpha = krippendorff.alpha(reliability_data, level_of_measurement='interval')
        
        # Also calculate pairwise correlations between judges
        pairwise_correlations = []
        for i in range(len(all_scores)):
            for j in range(i + 1, len(all_scores)):
                scores1 = [s for s in all_scores[i] if s is not None]
                scores2 = [s for s in all_scores[j] if s is not None]
                
                if len(scores1) >= 2 and len(scores2) >= 2:
                    # Align scores (only use predictions where both judges provided scores)
                    aligned1, aligned2 = [], []
                    for idx in range(min(len(all_scores[i]), len(all_scores[j]))):
                        if all_scores[i][idx] is not None and all_scores[j][idx] is not None:
                            aligned1.append(all_scores[i][idx])
                            aligned2.append(all_scores[j][idx])
                    
                    if len(aligned1) >= 2:
                        correlation, _ = spearmanr(aligned1, aligned2)
                        if not np.isnan(correlation):
                            pairwise_correlations.append(correlation)
        
        avg_correlation = np.mean(pairwise_correlations) if pairwise_correlations else 0.0
        
        return {
            'krippendorff_alpha': float(alpha),
            'average_pairwise_correlation': float(avg_correlation),
            'num_judges': len(judge_models),
            'num_predictions': len(predictions),
            'pairwise_correlations': [float(c) for c in pairwise_correlations]
        }
        
    except Exception as e:
        logger.error(f"Error calculating Krippendorff's Alpha: {e}")
        return _calculate_simplified_agreement(predictions, judge_models, medical_content, audience)


def _calculate_simplified_agreement(
    predictions: List[Dict[str, Any]],
    judge_models: List[Any],
    medical_content: Optional[str] = None,
    audience: str = 'patient'
) -> Dict[str, float]:
    """Simplified agreement calculation using pairwise correlations."""
    all_scores = []
    
    for judge in judge_models:
        judge_scores = []
        for prediction in predictions:
            try:
                explanation = prediction.get('generated_explanation', '')
                current_medical_content = prediction.get('medical_content', medical_content)
                current_audience = prediction.get('audience', audience)
                
                if current_medical_content and explanation:
                    score = _score_explanation(judge, current_medical_content, explanation, current_audience)
                    judge_scores.append(score if score is not None else np.nan)
                else:
                    judge_scores.append(np.nan)
            except Exception:
                judge_scores.append(np.nan)
        
        all_scores.append(judge_scores)
    
    # Calculate pairwise correlations
    correlations = []
    for i in range(len(all_scores)):
        for j in range(i + 1, len(all_scores)):
            valid_pairs = [(s1, s2) for s1, s2 in zip(all_scores[i], all_scores[j]) 
                          if not (np.isnan(s1) or np.isnan(s2))]
            
            if len(valid_pairs) >= 2:
                scores1, scores2 = zip(*valid_pairs)
                correlation, _ = spearmanr(scores1, scores2)
                if not np.isnan(correlation):
                    correlations.append(correlation)
    
    avg_correlation = np.mean(correlations) if correlations else 0.0
    
    return {
        'krippendorff_alpha': np.nan,  # Not calculated
        'average_pairwise_correlation': float(avg_correlation),
        'num_judges': len(judge_models),
        'num_predictions': len(predictions),
        'pairwise_correlations': [float(c) for c in correlations]
    }


def run_correlation_analysis(
    predictions: List[Dict[str, Any]],
    llm_judge: Any,
    quality_indicators: Optional[Dict[str, Any]] = None,
    medical_content: Optional[str] = None,
    audience: str = 'patient'
) -> Dict[str, Any]:
    """Run correlation analysis between automated scores and quality indicators.
    
    This implements the third part of the three-part validation strategy:
    Correlation Analysis. It correlates automated scores with quality indicators
    from source datasets (e.g., consumer explanations from MedQuAD).
    
    Args:
        predictions: List of predictions with quality indicators.
        llm_judge: LLM judge instance.
        quality_indicators: Additional quality metrics from source datasets.
        medical_content: Medical content for context.
        audience: Target audience.
        
    Returns:
        Dictionary with correlation analysis results.
        
    Example:
        ```python
        # Predictions with source quality indicators
        predictions = [
            {
                'generated_explanation': 'explanation text',
                'source_quality_score': 4.2,  # From original dataset
                'readability_score': 8.5,     # Flesch-Kincaid score
                'expert_rating': 4.0          # Expert evaluation
            }
        ]
        
        correlation_results = run_correlation_analysis(predictions, llm_judge)
        ```
    """
    logger.info("Running correlation analysis with quality indicators")
    
    llm_scores = []
    source_quality_scores = []
    readability_scores = []
    expert_ratings = []
    
    for i, prediction in enumerate(predictions):
        try:
            # Get LLM score
            explanation = prediction.get('generated_explanation', '')
            current_medical_content = prediction.get('medical_content', medical_content)
            current_audience = prediction.get('audience', audience)
            
            if not current_medical_content or not explanation:
                continue
            
            llm_score = _score_explanation(llm_judge, current_medical_content, explanation, current_audience)
            if llm_score is None:
                continue
            
            llm_scores.append(llm_score)
            
            # Collect quality indicators
            source_quality_scores.append(prediction.get('source_quality_score', np.nan))
            readability_scores.append(prediction.get('readability_score', np.nan))
            expert_ratings.append(prediction.get('expert_rating', np.nan))
            
        except Exception as e:
            logger.warning(f"Error processing prediction {i} for correlation analysis: {e}")
            continue
    
    if len(llm_scores) < 2:
        logger.warning("Insufficient valid scores for correlation analysis")
        return {'error': 'Insufficient data for correlation analysis'}
    
    results = {
        'num_samples': len(llm_scores),
        'correlations': {},
        'llm_score_stats': {
            'mean': float(np.mean(llm_scores)),
            'std': float(np.std(llm_scores)),
            'min': float(np.min(llm_scores)),
            'max': float(np.max(llm_scores))
        }
    }
    
    # Calculate correlations with each quality indicator
    quality_metrics = {
        'source_quality_score': source_quality_scores,
        'readability_score': readability_scores,
        'expert_rating': expert_ratings
    }
    
    for metric_name, metric_scores in quality_metrics.items():
        # Filter out NaN values
        valid_pairs = [(llm, metric) for llm, metric in zip(llm_scores, metric_scores) 
                      if not np.isnan(metric)]
        
        if len(valid_pairs) >= 2:
            llm_valid, metric_valid = zip(*valid_pairs)
            
            # Calculate both Spearman and Pearson correlations
            spearman_corr, spearman_p = spearmanr(llm_valid, metric_valid)
            pearson_corr, pearson_p = pearsonr(llm_valid, metric_valid)
            
            results['correlations'][metric_name] = {
                'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else None,
                'spearman_p_value': float(spearman_p) if not np.isnan(spearman_p) else None,
                'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else None,
                'pearson_p_value': float(pearson_p) if not np.isnan(pearson_p) else None,
                'n_samples': len(valid_pairs),
                'metric_stats': {
                    'mean': float(np.mean(metric_valid)),
                    'std': float(np.std(metric_valid)),
                    'min': float(np.min(metric_valid)),
                    'max': float(np.max(metric_valid))
                }
            }
        else:
            results['correlations'][metric_name] = {
                'error': f'Insufficient data (n={len(valid_pairs)})'
            }
    
    # Add overall quality assessment
    valid_correlations = [corr['spearman_correlation'] for corr in results['correlations'].values() 
                         if isinstance(corr, dict) and corr.get('spearman_correlation') is not None]
    
    if valid_correlations:
        results['overall_correlation_strength'] = float(np.mean(valid_correlations))
        results['correlation_consistency'] = float(np.std(valid_correlations))
    
    logger.info(f"Correlation analysis complete with {results['num_samples']} samples")
    return results


def run_comprehensive_validation(
    predictions: List[Dict[str, Any]],
    llm_judge: Any,
    additional_judges: Optional[List[Any]] = None,
    medical_content: Optional[str] = None,
    audience: str = 'patient',
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Run the complete three-part validation strategy.
    
    Combines synthetic agreement testing, cross-model agreement, and correlation
    analysis into a comprehensive validation report.
    
    Args:
        predictions: Validation predictions.
        llm_judge: Primary LLM judge to validate.
        additional_judges: Additional judges for inter-rater reliability.
        medical_content: Medical content for context.
        audience: Target audience.
        output_path: Optional path to save detailed results.
        
    Returns:
        Comprehensive validation results dictionary.
    """
    logger.info("Starting comprehensive three-part validation")
    
    comprehensive_results = {
        'validation_timestamp': time.time(),
        'validation_components': {
            'synthetic_agreement': None,
            'inter_rater_reliability': None,
            'correlation_analysis': None
        },
        'overall_assessment': {},
        'recommendations': []
    }
    
    # Part 1: Synthetic Agreement Testing
    try:
        logger.info("Running synthetic agreement testing...")
        synthetic_results = run_synthetic_agreement_testing(llm_judge)
        comprehensive_results['validation_components']['synthetic_agreement'] = synthetic_results
        
        if synthetic_results['overall_accuracy'] >= 0.8:
            comprehensive_results['recommendations'].append("✓ Synthetic testing passed - judge applies rubrics correctly")
        else:
            comprehensive_results['recommendations'].append("⚠ Synthetic testing concerns - judge may misapply rubrics")
            
    except Exception as e:
        logger.error(f"Synthetic agreement testing failed: {e}")
        comprehensive_results['validation_components']['synthetic_agreement'] = {'error': str(e)}
    
    # Part 2: Inter-Rater Reliability
    if additional_judges:
        try:
            logger.info("Running inter-rater reliability analysis...")
            all_judges = [llm_judge] + additional_judges
            reliability_results = calculate_inter_rater_reliability(predictions, all_judges, medical_content, audience)
            comprehensive_results['validation_components']['inter_rater_reliability'] = reliability_results
            
            avg_correlation = reliability_results.get('average_pairwise_correlation', 0)
            if avg_correlation >= 0.7:
                comprehensive_results['recommendations'].append("✓ High inter-rater reliability - consistent scoring")
            elif avg_correlation >= 0.5:
                comprehensive_results['recommendations'].append("~ Moderate inter-rater reliability - acceptable consistency")
            else:
                comprehensive_results['recommendations'].append("⚠ Low inter-rater reliability - inconsistent scoring")
                
        except Exception as e:
            logger.error(f"Inter-rater reliability analysis failed: {e}")
            comprehensive_results['validation_components']['inter_rater_reliability'] = {'error': str(e)}
    else:
        comprehensive_results['validation_components']['inter_rater_reliability'] = {'skipped': 'No additional judges provided'}
    
    # Part 3: Correlation Analysis
    try:
        logger.info("Running correlation analysis...")
        correlation_results = run_correlation_analysis(predictions, llm_judge, None, medical_content, audience)
        comprehensive_results['validation_components']['correlation_analysis'] = correlation_results
        
        overall_correlation = correlation_results.get('overall_correlation_strength')
        if overall_correlation and overall_correlation >= 0.6:
            comprehensive_results['recommendations'].append("✓ Good correlation with quality indicators")
        elif overall_correlation and overall_correlation >= 0.4:
            comprehensive_results['recommendations'].append("~ Moderate correlation with quality indicators")
        else:
            comprehensive_results['recommendations'].append("⚠ Weak correlation with quality indicators")
            
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        comprehensive_results['validation_components']['correlation_analysis'] = {'error': str(e)}
    
    # Overall Assessment
    success_count = sum(1 for component in comprehensive_results['validation_components'].values() 
                       if isinstance(component, dict) and 'error' not in component and 'skipped' not in component)
    
    comprehensive_results['overall_assessment'] = {
        'validation_score': success_count / 3.0,
        'components_completed': success_count,
        'total_components': 3,
        'overall_recommendation': 'PASS' if success_count >= 2 else 'REVIEW_NEEDED'
    }
    
    # Save results if path provided
    if output_path:
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Comprehensive validation results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
    
    logger.info("Comprehensive validation complete")
    return comprehensive_results