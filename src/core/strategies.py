"""Strategy pattern implementation for audience-specific scoring.

This module implements the Strategy pattern to handle audience-specific scoring
logic for medical explanation evaluation. Each audience (physician, nurse, patient,
caregiver) has different expectations for readability, terminology, and explanation
length, which are encapsulated in separate strategy classes.

The module provides:
    - Abstract base class for audience strategies
    - Concrete strategy implementations for each audience type
    - Factory pattern for creating appropriate strategies
    - Audience-specific scoring algorithms for readability and terminology

Example:
    ```python
    from strategies import StrategyFactory
    
    # Create strategy for specific audience
    strategy = StrategyFactory.create_strategy('patient')
    
    # Calculate audience-specific scores
    readability_score = strategy.calculate_readability_score(text, grade_level)
    terminology_score = strategy.calculate_terminology_score(text, term_density)
    ```
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type
import logging

from .config import config

logger = logging.getLogger(__name__)


class AudienceStrategy(ABC):
    """Abstract base class for audience-specific scoring strategies.

    This abstract class defines the interface for audience-specific scoring
    strategies used in MedExplain-Evals evaluation. Each concrete strategy implements
    scoring logic tailored to the expectations and needs of a specific healthcare
    audience.

    Attributes:
        audience: The target audience name (e.g., 'physician', 'nurse').
        eval_config: Evaluation configuration loaded from the config system.

    Abstract Methods:
        calculate_readability_score: Calculate readability score for the audience.
        calculate_terminology_score: Calculate terminology appropriateness score.
        get_expected_explanation_length: Get expected explanation length range.
    """

    def __init__(self, audience: str) -> None:
        """Initialize audience strategy.

        Args:
            audience: Target audience name (e.g., 'physician', 'nurse', 'patient', 'caregiver').
        """
        self.audience: str = audience
        self.eval_config: Dict[str, Any] = config.get_evaluation_config()

    @abstractmethod
    def calculate_readability_score(self, text: str, grade_level: float) -> float:
        """Calculate readability score for the audience.

        Args:
            text: Text to evaluate for readability.
            grade_level: Computed grade level (e.g., Flesch-Kincaid score).

        Returns:
            Readability score between 0.0 and 1.0, where 1.0 indicates
            optimal readability for the target audience.
        """
        pass

    @abstractmethod
    def calculate_terminology_score(self, text: str, term_density: float) -> float:
        """Calculate terminology appropriateness score for the audience.

        Args:
            text: Text to evaluate for terminology usage.
            term_density: Ratio of medical terms to total words in the text.

        Returns:
            Terminology score between 0.0 and 1.0, where 1.0 indicates
            optimal terminology usage for the target audience.
        """
        pass

    @abstractmethod
    def get_expected_explanation_length(self) -> Dict[str, int]:
        """Get expected explanation length range for the audience.

        Returns:
            Dictionary with 'min' and 'max' keys indicating the expected
            word count range for explanations targeting this audience.
        """
        pass

    def get_readability_targets(self) -> Dict[str, float]:
        """Get readability targets for this audience.

        Returns:
            Dictionary containing readability targets including minimum and
            maximum grade levels appropriate for this audience.
        """
        return self.eval_config["readability_targets"][self.audience]

    def get_terminology_targets(self) -> Dict[str, float]:
        """Get terminology density targets for this audience.

        Returns:
            Dictionary containing terminology density targets including
            target density and acceptable tolerance range.
        """
        return self.eval_config["terminology_density"][self.audience]


class PhysicianStrategy(AudienceStrategy):
    """Strategy for physician audience scoring.

    Physicians expect technical, evidence-based explanations with precise medical
    terminology. This strategy scores explanations based on graduate-level complexity
    (12-16 grade level) and high medical terminology density.

    Scoring characteristics:
        - Readability: Favors higher complexity (graduate level)
        - Terminology: Expects high medical term density
        - Length: Accommodates longer, detailed explanations
    """

    def __init__(self) -> None:
        super().__init__("physician")

    def calculate_readability_score(self, text: str, grade_level: float) -> float:
        """Calculate readability score for physician audience.

        Physicians expect graduate-level complexity (12-16 grade level).
        Higher complexity is generally better for physicians, but extremely
        high complexity (>16) may still be penalized.

        Args:
            text: Text to evaluate for readability.
            grade_level: Computed grade level (e.g., Flesch-Kincaid score).

        Returns:
            Readability score between 0.0 and 1.0, with 1.0 for optimal complexity.
        """
        targets = self.get_readability_targets()
        min_level = targets["min_grade_level"]
        max_level = targets["max_grade_level"]

        if grade_level < min_level:
            # Too simple for physicians
            return max(0.0, grade_level / min_level)
        elif grade_level > max_level:
            # Too complex even for physicians
            return max(0.0, 1.0 - (grade_level - max_level) / 4.0)
        else:
            # In the sweet spot
            return 1.0

    def calculate_terminology_score(self, text: str, term_density: float) -> float:
        """Calculate terminology score for physician audience.

        Physicians expect high medical terminology density as they are
        comfortable with technical medical language and precise terminology.

        Args:
            text: Text to evaluate for terminology usage.
            term_density: Ratio of medical terms to total words in the text.

        Returns:
            Terminology score between 0.0 and 1.0, with 1.0 for optimal density.
        """
        targets = self.get_terminology_targets()
        target = targets["target"]
        tolerance = targets["tolerance"]

        if abs(term_density - target) <= tolerance:
            return 1.0
        elif term_density < target:
            # Too little medical terminology
            return max(0.0, term_density / target)
        else:
            # Too much terminology (even for physicians)
            excess = term_density - target - tolerance
            return max(0.0, 1.0 - excess * 2.0)

    def get_expected_explanation_length(self) -> Dict[str, int]:
        """Get expected explanation length for physician audience.

        Physicians can handle longer, detailed explanations with comprehensive
        medical information and technical details.

        Returns:
            Dictionary with 'min' and 'max' keys for expected word count range.
        """
        scoring_config = config.get_scoring_config()
        max_length = scoring_config["parameters"]["max_explanation_length"]["physician"]
        min_length = scoring_config["parameters"]["min_explanation_length"]

        return {"min": min_length, "max": max_length}


class NurseStrategy(AudienceStrategy):
    """Strategy for nurse audience scoring.

    Nurses expect moderate complexity explanations that balance technical accuracy
    with practical application. They need information that supports patient care
    and education responsibilities.

    Scoring characteristics:
        - Readability: Moderate complexity (10-14 grade level)
        - Terminology: Balanced medical terminology with practical language
        - Length: Practical, actionable explanations
    """

    def __init__(self) -> None:
        super().__init__("nurse")

    def calculate_readability_score(self, text: str, grade_level: float) -> float:
        """Calculate readability score for nurse audience.

        Nurses expect moderate complexity (10-14 grade level) that balances
        technical accuracy with practical application in patient care settings.

        Args:
            text: Text to evaluate for readability.
            grade_level: Computed grade level (e.g., Flesch-Kincaid score).

        Returns:
            Readability score between 0.0 and 1.0, with 1.0 for optimal complexity.
        """
        targets = self.get_readability_targets()
        min_level = targets["min_grade_level"]
        max_level = targets["max_grade_level"]

        if min_level <= grade_level <= max_level:
            return 1.0
        elif grade_level < min_level:
            return max(0.0, grade_level / min_level)
        else:
            return max(0.0, 1.0 - (grade_level - max_level) / 6.0)

    def calculate_terminology_score(self, text: str, term_density: float) -> float:
        """Calculate terminology score for nurse audience.

        Nurses expect moderate medical terminology that includes technical terms
        but also incorporates practical language for patient care contexts.

        Args:
            text: Text to evaluate for terminology usage.
            term_density: Ratio of medical terms to total words in the text.

        Returns:
            Terminology score between 0.0 and 1.0, with 1.0 for optimal density.
        """
        targets = self.get_terminology_targets()
        target = targets["target"]
        tolerance = targets["tolerance"]

        if abs(term_density - target) <= tolerance:
            return 1.0
        else:
            deviation = abs(term_density - target) - tolerance
            return max(0.0, 1.0 - deviation * 3.0)

    def get_expected_explanation_length(self) -> Dict[str, int]:
        """Get expected explanation length for nurse audience.

        Nurses need practical, actionable explanations that provide clear
        guidance for patient care and education.

        Returns:
            Dictionary with 'min' and 'max' keys for expected word count range.
        """
        scoring_config = config.get_scoring_config()
        max_length = scoring_config["parameters"]["max_explanation_length"]["nurse"]
        min_length = scoring_config["parameters"]["min_explanation_length"]

        return {"min": min_length, "max": max_length}


class PatientStrategy(AudienceStrategy):
    """Strategy for patient audience scoring.

    Patients need simple, accessible explanations that avoid medical jargon
    and focus on understanding their condition and next steps. Explanations
    should be empathetic and reassuring.

    Scoring characteristics:
        - Readability: Simple language (6-10 grade level)
        - Terminology: Minimal medical terminology, jargon explained
        - Length: Concise, clear explanations
    """

    def __init__(self) -> None:
        super().__init__("patient")

    def calculate_readability_score(self, text: str, grade_level: float) -> float:
        """Calculate readability score for patient audience.

        Patients need simple, accessible language (6-10 grade level).
        Lower complexity is generally better for patients to ensure
        understanding and engagement.

        Args:
            text: Text to evaluate for readability.
            grade_level: Computed grade level (e.g., Flesch-Kincaid score).

        Returns:
            Readability score between 0.0 and 1.0, with 1.0 for optimal simplicity.
        """
        targets = self.get_readability_targets()
        min_level = targets["min_grade_level"]
        max_level = targets["max_grade_level"]

        if min_level <= grade_level <= max_level:
            return 1.0
        elif grade_level < min_level:
            # Very simple is still okay for patients
            return 0.8
        else:
            # Too complex for patients
            return max(0.0, 1.0 - (grade_level - max_level) / 4.0)

    def calculate_terminology_score(self, text: str, term_density: float) -> float:
        """Calculate terminology score for patient audience.

        Patients should have minimal medical terminology in their explanations.
        Medical jargon should be avoided or clearly explained in simple terms.

        Args:
            text: Text to evaluate for terminology usage.
            term_density: Ratio of medical terms to total words in the text.

        Returns:
            Terminology score between 0.0 and 1.0, with 1.0 for minimal jargon.
        """
        targets = self.get_terminology_targets()
        target = targets["target"]
        tolerance = targets["tolerance"]

        if term_density <= target + tolerance:
            return 1.0
        else:
            # Penalty for too much medical terminology
            excess = term_density - target - tolerance
            return max(0.0, 1.0 - excess * 10.0)

    def get_expected_explanation_length(self) -> Dict[str, int]:
        """Get expected explanation length for patient audience.

        Patients need concise, clear explanations that don't overwhelm
        with too much information at once.

        Returns:
            Dictionary with 'min' and 'max' keys for expected word count range.
        """
        scoring_config = config.get_scoring_config()
        max_length = scoring_config["parameters"]["max_explanation_length"]["patient"]
        min_length = scoring_config["parameters"]["min_explanation_length"]

        return {"min": min_length, "max": max_length}


class CaregiverStrategy(AudienceStrategy):
    """Strategy for caregiver audience scoring.

    Caregivers need actionable, practical explanations that focus on
    observable symptoms, clear instructions, and when to seek help.
    They need simple language but with specific guidance.

    Scoring characteristics:
        - Readability: Clear, actionable language (6-10 grade level)
        - Terminology: Minimal medical terminology, focus on observable signs
        - Length: Practical, step-by-step guidance
    """

    def __init__(self) -> None:
        super().__init__("caregiver")

    def calculate_readability_score(self, text: str, grade_level: float) -> float:
        """Calculate readability score for caregiver audience.

        Caregivers need actionable, clear language (6-10 grade level)
        with a focus on practical instructions and observable guidance.

        Args:
            text: Text to evaluate for readability.
            grade_level: Computed grade level (e.g., Flesch-Kincaid score).

        Returns:
            Readability score between 0.0 and 1.0, with 1.0 for optimal clarity.
        """
        targets = self.get_readability_targets()
        min_level = targets["min_grade_level"]
        max_level = targets["max_grade_level"]

        if min_level <= grade_level <= max_level:
            return 1.0
        elif grade_level < min_level:
            return 0.9  # Simple is good for caregivers
        else:
            return max(0.0, 1.0 - (grade_level - max_level) / 4.0)

    def calculate_terminology_score(self, text: str, term_density: float) -> float:
        """Calculate terminology score for caregiver audience.

        Caregivers need minimal medical terminology with focus on observable
        symptoms and clear actions they can take or monitor.

        Args:
            text: Text to evaluate for terminology usage.
            term_density: Ratio of medical terms to total words in the text.

        Returns:
            Terminology score between 0.0 and 1.0, with 1.0 for optimal practical language.
        """
        targets = self.get_terminology_targets()
        target = targets["target"]
        tolerance = targets["tolerance"]

        if term_density <= target + tolerance:
            return 1.0
        else:
            excess = term_density - target - tolerance
            return max(0.0, 1.0 - excess * 8.0)

    def get_expected_explanation_length(self) -> Dict[str, int]:
        """Get expected explanation length for caregiver audience.

        Caregivers need practical, step-by-step guidance that provides
        clear instructions and actionable information.

        Returns:
            Dictionary with 'min' and 'max' keys for expected word count range.
        """
        scoring_config = config.get_scoring_config()
        max_length = scoring_config["parameters"]["max_explanation_length"]["caregiver"]
        min_length = scoring_config["parameters"]["min_explanation_length"]

        return {"min": min_length, "max": max_length}


class StrategyFactory:
    """Factory for creating audience strategies.

    This factory class implements the Factory pattern to create appropriate
    audience strategy instances based on the target audience. It provides
    a centralized way to instantiate strategies and maintains a registry
    of supported audiences.

    Attributes:
        _strategies: Dictionary mapping audience names to strategy classes.
    """

    _strategies: Dict[str, Type[AudienceStrategy]] = {
        "physician": PhysicianStrategy,
        "nurse": NurseStrategy,
        "patient": PatientStrategy,
        "caregiver": CaregiverStrategy,
    }

    @classmethod
    def create_strategy(cls, audience: str) -> AudienceStrategy:
        """Create strategy for given audience.

        Instantiates the appropriate strategy class for the specified audience.
        The strategy encapsulates audience-specific scoring logic and expectations.

        Args:
            audience: Target audience name (e.g., 'physician', 'nurse', 'patient', 'caregiver').

        Returns:
            AudienceStrategy instance configured for the specified audience.

        Raises:
            ValueError: If the audience is not supported. Use get_supported_audiences()
                to see available options.

        Example:
            ```python
            strategy = StrategyFactory.create_strategy('patient')
            readability_score = strategy.calculate_readability_score(text, grade_level)
            ```
        """
        if audience not in cls._strategies:
            supported = list(cls._strategies.keys())
            raise ValueError(f"Unsupported audience: {audience}. Supported: {supported}")

        strategy_class = cls._strategies[audience]
        return strategy_class(audience)  # type: ignore[misc]

    @classmethod
    def get_supported_audiences(cls) -> List[str]:
        """Get list of supported audiences.

        Returns:
            List of audience names that have corresponding strategy implementations.
        """
        return list(cls._strategies.keys())
