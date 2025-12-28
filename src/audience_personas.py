"""Sophisticated audience persona system for MEQ-Bench 2.0.

This module provides a comprehensive audience modeling framework that goes beyond
simple readability metrics. It includes health literacy levels, cultural contexts,
age-appropriate language detection, and persona-based evaluation scoring.

Key Features:
    - 8 distinct audience personas (4 audience types x 2 literacy levels)
    - Health literacy assessment using validated instruments
    - Cultural sensitivity evaluation
    - Age-appropriate language detection
    - Dynamic persona-based scoring

Example:
    ```python
    from audience_personas import PersonaFactory, HealthLiteracyAssessor
    
    # Create persona for a patient with low health literacy
    persona = PersonaFactory.create_persona(
        audience_type="patient",
        health_literacy="low",
        age_group="elderly"
    )
    
    # Evaluate explanation for this persona
    scorer = PersonaBasedScorer(persona)
    score = scorer.evaluate(explanation_text)
    ```
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import re
import logging

logger = logging.getLogger("meq_bench.audience_personas")


class AudienceType(str, Enum):
    """Primary audience types for medical explanations."""
    PHYSICIAN = "physician"
    NURSE = "nurse"
    PATIENT = "patient"
    CAREGIVER = "caregiver"
    PHARMACIST = "pharmacist"
    MEDICAL_STUDENT = "medical_student"


class HealthLiteracy(str, Enum):
    """Health literacy levels based on validated assessment frameworks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DetailLevel(str, Enum):
    """Preferred level of detail in explanations."""
    BRIEF = "brief"
    MODERATE = "moderate"
    COMPREHENSIVE = "comprehensive"


class AgeGroup(str, Enum):
    """Age group categories affecting communication style."""
    PEDIATRIC_PARENT = "pediatric_parent"  # Parent of young child
    ADOLESCENT = "adolescent"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    ELDERLY = "elderly"


@dataclass
class TerminologyExpectations:
    """Expectations for medical terminology usage."""
    technical_terms_density: float  # 0-1, expected density of technical terms
    abbreviations_allowed: bool
    latin_terms_allowed: bool
    dosage_format: str  # "clinical" (mg/kg/day) or "simple" (number of pills)
    anatomy_detail_level: str  # "basic", "moderate", "detailed"
    
    @classmethod
    def for_physician(cls) -> "TerminologyExpectations":
        return cls(
            technical_terms_density=0.15,
            abbreviations_allowed=True,
            latin_terms_allowed=True,
            dosage_format="clinical",
            anatomy_detail_level="detailed",
        )
    
    @classmethod
    def for_nurse(cls) -> "TerminologyExpectations":
        return cls(
            technical_terms_density=0.10,
            abbreviations_allowed=True,
            latin_terms_allowed=False,
            dosage_format="clinical",
            anatomy_detail_level="moderate",
        )
    
    @classmethod
    def for_patient_low_literacy(cls) -> "TerminologyExpectations":
        return cls(
            technical_terms_density=0.02,
            abbreviations_allowed=False,
            latin_terms_allowed=False,
            dosage_format="simple",
            anatomy_detail_level="basic",
        )
    
    @classmethod
    def for_patient_high_literacy(cls) -> "TerminologyExpectations":
        return cls(
            technical_terms_density=0.05,
            abbreviations_allowed=False,
            latin_terms_allowed=False,
            dosage_format="simple",
            anatomy_detail_level="moderate",
        )
    
    @classmethod
    def for_caregiver(cls) -> "TerminologyExpectations":
        return cls(
            technical_terms_density=0.03,
            abbreviations_allowed=False,
            latin_terms_allowed=False,
            dosage_format="simple",
            anatomy_detail_level="basic",
        )


@dataclass
class CommunicationPreferences:
    """Communication style preferences for an audience."""
    empathy_level: str  # "clinical", "moderate", "high"
    action_orientation: str  # "informational", "actionable", "directive"
    visual_aids_preference: bool
    numerical_data_preference: str  # "avoid", "simplified", "detailed"
    uncertainty_communication: str  # "explicit", "simplified", "avoid"
    question_encouragement: bool
    
    @classmethod
    def for_physician(cls) -> "CommunicationPreferences":
        return cls(
            empathy_level="clinical",
            action_orientation="actionable",
            visual_aids_preference=True,
            numerical_data_preference="detailed",
            uncertainty_communication="explicit",
            question_encouragement=False,
        )
    
    @classmethod
    def for_patient(cls) -> "CommunicationPreferences":
        return cls(
            empathy_level="high",
            action_orientation="actionable",
            visual_aids_preference=True,
            numerical_data_preference="simplified",
            uncertainty_communication="simplified",
            question_encouragement=True,
        )
    
    @classmethod
    def for_caregiver(cls) -> "CommunicationPreferences":
        return cls(
            empathy_level="high",
            action_orientation="directive",
            visual_aids_preference=True,
            numerical_data_preference="simplified",
            uncertainty_communication="simplified",
            question_encouragement=True,
        )


@dataclass
class AudiencePersona:
    """Complete audience persona for evaluation.
    
    This dataclass encapsulates all relevant characteristics of a target
    audience, enabling sophisticated, persona-based evaluation of medical
    explanations.
    
    Attributes:
        audience_type: Primary audience category
        health_literacy: Health literacy level
        medical_familiarity: Level of medical knowledge
        age_group: Target age group
        cultural_context: Optional cultural considerations
        preferred_detail_level: Desired explanation detail
        reading_level_target: Target Flesch-Kincaid grade range
        terminology: Expected terminology usage
        communication: Communication style preferences
        persona_id: Unique identifier for this persona
    """
    
    # Core attributes
    audience_type: str
    health_literacy: str
    medical_familiarity: str  # "novice", "some_experience", "expert"
    
    # Demographic context
    age_group: Optional[str] = None
    cultural_context: Optional[str] = None
    language_preference: str = "en"
    
    # Content preferences
    preferred_detail_level: str = DetailLevel.MODERATE.value
    reading_level_target: Tuple[int, int] = (6, 10)  # Grade level range
    max_explanation_length: int = 500  # Words
    
    # Terminology and communication
    terminology: Optional[TerminologyExpectations] = None
    communication: Optional[CommunicationPreferences] = None
    
    # Identification
    persona_id: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default terminology and communication preferences."""
        if self.terminology is None:
            self.terminology = self._default_terminology()
        if self.communication is None:
            self.communication = self._default_communication()
        if self.persona_id is None:
            self.persona_id = f"{self.audience_type}_{self.health_literacy}"
    
    def _default_terminology(self) -> TerminologyExpectations:
        """Get default terminology expectations based on audience type."""
        if self.audience_type == AudienceType.PHYSICIAN.value:
            return TerminologyExpectations.for_physician()
        elif self.audience_type == AudienceType.NURSE.value:
            return TerminologyExpectations.for_nurse()
        elif self.audience_type == AudienceType.PATIENT.value:
            if self.health_literacy == HealthLiteracy.LOW.value:
                return TerminologyExpectations.for_patient_low_literacy()
            else:
                return TerminologyExpectations.for_patient_high_literacy()
        else:
            return TerminologyExpectations.for_caregiver()
    
    def _default_communication(self) -> CommunicationPreferences:
        """Get default communication preferences based on audience type."""
        if self.audience_type == AudienceType.PHYSICIAN.value:
            return CommunicationPreferences.for_physician()
        elif self.audience_type == AudienceType.PATIENT.value:
            return CommunicationPreferences.for_patient()
        else:
            return CommunicationPreferences.for_caregiver()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "persona_id": self.persona_id,
            "audience_type": self.audience_type,
            "health_literacy": self.health_literacy,
            "medical_familiarity": self.medical_familiarity,
            "age_group": self.age_group,
            "cultural_context": self.cultural_context,
            "language_preference": self.language_preference,
            "preferred_detail_level": self.preferred_detail_level,
            "reading_level_target": self.reading_level_target,
            "max_explanation_length": self.max_explanation_length,
            "description": self.description,
        }


class PersonaFactory:
    """Factory for creating audience personas."""
    
    # Predefined personas covering key use cases
    PREDEFINED_PERSONAS = {
        # Physician personas
        "physician_specialist": {
            "audience_type": AudienceType.PHYSICIAN.value,
            "health_literacy": HealthLiteracy.HIGH.value,
            "medical_familiarity": "expert",
            "preferred_detail_level": DetailLevel.COMPREHENSIVE.value,
            "reading_level_target": (14, 18),
            "max_explanation_length": 1000,
            "description": "Medical specialist expecting comprehensive technical explanations",
        },
        "physician_generalist": {
            "audience_type": AudienceType.PHYSICIAN.value,
            "health_literacy": HealthLiteracy.HIGH.value,
            "medical_familiarity": "expert",
            "preferred_detail_level": DetailLevel.MODERATE.value,
            "reading_level_target": (12, 16),
            "max_explanation_length": 800,
            "description": "General practitioner needing practical clinical guidance",
        },
        
        # Nurse personas
        "nurse_icu": {
            "audience_type": AudienceType.NURSE.value,
            "health_literacy": HealthLiteracy.HIGH.value,
            "medical_familiarity": "expert",
            "preferred_detail_level": DetailLevel.COMPREHENSIVE.value,
            "reading_level_target": (12, 14),
            "max_explanation_length": 700,
            "description": "ICU nurse needing detailed monitoring parameters",
        },
        "nurse_general": {
            "audience_type": AudienceType.NURSE.value,
            "health_literacy": HealthLiteracy.MEDIUM.value,
            "medical_familiarity": "some_experience",
            "preferred_detail_level": DetailLevel.MODERATE.value,
            "reading_level_target": (10, 12),
            "max_explanation_length": 600,
            "description": "General ward nurse focusing on practical care instructions",
        },
        
        # Patient personas
        "patient_low_literacy": {
            "audience_type": AudienceType.PATIENT.value,
            "health_literacy": HealthLiteracy.LOW.value,
            "medical_familiarity": "novice",
            "preferred_detail_level": DetailLevel.BRIEF.value,
            "reading_level_target": (4, 6),
            "max_explanation_length": 300,
            "description": "Patient with limited health literacy needing simple explanations",
        },
        "patient_medium_literacy": {
            "audience_type": AudienceType.PATIENT.value,
            "health_literacy": HealthLiteracy.MEDIUM.value,
            "medical_familiarity": "some_experience",
            "preferred_detail_level": DetailLevel.MODERATE.value,
            "reading_level_target": (6, 10),
            "max_explanation_length": 500,
            "description": "Average patient with moderate health understanding",
        },
        "patient_high_literacy": {
            "audience_type": AudienceType.PATIENT.value,
            "health_literacy": HealthLiteracy.HIGH.value,
            "medical_familiarity": "some_experience",
            "preferred_detail_level": DetailLevel.MODERATE.value,
            "reading_level_target": (8, 12),
            "max_explanation_length": 600,
            "description": "Health-literate patient wanting detailed understanding",
        },
        "patient_elderly": {
            "audience_type": AudienceType.PATIENT.value,
            "health_literacy": HealthLiteracy.MEDIUM.value,
            "medical_familiarity": "some_experience",
            "age_group": AgeGroup.ELDERLY.value,
            "preferred_detail_level": DetailLevel.MODERATE.value,
            "reading_level_target": (6, 8),
            "max_explanation_length": 400,
            "description": "Elderly patient needing clear, slower-paced explanations",
        },
        
        # Caregiver personas
        "caregiver_family": {
            "audience_type": AudienceType.CAREGIVER.value,
            "health_literacy": HealthLiteracy.MEDIUM.value,
            "medical_familiarity": "some_experience",
            "preferred_detail_level": DetailLevel.MODERATE.value,
            "reading_level_target": (6, 10),
            "max_explanation_length": 500,
            "description": "Family caregiver needing practical care instructions",
        },
        "caregiver_professional": {
            "audience_type": AudienceType.CAREGIVER.value,
            "health_literacy": HealthLiteracy.HIGH.value,
            "medical_familiarity": "some_experience",
            "preferred_detail_level": DetailLevel.COMPREHENSIVE.value,
            "reading_level_target": (8, 12),
            "max_explanation_length": 600,
            "description": "Professional caregiver with medical training",
        },
        "caregiver_pediatric": {
            "audience_type": AudienceType.CAREGIVER.value,
            "health_literacy": HealthLiteracy.MEDIUM.value,
            "medical_familiarity": "novice",
            "age_group": AgeGroup.PEDIATRIC_PARENT.value,
            "preferred_detail_level": DetailLevel.MODERATE.value,
            "reading_level_target": (6, 10),
            "max_explanation_length": 500,
            "description": "Parent of young child needing reassuring, clear guidance",
        },
    }
    
    @classmethod
    def create_persona(
        cls,
        audience_type: str,
        health_literacy: str = HealthLiteracy.MEDIUM.value,
        medical_familiarity: str = "some_experience",
        age_group: Optional[str] = None,
        cultural_context: Optional[str] = None,
        **kwargs
    ) -> AudiencePersona:
        """Create a custom audience persona.
        
        Args:
            audience_type: Primary audience category
            health_literacy: Health literacy level (low/medium/high)
            medical_familiarity: Level of medical knowledge
            age_group: Optional age group
            cultural_context: Optional cultural considerations
            **kwargs: Additional persona attributes
            
        Returns:
            Configured AudiencePersona instance
        """
        # Determine reading level targets based on audience and literacy
        reading_targets = cls._get_reading_targets(audience_type, health_literacy)
        
        persona = AudiencePersona(
            audience_type=audience_type,
            health_literacy=health_literacy,
            medical_familiarity=medical_familiarity,
            age_group=age_group,
            cultural_context=cultural_context,
            reading_level_target=reading_targets,
            **kwargs
        )
        
        return persona
    
    @classmethod
    def get_predefined_persona(cls, persona_id: str) -> AudiencePersona:
        """Get a predefined persona by ID.
        
        Args:
            persona_id: Identifier for the predefined persona
            
        Returns:
            Configured AudiencePersona instance
            
        Raises:
            ValueError: If persona_id is not found
        """
        if persona_id not in cls.PREDEFINED_PERSONAS:
            available = list(cls.PREDEFINED_PERSONAS.keys())
            raise ValueError(f"Unknown persona: {persona_id}. Available: {available}")
        
        config = cls.PREDEFINED_PERSONAS[persona_id]
        return AudiencePersona(persona_id=persona_id, **config)
    
    @classmethod
    def list_predefined_personas(cls) -> List[str]:
        """List all predefined persona IDs."""
        return list(cls.PREDEFINED_PERSONAS.keys())
    
    @classmethod
    def _get_reading_targets(
        cls,
        audience_type: str,
        health_literacy: str
    ) -> Tuple[int, int]:
        """Determine reading level targets based on audience and literacy."""
        targets = {
            (AudienceType.PHYSICIAN.value, HealthLiteracy.HIGH.value): (14, 18),
            (AudienceType.NURSE.value, HealthLiteracy.HIGH.value): (12, 14),
            (AudienceType.NURSE.value, HealthLiteracy.MEDIUM.value): (10, 12),
            (AudienceType.PATIENT.value, HealthLiteracy.HIGH.value): (8, 12),
            (AudienceType.PATIENT.value, HealthLiteracy.MEDIUM.value): (6, 10),
            (AudienceType.PATIENT.value, HealthLiteracy.LOW.value): (4, 6),
            (AudienceType.CAREGIVER.value, HealthLiteracy.HIGH.value): (8, 12),
            (AudienceType.CAREGIVER.value, HealthLiteracy.MEDIUM.value): (6, 10),
            (AudienceType.CAREGIVER.value, HealthLiteracy.LOW.value): (4, 8),
        }
        
        key = (audience_type, health_literacy)
        return targets.get(key, (6, 10))


class HealthLiteracyAssessor:
    """Assess and score text for health literacy appropriateness.
    
    Uses multiple heuristics to evaluate whether text matches the expected
    health literacy level of a target persona.
    """
    
    # Words that indicate complex medical content
    COMPLEX_TERMS = {
        "pathophysiology", "etiology", "prognosis", "differential",
        "contraindication", "pharmacokinetics", "hemodynamic",
        "immunosuppressive", "thromboembolism", "cardiomyopathy",
        "hyperlipidemia", "atherosclerosis", "arrhythmia",
        "bronchospasm", "nephrotoxicity", "hepatotoxicity",
    }
    
    # Common medical abbreviations
    MEDICAL_ABBREVIATIONS = {
        "BP", "HR", "RR", "SpO2", "BID", "TID", "QID", "PRN",
        "IV", "IM", "PO", "NPO", "CBC", "BMP", "CMP", "ABG",
        "ECG", "EKG", "CT", "MRI", "CXR", "MI", "CVA", "DVT",
        "PE", "COPD", "CHF", "DM", "HTN", "CAD", "GERD", "UTI",
    }
    
    # Simple alternatives to complex terms (for patient explanations)
    SIMPLE_ALTERNATIVES = {
        "hypertension": "high blood pressure",
        "myocardial infarction": "heart attack",
        "cerebrovascular accident": "stroke",
        "dyspnea": "shortness of breath",
        "edema": "swelling",
        "pyrexia": "fever",
        "emesis": "vomiting",
        "ambulate": "walk",
        "administer": "give",
    }
    
    def __init__(self, persona: AudiencePersona):
        """Initialize assessor with target persona."""
        self.persona = persona
        self.target_min, self.target_max = persona.reading_level_target
    
    def assess_readability(self, text: str) -> Dict[str, Any]:
        """Assess text readability against persona expectations.
        
        Args:
            text: Text to assess
            
        Returns:
            Dictionary with readability metrics and scores
        """
        # Calculate grade level
        try:
            import textstat
            grade_level = textstat.flesch_kincaid_grade(text)
            flesch_score = textstat.flesch_reading_ease(text)
        except ImportError:
            grade_level = self._estimate_grade_level(text)
            flesch_score = None
        
        # Check if within target range
        in_range = self.target_min <= grade_level <= self.target_max
        
        # Calculate deviation from target
        if grade_level < self.target_min:
            deviation = self.target_min - grade_level
            direction = "too_simple"
        elif grade_level > self.target_max:
            deviation = grade_level - self.target_max
            direction = "too_complex"
        else:
            deviation = 0
            direction = "appropriate"
        
        # Calculate score (0-1)
        if deviation == 0:
            score = 1.0
        else:
            # Penalize based on deviation, more severely for complexity
            penalty_factor = 1.5 if direction == "too_complex" else 1.0
            score = max(0.0, 1.0 - (deviation * 0.1 * penalty_factor))
        
        return {
            "grade_level": grade_level,
            "flesch_score": flesch_score,
            "target_range": (self.target_min, self.target_max),
            "in_range": in_range,
            "direction": direction,
            "deviation": deviation,
            "score": score,
        }
    
    def assess_terminology(self, text: str) -> Dict[str, Any]:
        """Assess terminology appropriateness for persona.
        
        Args:
            text: Text to assess
            
        Returns:
            Dictionary with terminology metrics and scores
        """
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        
        if total_words == 0:
            return {"score": 0.0, "error": "Empty text"}
        
        # Count complex terms
        complex_count = sum(
            1 for term in self.COMPLEX_TERMS
            if term.lower() in text_lower
        )
        
        # Count abbreviations
        abbrev_count = sum(
            1 for abbrev in self.MEDICAL_ABBREVIATIONS
            if re.search(rf'\b{abbrev}\b', text)
        )
        
        # Calculate densities
        complex_density = complex_count / total_words
        abbrev_density = abbrev_count / total_words
        
        # Get expected density from persona
        expected_density = self.persona.terminology.technical_terms_density
        abbrevs_allowed = self.persona.terminology.abbreviations_allowed
        
        # Score terminology appropriateness
        term_score = 1.0
        
        # Penalize if density deviates from expected
        density_diff = abs(complex_density - expected_density)
        term_score -= min(0.5, density_diff * 5)
        
        # Penalize abbreviations if not allowed
        if not abbrevs_allowed and abbrev_count > 0:
            term_score -= min(0.3, abbrev_count * 0.1)
        
        term_score = max(0.0, term_score)
        
        return {
            "complex_term_count": complex_count,
            "abbreviation_count": abbrev_count,
            "complex_density": complex_density,
            "expected_density": expected_density,
            "abbreviations_allowed": abbrevs_allowed,
            "score": term_score,
        }
    
    def assess_empathy(self, text: str) -> Dict[str, Any]:
        """Assess empathy and tone appropriateness.
        
        Args:
            text: Text to assess
            
        Returns:
            Dictionary with empathy metrics and scores
        """
        text_lower = text.lower()
        
        # Empathy indicators
        empathy_phrases = [
            "understand", "know this is", "it's normal to",
            "you may feel", "you're not alone", "we're here",
            "i'm sorry", "this can be", "take your time",
            "it's okay", "common concern", "many people",
        ]
        
        # Clinical/distant language
        clinical_phrases = [
            "the patient", "one should", "it is recommended",
            "compliance", "adherence", "non-compliant",
        ]
        
        empathy_count = sum(
            1 for phrase in empathy_phrases
            if phrase in text_lower
        )
        
        clinical_count = sum(
            1 for phrase in clinical_phrases
            if phrase in text_lower
        )
        
        # Expected empathy level
        expected = self.persona.communication.empathy_level
        
        # Score based on expected level
        if expected == "high":
            # Patients/caregivers expect high empathy
            score = min(1.0, empathy_count * 0.2)
            score -= clinical_count * 0.15
        elif expected == "moderate":
            # Nurses expect moderate empathy
            score = 0.8 if empathy_count > 0 else 0.6
        else:
            # Clinical audiences expect professional tone
            score = 0.9 - (empathy_count * 0.05)  # Slight penalty for over-empathy
        
        return {
            "empathy_phrase_count": empathy_count,
            "clinical_phrase_count": clinical_count,
            "expected_empathy_level": expected,
            "score": max(0.0, min(1.0, score)),
        }
    
    def assess_actionability(self, text: str) -> Dict[str, Any]:
        """Assess whether text provides actionable guidance.
        
        Args:
            text: Text to assess
            
        Returns:
            Dictionary with actionability metrics and scores
        """
        text_lower = text.lower()
        
        # Action-oriented phrases
        action_phrases = [
            "you should", "you need to", "make sure",
            "remember to", "don't forget", "it's important to",
            "call your doctor if", "seek help if", "return if",
            "take", "avoid", "stop", "start", "continue",
            "next step", "follow up", "schedule",
        ]
        
        action_count = sum(
            1 for phrase in action_phrases
            if phrase in text_lower
        )
        
        # Check for numbered/bulleted lists (actionable structure)
        has_list = bool(re.search(r'(\d+\.|•|-).*\n', text))
        
        expected = self.persona.communication.action_orientation
        
        if expected == "directive":
            # Caregivers need very actionable content
            score = min(1.0, action_count * 0.15)
            if has_list:
                score = min(1.0, score + 0.2)
        elif expected == "actionable":
            # Patients need clear actions
            score = min(1.0, action_count * 0.12)
            if has_list:
                score = min(1.0, score + 0.15)
        else:
            # Informational is fine
            score = 0.8 if action_count > 0 else 0.7
        
        return {
            "action_phrase_count": action_count,
            "has_list_structure": has_list,
            "expected_orientation": expected,
            "score": max(0.0, min(1.0, score)),
        }
    
    def _estimate_grade_level(self, text: str) -> float:
        """Fallback grade level estimation without textstat."""
        words = text.split()
        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        
        avg_words_per_sentence = len(words) / sentences
        
        # Count syllables (rough approximation)
        syllable_count = 0
        for word in words:
            word = re.sub(r'[^a-zA-Z]', '', word)
            if word:
                vowels = len(re.findall(r'[aeiouyAEIOUY]+', word))
                syllable_count += max(1, vowels)
        
        avg_syllables_per_word = syllable_count / len(words) if words else 1
        
        # Flesch-Kincaid Grade Level formula
        grade = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
        return max(0, grade)
    
    def comprehensive_assessment(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive health literacy assessment.
        
        Args:
            text: Text to assess
            
        Returns:
            Complete assessment results with overall score
        """
        readability = self.assess_readability(text)
        terminology = self.assess_terminology(text)
        empathy = self.assess_empathy(text)
        actionability = self.assess_actionability(text)
        
        # Weighted overall score
        weights = {
            "readability": 0.30,
            "terminology": 0.25,
            "empathy": 0.20,
            "actionability": 0.25,
        }
        
        overall_score = (
            readability["score"] * weights["readability"] +
            terminology["score"] * weights["terminology"] +
            empathy["score"] * weights["empathy"] +
            actionability["score"] * weights["actionability"]
        )
        
        return {
            "persona_id": self.persona.persona_id,
            "readability": readability,
            "terminology": terminology,
            "empathy": empathy,
            "actionability": actionability,
            "overall_score": overall_score,
            "weights": weights,
        }


class PersonaBasedScorer:
    """Score explanations based on persona expectations.
    
    Provides comprehensive scoring across all six evaluation dimensions:
    1. Clinical Accuracy (requires external input)
    2. Terminological Appropriateness
    3. Explanatory Completeness
    4. Actionability
    5. Safety & Harm Avoidance
    6. Empathy & Tone
    """
    
    def __init__(self, persona: AudiencePersona):
        """Initialize scorer with target persona."""
        self.persona = persona
        self.literacy_assessor = HealthLiteracyAssessor(persona)
    
    def score_explanation(
        self,
        explanation: str,
        original_content: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score an explanation for the target persona.
        
        Args:
            explanation: Generated explanation to score
            original_content: Original medical content (for coverage)
            reference: Reference explanation (for comparison)
            
        Returns:
            Comprehensive scoring results
        """
        # Get health literacy assessment
        literacy_assessment = self.literacy_assessor.comprehensive_assessment(explanation)
        
        # Length appropriateness
        word_count = len(explanation.split())
        max_length = self.persona.max_explanation_length
        
        length_score = 1.0
        if word_count > max_length * 1.5:
            length_score = 0.5  # Too long
        elif word_count > max_length:
            length_score = 0.8  # Slightly too long
        elif word_count < 50:
            length_score = 0.6  # Too short
        
        # Compile scores
        scores = {
            "terminological_appropriateness": literacy_assessment["terminology"]["score"],
            "explanatory_completeness": self._score_completeness(explanation, original_content),
            "actionability": literacy_assessment["actionability"]["score"],
            "empathy_tone": literacy_assessment["empathy"]["score"],
            "readability": literacy_assessment["readability"]["score"],
            "length_appropriateness": length_score,
        }
        
        # Calculate overall
        weights = {
            "terminological_appropriateness": 0.20,
            "explanatory_completeness": 0.20,
            "actionability": 0.15,
            "empathy_tone": 0.15,
            "readability": 0.20,
            "length_appropriateness": 0.10,
        }
        
        overall = sum(scores[k] * weights[k] for k in scores)
        
        return {
            "persona_id": self.persona.persona_id,
            "scores": scores,
            "weights": weights,
            "overall_score": overall,
            "details": literacy_assessment,
            "word_count": word_count,
            "max_length": max_length,
        }
    
    def _score_completeness(
        self,
        explanation: str,
        original: Optional[str]
    ) -> float:
        """Score explanatory completeness."""
        if not original:
            # Without original, use heuristics
            word_count = len(explanation.split())
            if word_count < 30:
                return 0.4
            elif word_count < 100:
                return 0.7
            else:
                return 0.9
        
        # Check coverage of key terms from original
        original_words = set(original.lower().split())
        explanation_words = set(explanation.lower().split())
        
        # Filter to content words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                     "and", "or", "but", "of", "to", "for", "in", "on", "at", "with"}
        original_content = original_words - stopwords
        explanation_content = explanation_words - stopwords
        
        if not original_content:
            return 0.7
        
        overlap = len(original_content & explanation_content)
        coverage = overlap / len(original_content)
        
        return min(1.0, coverage + 0.2)  # Boost a bit as paraphrasing is expected


def get_default_personas() -> Dict[str, AudiencePersona]:
    """Get default set of personas for evaluation."""
    return {
        "physician": PersonaFactory.get_predefined_persona("physician_generalist"),
        "nurse": PersonaFactory.get_predefined_persona("nurse_general"),
        "patient": PersonaFactory.get_predefined_persona("patient_medium_literacy"),
        "caregiver": PersonaFactory.get_predefined_persona("caregiver_family"),
    }


def get_comprehensive_personas() -> Dict[str, AudiencePersona]:
    """Get comprehensive set of all predefined personas."""
    return {
        persona_id: PersonaFactory.get_predefined_persona(persona_id)
        for persona_id in PersonaFactory.list_predefined_personas()
    }

