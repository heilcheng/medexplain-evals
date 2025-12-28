"""Enhanced data schema for MedExplain-Evals.

This module defines the core data structures for the benchmark, including
the enhanced MedExplainItem with support for multimodal content, medical
specialty tagging, and safety-critical flagging.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
from pathlib import Path


class MedicalSpecialty(str, Enum):
    """Medical specialties supported by the benchmark."""
    CARDIOLOGY = "cardiology"
    ONCOLOGY = "oncology"
    PEDIATRICS = "pediatrics"
    EMERGENCY = "emergency"
    MENTAL_HEALTH = "mental_health"
    NEUROLOGY = "neurology"
    PULMONOLOGY = "pulmonology"
    ENDOCRINOLOGY = "endocrinology"
    GASTROENTEROLOGY = "gastroenterology"
    NEPHROLOGY = "nephrology"
    DERMATOLOGY = "dermatology"
    OPHTHALMOLOGY = "ophthalmology"
    ORTHOPEDICS = "orthopedics"
    INFECTIOUS_DISEASE = "infectious_disease"
    GENERAL_MEDICINE = "general_medicine"
    OBSTETRICS_GYNECOLOGY = "obstetrics_gynecology"
    RHEUMATOLOGY = "rheumatology"
    HEMATOLOGY = "hematology"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"


class ComplexityLevel(str, Enum):
    """Complexity levels for medical content."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class DatasetSource(str, Enum):
    """Supported dataset sources."""
    MEDQA_USMLE = "MedQA-USMLE"
    PUBMEDQA = "PubMedQA"
    MEDMCQA = "MedMCQA"
    LIVEQA = "LiveQA"
    HEALTHSEARCHQA = "HealthSearchQA"
    ICLINIQ = "iCliniq"
    COCHRANE = "Cochrane"
    MIMIC_IV = "MIMIC-IV"
    VQA_RAD = "VQA-RAD"
    PATHVQA = "PathVQA"
    CLINICAL_VIGNETTE = "ClinicalVignette"
    CUSTOM = "Custom"


@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from text."""
    text: str
    entity_type: str  # condition, medication, procedure, symptom, anatomy
    umls_cui: Optional[str] = None  # UMLS Concept Unique Identifier
    snomed_code: Optional[str] = None
    rxnorm_code: Optional[str] = None  # For medications
    confidence: float = 1.0
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "umls_cui": self.umls_cui,
            "snomed_code": self.snomed_code,
            "rxnorm_code": self.rxnorm_code,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedicalEntity":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            entity_type=data["entity_type"],
            umls_cui=data.get("umls_cui"),
            snomed_code=data.get("snomed_code"),
            rxnorm_code=data.get("rxnorm_code"),
            confidence=data.get("confidence", 1.0),
            start_pos=data.get("start_pos"),
            end_pos=data.get("end_pos"),
        )


@dataclass
class ClinicalContext:
    """Clinical context for a benchmark item."""
    patient_age: Optional[str] = None  # e.g., "45 years", "pediatric", "elderly"
    patient_sex: Optional[str] = None
    setting: Optional[str] = None  # emergency, inpatient, outpatient, telehealth
    chief_complaint: Optional[str] = None
    relevant_history: Optional[str] = None
    current_medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    social_history: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "patient_age": self.patient_age,
            "patient_sex": self.patient_sex,
            "setting": self.setting,
            "chief_complaint": self.chief_complaint,
            "relevant_history": self.relevant_history,
            "current_medications": self.current_medications,
            "allergies": self.allergies,
            "social_history": self.social_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalContext":
        """Create from dictionary."""
        return cls(
            patient_age=data.get("patient_age"),
            patient_sex=data.get("patient_sex"),
            setting=data.get("setting"),
            chief_complaint=data.get("chief_complaint"),
            relevant_history=data.get("relevant_history"),
            current_medications=data.get("current_medications"),
            allergies=data.get("allergies"),
            social_history=data.get("social_history"),
        )


@dataclass
class MultimodalContent:
    """Multimodal content for image-based medical scenarios."""
    image_paths: List[str] = field(default_factory=list)
    image_modality: Optional[str] = None  # radiology, dermatology, pathology, ophthalmology
    image_descriptions: Optional[List[str]] = None
    findings: Optional[str] = None  # Key findings in the image
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "image_paths": self.image_paths,
            "image_modality": self.image_modality,
            "image_descriptions": self.image_descriptions,
            "findings": self.findings,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalContent":
        """Create from dictionary."""
        return cls(
            image_paths=data.get("image_paths", []),
            image_modality=data.get("image_modality"),
            image_descriptions=data.get("image_descriptions"),
            findings=data.get("findings"),
        )


@dataclass
class ReferenceExplanation:
    """Gold-standard reference explanation for an audience."""
    audience_type: str
    explanation: str
    health_literacy_level: str = "medium"  # low, medium, high
    source: Optional[str] = None  # expert, validated, synthetic
    quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "audience_type": self.audience_type,
            "explanation": self.explanation,
            "health_literacy_level": self.health_literacy_level,
            "source": self.source,
            "quality_score": self.quality_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceExplanation":
        """Create from dictionary."""
        return cls(
            audience_type=data["audience_type"],
            explanation=data["explanation"],
            health_literacy_level=data.get("health_literacy_level", "medium"),
            source=data.get("source"),
            quality_score=data.get("quality_score"),
        )


@dataclass
class MedExplainItemV2:
    """Enhanced benchmark item for MedExplain-Evals.
    
    This is the core data structure for benchmark items, supporting:
    - Multimodal content (text + images)
    - Medical specialty tagging
    - Clinical context
    - Safety-critical flagging
    - Extracted medical entities
    - Reference explanations per audience
    """
    
    # Core fields
    id: str
    medical_content: str
    
    # Classification
    specialty: str = MedicalSpecialty.GENERAL_MEDICINE.value
    complexity_level: str = ComplexityLevel.INTERMEDIATE.value
    source_dataset: str = DatasetSource.CUSTOM.value
    
    # Extended content
    clinical_context: Optional[ClinicalContext] = None
    multimodal: Optional[MultimodalContent] = None
    
    # Medical knowledge
    medical_entities: List[MedicalEntity] = field(default_factory=list)
    icd10_codes: List[str] = field(default_factory=list)
    
    # Reference data
    reference_explanations: Dict[str, ReferenceExplanation] = field(default_factory=dict)
    
    # Safety flags
    safety_critical: bool = False
    safety_categories: List[str] = field(default_factory=list)  # dosage, allergy, emergency, etc.
    
    # Metadata
    language: str = "en"
    created_at: Optional[str] = None
    version: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "medical_content": self.medical_content,
            "specialty": self.specialty,
            "complexity_level": self.complexity_level,
            "source_dataset": self.source_dataset,
            "clinical_context": self.clinical_context.to_dict() if self.clinical_context else None,
            "multimodal": self.multimodal.to_dict() if self.multimodal else None,
            "medical_entities": [e.to_dict() for e in self.medical_entities],
            "icd10_codes": self.icd10_codes,
            "reference_explanations": {
                k: v.to_dict() for k, v in self.reference_explanations.items()
            },
            "safety_critical": self.safety_critical,
            "safety_categories": self.safety_categories,
            "language": self.language,
            "created_at": self.created_at,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedExplainItemV2":
        """Create from dictionary."""
        clinical_context = None
        if data.get("clinical_context"):
            clinical_context = ClinicalContext.from_dict(data["clinical_context"])
        
        multimodal = None
        if data.get("multimodal"):
            multimodal = MultimodalContent.from_dict(data["multimodal"])
        
        medical_entities = [
            MedicalEntity.from_dict(e) for e in data.get("medical_entities", [])
        ]
        
        reference_explanations = {
            k: ReferenceExplanation.from_dict(v)
            for k, v in data.get("reference_explanations", {}).items()
        }
        
        return cls(
            id=data["id"],
            medical_content=data["medical_content"],
            specialty=data.get("specialty", MedicalSpecialty.GENERAL_MEDICINE.value),
            complexity_level=data.get("complexity_level", ComplexityLevel.INTERMEDIATE.value),
            source_dataset=data.get("source_dataset", DatasetSource.CUSTOM.value),
            clinical_context=clinical_context,
            multimodal=multimodal,
            medical_entities=medical_entities,
            icd10_codes=data.get("icd10_codes", []),
            reference_explanations=reference_explanations,
            safety_critical=data.get("safety_critical", False),
            safety_categories=data.get("safety_categories", []),
            language=data.get("language", "en"),
            created_at=data.get("created_at"),
            version=data.get("version", "2.0"),
        )
    
    def has_images(self) -> bool:
        """Check if item has multimodal content."""
        return self.multimodal is not None and len(self.multimodal.image_paths) > 0
    
    def get_reference_for_audience(self, audience: str) -> Optional[str]:
        """Get reference explanation for a specific audience."""
        if audience in self.reference_explanations:
            return self.reference_explanations[audience].explanation
        return None
    
    def is_high_risk(self) -> bool:
        """Check if item is high-risk (safety critical or emergency)."""
        high_risk_categories = ["emergency", "dosage", "allergy", "contraindication"]
        return self.safety_critical or any(
            cat in self.safety_categories for cat in high_risk_categories
        )


def save_benchmark_items_v2(
    items: List[MedExplainItemV2],
    output_path: str,
    pretty_print: bool = True
) -> None:
    """Save MedExplainItemV2 objects to a JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    items_data = [item.to_dict() for item in items]
    
    with open(output_file, "w", encoding="utf-8") as f:
        if pretty_print:
            json.dump(items_data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(items_data, f, ensure_ascii=False)


def load_benchmark_items_v2(input_path: str) -> List[MedExplainItemV2]:
    """Load MedExplainItemV2 objects from a JSON file."""
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {input_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [MedExplainItemV2.from_dict(item) for item in data]

