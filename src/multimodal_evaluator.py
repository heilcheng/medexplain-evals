"""Multimodal evaluation support for MEQ-Bench 2.0.

This module provides evaluation capabilities for image-to-text medical
explanations, supporting radiology, dermatology, pathology, and
ophthalmology modalities.

Supported Imaging Modalities:
    - Radiology: X-ray, CT, MRI scans
    - Dermatology: Skin condition images
    - Pathology: Histology slides
    - Ophthalmology: Retinal images

Features:
    - Image-aware evaluation using multimodal LLMs
    - Modality-specific evaluation criteria
    - Visual-textual alignment scoring
    - Medical imaging terminology verification
    - Finding coverage analysis

Example:
    ```python
    from multimodal_evaluator import MultimodalMedicalEvaluator
    
    evaluator = MultimodalMedicalEvaluator()
    
    score = evaluator.evaluate_with_image(
        image_path="chest_xray.png",
        explanation="The X-ray shows...",
        modality="radiology",
        audience="patient"
    )
    
    print(f"Visual alignment: {score.visual_alignment}")
    print(f"Finding coverage: {score.finding_coverage}")
    ```
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum

from .model_clients import UnifiedModelClient
from .audience_personas import AudiencePersona, PersonaFactory
from .ensemble_judge import EnsembleLLMJudge, EnsembleScore, EVALUATION_DIMENSIONS

logger = logging.getLogger("meq_bench.multimodal_evaluator")


class ImagingModality(str, Enum):
    """Supported medical imaging modalities."""
    RADIOLOGY = "radiology"
    DERMATOLOGY = "dermatology"
    PATHOLOGY = "pathology"
    OPHTHALMOLOGY = "ophthalmology"


@dataclass
class ImageContent:
    """Represents medical image content for evaluation."""
    image_path: str
    modality: str
    expected_findings: Optional[List[str]] = None
    region_of_interest: Optional[str] = None
    clinical_context: Optional[str] = None
    reference_explanation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "modality": self.modality,
            "expected_findings": self.expected_findings,
            "region_of_interest": self.region_of_interest,
            "clinical_context": self.clinical_context,
        }


@dataclass
class MultimodalScore:
    """Score for multimodal evaluation."""
    # Core evaluation dimensions (from ensemble judge)
    text_score: EnsembleScore
    
    # Multimodal-specific dimensions
    visual_alignment: float  # How well explanation aligns with image content
    finding_coverage: float  # Coverage of key findings from image
    image_reference_accuracy: float  # Accuracy of visual references
    modality_appropriateness: float  # Appropriate terminology for modality
    
    # Overall
    overall: float
    
    # Details
    detected_findings: List[str] = field(default_factory=list)
    referenced_findings: List[str] = field(default_factory=list)
    missing_findings: List[str] = field(default_factory=list)
    image_analysis: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_score": self.text_score.to_dict(),
            "visual_alignment": self.visual_alignment,
            "finding_coverage": self.finding_coverage,
            "image_reference_accuracy": self.image_reference_accuracy,
            "modality_appropriateness": self.modality_appropriateness,
            "overall": self.overall,
            "detected_findings": self.detected_findings,
            "referenced_findings": self.referenced_findings,
            "missing_findings": self.missing_findings,
        }


# Modality-specific terminology and expected content
MODALITY_CONFIG = {
    ImagingModality.RADIOLOGY.value: {
        "name": "Radiology",
        "terminology": [
            "density", "opacity", "lucency", "consolidation", "infiltrate",
            "effusion", "cardiomegaly", "atelectasis", "pneumothorax",
            "nodule", "mass", "cavity", "interstitial", "bilateral",
            "unilateral", "hilar", "mediastinal", "diaphragm", "costophrenic",
        ],
        "finding_patterns": [
            r"(?:increased|decreased)\s+(?:density|opacity)",
            r"(?:no\s+)?(?:acute|abnormal)\s+findings",
            r"(?:normal|abnormal)\s+(?:heart|lung|chest)",
            r"(?:left|right|bilateral)\s+(?:lung|side)",
        ],
        "description": "X-ray, CT, or MRI scan interpretation",
    },
    ImagingModality.DERMATOLOGY.value: {
        "name": "Dermatology",
        "terminology": [
            "lesion", "macule", "papule", "vesicle", "bulla", "pustule",
            "nodule", "plaque", "patch", "erythema", "pigmented",
            "borders", "asymmetry", "color", "diameter", "evolving",
            "scaling", "crusting", "ulceration", "distribution",
        ],
        "finding_patterns": [
            r"(?:irregular|regular)\s+borders",
            r"(?:raised|flat)\s+lesion",
            r"(?:uniform|varied)\s+color",
            r"(?:localized|widespread)\s+(?:rash|eruption)",
        ],
        "description": "Skin condition image analysis",
    },
    ImagingModality.PATHOLOGY.value: {
        "name": "Pathology",
        "terminology": [
            "cells", "nuclei", "cytoplasm", "stroma", "tissue",
            "malignant", "benign", "dysplasia", "necrosis", "mitosis",
            "differentiation", "grade", "margin", "invasion", "metastasis",
            "histology", "biopsy", "specimen", "architecture",
        ],
        "finding_patterns": [
            r"(?:well|poorly|moderately)\s+differentiated",
            r"(?:nuclear|cellular)\s+(?:atypia|pleomorphism)",
            r"(?:positive|negative)\s+(?:margin|staining)",
            r"(?:presence|absence)\s+of\s+(?:invasion|necrosis)",
        ],
        "description": "Histopathology slide interpretation",
    },
    ImagingModality.OPHTHALMOLOGY.value: {
        "name": "Ophthalmology",
        "terminology": [
            "retina", "macula", "optic disc", "cup-to-disc ratio",
            "vessels", "hemorrhage", "exudate", "drusen", "edema",
            "neovascularization", "microaneurysm", "cotton wool spots",
            "fovea", "choroid", "pigment epithelium",
        ],
        "finding_patterns": [
            r"(?:normal|abnormal)\s+optic\s+(?:disc|nerve)",
            r"(?:presence|absence)\s+of\s+(?:hemorrhage|exudate)",
            r"macular\s+(?:edema|degeneration|hole)",
            r"(?:diabetic|hypertensive)\s+retinopathy",
        ],
        "description": "Retinal and ocular image analysis",
    },
}

# Multimodal models that can analyze images
MULTIMODAL_MODELS = [
    "gpt-5.2",
    "gpt-5.1",
    "claude-opus-4.5",
    "claude-sonnet-4.5",
    "gemini-3-pro",
    "gemini-3-ultra",
    "llama-4-maverick",
]


class ImageAnalyzer:
    """Analyze medical images using multimodal LLMs."""
    
    IMAGE_ANALYSIS_PROMPT = """You are an expert medical image analyst. Analyze the provided medical image and identify:

1. **Imaging Modality**: What type of medical image is this (X-ray, CT, MRI, skin photo, histology, retinal scan, etc.)?

2. **Key Findings**: List all significant findings visible in the image.

3. **Normal Structures**: List normal anatomical structures visible.

4. **Abnormalities**: List any abnormalities or pathological findings.

5. **Image Quality**: Note any issues with image quality that might affect interpretation.

Provide your analysis in the following JSON format:
{
    "modality": "detected imaging modality",
    "body_region": "region shown",
    "key_findings": ["finding 1", "finding 2", ...],
    "normal_structures": ["structure 1", "structure 2", ...],
    "abnormalities": ["abnormality 1", ...],
    "image_quality": "good/fair/poor",
    "confidence": 0.0-1.0
}"""

    def __init__(
        self,
        model: str = "gpt-5.1",
        client: Optional[UnifiedModelClient] = None,
    ):
        """Initialize image analyzer.
        
        Args:
            model: Multimodal model to use for analysis
            client: Model client
        """
        if model not in MULTIMODAL_MODELS:
            logger.warning(f"Model {model} may not support images. Using gpt-5.1")
            model = "gpt-5.1"
        
        self.model = model
        self.client = client or UnifiedModelClient()
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze a medical image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Analysis results including findings and modality
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            response = self.client.generate_with_image(
                model=self.model,
                messages=[{"role": "user", "content": self.IMAGE_ANALYSIS_PROMPT}],
                image_path=image_path,
                temperature=0.2,
            )
            
            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback: return raw response
            return {
                "modality": "unknown",
                "key_findings": [],
                "raw_response": response.content,
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "modality": "unknown",
                "key_findings": [],
                "error": str(e),
            }


class VisualAlignmentScorer:
    """Score alignment between image content and text explanation."""
    
    ALIGNMENT_PROMPT = """You are evaluating how well a text explanation aligns with a medical image.

## Image Analysis Results
{image_analysis}

## Text Explanation to Evaluate
{explanation}

## Evaluation Criteria

Rate the following on a scale of 1-5:

1. **Visual Alignment** (1-5): Does the explanation accurately describe what is visible in the image?
   - 5: Perfect alignment, all visual elements correctly described
   - 3: Partial alignment, some elements described correctly
   - 1: Poor alignment, explanation doesn't match image

2. **Finding Coverage** (1-5): Are all important findings from the image mentioned?
   - 5: All significant findings covered
   - 3: Most findings covered, some omissions
   - 1: Major findings missing

3. **Accuracy** (1-5): Are references to image content accurate?
   - 5: All image references accurate
   - 3: Some inaccuracies in image description
   - 1: Major inaccuracies or contradictions

Provide your evaluation in JSON format:
{
    "visual_alignment": {"score": 1-5, "reasoning": "..."},
    "finding_coverage": {"score": 1-5, "reasoning": "..."},
    "accuracy": {"score": 1-5, "reasoning": "..."},
    "findings_mentioned": ["finding1", "finding2", ...],
    "findings_missed": ["finding1", "finding2", ...]
}"""

    def __init__(
        self,
        model: str = "gpt-5.1",
        client: Optional[UnifiedModelClient] = None,
    ):
        """Initialize alignment scorer.
        
        Args:
            model: Model for scoring
            client: Model client
        """
        self.model = model
        self.client = client or UnifiedModelClient()
    
    def score_alignment(
        self,
        image_analysis: Dict[str, Any],
        explanation: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score alignment between image and explanation.
        
        Args:
            image_analysis: Results from ImageAnalyzer
            explanation: Text explanation to evaluate
            image_path: Optional image path for visual verification
            
        Returns:
            Alignment scores and details
        """
        import json
        
        prompt = self.ALIGNMENT_PROMPT.format(
            image_analysis=json.dumps(image_analysis, indent=2),
            explanation=explanation,
        )
        
        try:
            # If image available, use multimodal for verification
            if image_path and Path(image_path).exists():
                response = self.client.generate_with_image(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    image_path=image_path,
                    temperature=0.2,
                )
            else:
                response = self.client.generate(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            
            # Parse response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())
            
            return {"error": "Could not parse response"}
            
        except Exception as e:
            logger.error(f"Alignment scoring failed: {e}")
            return {"error": str(e)}


class ModalityTerminologyChecker:
    """Check appropriate use of modality-specific terminology."""
    
    def __init__(self, modality: str):
        """Initialize terminology checker.
        
        Args:
            modality: Imaging modality to check for
        """
        self.modality = modality
        self.config = MODALITY_CONFIG.get(modality, {})
        self.expected_terms = self.config.get("terminology", [])
        self.patterns = self.config.get("finding_patterns", [])
    
    def check_terminology(
        self,
        explanation: str,
        audience: str
    ) -> Dict[str, Any]:
        """Check terminology usage in explanation.
        
        Args:
            explanation: Text to check
            audience: Target audience
            
        Returns:
            Terminology check results
        """
        import re
        
        text_lower = explanation.lower()
        
        # Count modality-specific terms used
        terms_found = []
        for term in self.expected_terms:
            if term.lower() in text_lower:
                terms_found.append(term)
        
        # Check for finding patterns
        patterns_matched = []
        for pattern in self.patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            patterns_matched.extend(matches)
        
        # Calculate appropriateness based on audience
        words = explanation.split()
        term_density = len(terms_found) / len(words) if words else 0
        
        # Expected density varies by audience
        expected_density = {
            "physician": 0.05,
            "nurse": 0.03,
            "patient": 0.01,
            "caregiver": 0.01,
        }.get(audience, 0.02)
        
        # Score based on density match
        if audience in ["patient", "caregiver"]:
            # Lower density is better for lay audiences
            if term_density > expected_density * 2:
                appropriateness_score = 0.5  # Too technical
            else:
                appropriateness_score = 1.0 - (term_density / (expected_density * 2))
        else:
            # Higher density expected for clinical audiences
            appropriateness_score = min(1.0, term_density / expected_density)
        
        return {
            "modality": self.modality,
            "terms_found": terms_found,
            "term_count": len(terms_found),
            "term_density": term_density,
            "expected_density": expected_density,
            "patterns_matched": patterns_matched,
            "appropriateness_score": max(0.0, min(1.0, appropriateness_score)),
        }


class MultimodalMedicalEvaluator:
    """Main evaluator for multimodal medical content.
    
    Combines image analysis, text evaluation, and visual-textual
    alignment scoring for comprehensive multimodal assessment.
    """
    
    def __init__(
        self,
        multimodal_model: str = "gpt-5.1",
        client: Optional[UnifiedModelClient] = None,
        text_judge: Optional[EnsembleLLMJudge] = None,
    ):
        """Initialize multimodal evaluator.
        
        Args:
            multimodal_model: Model for image analysis
            client: Model client
            text_judge: Judge for text evaluation
        """
        self.client = client or UnifiedModelClient()
        self.multimodal_model = multimodal_model
        
        self.image_analyzer = ImageAnalyzer(multimodal_model, self.client)
        self.alignment_scorer = VisualAlignmentScorer(multimodal_model, self.client)
        self.text_judge = text_judge
    
    def evaluate_with_image(
        self,
        image_path: str,
        explanation: str,
        modality: str,
        audience: str,
        expected_findings: Optional[List[str]] = None,
        reference_explanation: Optional[str] = None,
        persona: Optional[AudiencePersona] = None,
    ) -> MultimodalScore:
        """Evaluate a medical image explanation.
        
        Args:
            image_path: Path to medical image
            explanation: Text explanation to evaluate
            modality: Imaging modality
            audience: Target audience
            expected_findings: Optional list of expected findings
            reference_explanation: Optional reference for comparison
            persona: Optional audience persona
            
        Returns:
            MultimodalScore with comprehensive evaluation
        """
        # Step 1: Analyze the image
        logger.info(f"Analyzing image: {image_path}")
        image_analysis = self.image_analyzer.analyze_image(image_path)
        
        # Step 2: Score visual alignment
        logger.info("Scoring visual-textual alignment")
        alignment_result = self.alignment_scorer.score_alignment(
            image_analysis=image_analysis,
            explanation=explanation,
            image_path=image_path,
        )
        
        # Step 3: Check modality-specific terminology
        term_checker = ModalityTerminologyChecker(modality)
        term_result = term_checker.check_terminology(explanation, audience)
        
        # Step 4: Evaluate text quality (if judge available)
        text_score = None
        if self.text_judge:
            # Create text content for judge
            original_content = self._build_original_content(
                image_analysis, modality, reference_explanation
            )
            
            text_score = self.text_judge.evaluate(
                original=original_content,
                explanation=explanation,
                audience=audience,
                persona=persona,
            )
        else:
            # Create minimal text score
            from .ensemble_judge import EnsembleScore
            text_score = EnsembleScore(
                overall=3.0,
                dimensions={},
                dimension_details={},
                judge_results=[],
                agreement_score=0,
                confidence=0,
                audience=audience,
            )
        
        # Step 5: Calculate finding coverage
        detected_findings = image_analysis.get("key_findings", [])
        if expected_findings:
            detected_findings = expected_findings
        
        referenced = alignment_result.get("findings_mentioned", [])
        missing = alignment_result.get("findings_missed", [])
        
        if detected_findings:
            finding_coverage = len(referenced) / len(detected_findings)
        else:
            finding_coverage = 1.0  # No findings to cover
        
        # Step 6: Extract scores
        visual_alignment = self._extract_score(
            alignment_result.get("visual_alignment", {}), 3.0
        ) / 5.0
        
        image_accuracy = self._extract_score(
            alignment_result.get("accuracy", {}), 3.0
        ) / 5.0
        
        modality_appropriateness = term_result.get("appropriateness_score", 0.7)
        
        # Step 7: Calculate overall score
        weights = {
            "text_quality": 0.40,
            "visual_alignment": 0.25,
            "finding_coverage": 0.20,
            "modality_appropriateness": 0.15,
        }
        
        overall = (
            (text_score.overall / 5.0) * weights["text_quality"] +
            visual_alignment * weights["visual_alignment"] +
            finding_coverage * weights["finding_coverage"] +
            modality_appropriateness * weights["modality_appropriateness"]
        )
        
        # Scale to 1-5
        overall = overall * 5.0
        
        return MultimodalScore(
            text_score=text_score,
            visual_alignment=visual_alignment * 5.0,
            finding_coverage=finding_coverage * 5.0,
            image_reference_accuracy=image_accuracy * 5.0,
            modality_appropriateness=modality_appropriateness * 5.0,
            overall=overall,
            detected_findings=detected_findings,
            referenced_findings=referenced,
            missing_findings=missing,
            image_analysis=str(image_analysis),
        )
    
    def _build_original_content(
        self,
        image_analysis: Dict[str, Any],
        modality: str,
        reference: Optional[str],
    ) -> str:
        """Build original content from image analysis for text evaluation."""
        parts = [f"Medical Image: {modality}"]
        
        if image_analysis.get("body_region"):
            parts.append(f"Region: {image_analysis['body_region']}")
        
        if image_analysis.get("key_findings"):
            parts.append(f"Key Findings: {', '.join(image_analysis['key_findings'])}")
        
        if image_analysis.get("abnormalities"):
            parts.append(f"Abnormalities: {', '.join(image_analysis['abnormalities'])}")
        
        if reference:
            parts.append(f"\nReference Explanation:\n{reference}")
        
        return "\n".join(parts)
    
    def _extract_score(
        self,
        score_data: Any,
        default: float
    ) -> float:
        """Extract score from various formats."""
        if isinstance(score_data, dict):
            return float(score_data.get("score", default))
        elif isinstance(score_data, (int, float)):
            return float(score_data)
        else:
            return default
    
    def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
        audience: str,
    ) -> List[MultimodalScore]:
        """Evaluate multiple image-explanation pairs.
        
        Args:
            items: List of dicts with 'image_path', 'explanation', 'modality'
            audience: Target audience
            
        Returns:
            List of MultimodalScore results
        """
        results = []
        
        for item in items:
            try:
                score = self.evaluate_with_image(
                    image_path=item["image_path"],
                    explanation=item["explanation"],
                    modality=item.get("modality", ImagingModality.RADIOLOGY.value),
                    audience=audience,
                    expected_findings=item.get("expected_findings"),
                )
                results.append(score)
            except Exception as e:
                logger.error(f"Error evaluating {item.get('image_path')}: {e}")
                # Append a zero score
                results.append(MultimodalScore(
                    text_score=EnsembleScore(
                        overall=0, dimensions={}, dimension_details={},
                        judge_results=[], agreement_score=0, confidence=0,
                        audience=audience,
                    ),
                    visual_alignment=0,
                    finding_coverage=0,
                    image_reference_accuracy=0,
                    modality_appropriateness=0,
                    overall=0,
                ))
        
        return results


def get_supported_modalities() -> List[str]:
    """Get list of supported imaging modalities."""
    return [m.value for m in ImagingModality]


def get_modality_info(modality: str) -> Dict[str, Any]:
    """Get information about a specific modality."""
    return MODALITY_CONFIG.get(modality, {})


def get_multimodal_models() -> List[str]:
    """Get list of models that support image input."""
    return MULTIMODAL_MODELS

