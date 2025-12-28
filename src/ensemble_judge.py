"""Ensemble LLM-as-Judge framework for MedExplain-Evals.

This module implements a multi-model judge ensemble using late-2025 frontier
models for robust, calibrated evaluation of medical explanations.

Features:
    - Weighted ensemble of top reasoning models
    - G-Eval style structured rubrics with chain-of-thought
    - Per-audience specialized evaluation criteria
    - Disagreement analysis and calibration
    - Detailed scoring across 6 dimensions

Judge Ensemble:
    Primary Judges (85% weight):
        - GPT-5.1 (OpenAI): 30%
        - Claude Opus 4.5 (Anthropic): 30%
        - Gemini 3 Pro (Google): 25%
    Secondary Judges (15% weight):
        - DeepSeek-V3: 10%
        - Qwen3-Max: 5%

Example:
    ```python
    from ensemble_judge import EnsembleLLMJudge
    
    judge = EnsembleLLMJudge()
    
    score = judge.evaluate(
        original="Diabetes is a metabolic disorder...",
        explanation="Your blood sugar levels are high...",
        audience="patient",
    )
    
    print(f"Overall: {score.overall}")
    print(f"Dimensions: {score.dimensions}")
    ```
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .model_clients import UnifiedModelClient, GenerationResult
from .audience_personas import AudiencePersona, PersonaFactory

logger = logging.getLogger("medexplain.ensemble_judge")


@dataclass
class JudgeConfig:
    """Configuration for an individual judge in the ensemble."""
    model: str
    provider: str
    weight: float
    enabled: bool = True
    temperature: float = 0.1
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "weight": self.weight,
            "enabled": self.enabled,
        }


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    dimension: str
    score: float  # 1-5 scale
    reasoning: str
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


@dataclass
class JudgeResult:
    """Result from a single judge evaluation."""
    judge_model: str
    dimensions: Dict[str, DimensionScore]
    overall_score: float
    raw_response: str
    latency_ms: float
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_model": self.judge_model,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "overall_score": self.overall_score,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class EnsembleScore:
    """Final ensemble evaluation score."""
    overall: float  # 1-5 scale
    dimensions: Dict[str, float]
    dimension_details: Dict[str, Dict[str, Any]]
    judge_results: List[JudgeResult]
    agreement_score: float  # Inter-judge agreement
    confidence: float
    audience: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "dimensions": self.dimensions,
            "dimension_details": self.dimension_details,
            "judge_results": [j.to_dict() for j in self.judge_results],
            "agreement_score": self.agreement_score,
            "confidence": self.confidence,
            "audience": self.audience,
            "metadata": self.metadata,
        }


# Evaluation dimensions with detailed rubrics
EVALUATION_DIMENSIONS = {
    "factual_accuracy": {
        "name": "Factual & Clinical Accuracy",
        "description": "Accuracy of medical information, absence of errors or contradictions",
        "weight": 0.25,
        "rubric": """
5 - All medical facts are accurate, current, and evidence-based. No errors or misleading information.
4 - Almost all facts are accurate with very minor imprecisions that don't affect patient safety.
3 - Generally accurate but contains some imprecisions or outdated information.
2 - Contains significant factual errors or misleading information.
1 - Severely inaccurate, dangerous misinformation, or contradicts established medical knowledge.
""",
    },
    "terminological_appropriateness": {
        "name": "Terminological Appropriateness",
        "description": "Appropriate use of medical terminology for the target audience",
        "weight": 0.15,
        "rubric": """
5 - Terminology perfectly calibrated to audience. Technical terms explained when needed. Appropriate abbreviations.
4 - Very appropriate terminology with only minor mismatches in complexity.
3 - Generally appropriate but some terms too complex or too simplified for audience.
2 - Significant terminology mismatch - too technical or too simplified, causing confusion.
1 - Completely inappropriate terminology - incomprehensible to audience or insultingly simple.
""",
    },
    "explanatory_completeness": {
        "name": "Explanatory Completeness",
        "description": "Coverage of essential information without overwhelming",
        "weight": 0.20,
        "rubric": """
5 - All essential information covered comprehensively. Perfect balance of depth and accessibility.
4 - Most essential information covered with appropriate detail.
3 - Covers main points but misses some important details or includes too much/too little.
2 - Significant gaps in coverage or overwhelming with unnecessary information.
1 - Critically incomplete or so overwhelming as to be useless.
""",
    },
    "actionability": {
        "name": "Actionability & Utility",
        "description": "Clear, practical guidance the reader can act upon",
        "weight": 0.15,
        "rubric": """
5 - Crystal clear, specific, actionable steps. Reader knows exactly what to do.
4 - Clear actionable guidance with only minor ambiguities.
3 - Some actionable information but lacks specificity or clarity.
2 - Vague or confusing guidance. Reader unsure what to do.
1 - No actionable information or contradictory/impossible instructions.
""",
    },
    "safety": {
        "name": "Safety & Harm Avoidance",
        "description": "Appropriate warnings, no dangerous advice, safety consciousness",
        "weight": 0.15,
        "rubric": """
5 - Excellent safety awareness. All necessary warnings included. No harmful advice.
4 - Good safety with appropriate warnings. Minor omissions that don't pose risk.
3 - Adequate safety but missing some important warnings or precautions.
2 - Concerning safety gaps. Missing critical warnings or potentially harmful advice.
1 - Dangerous content. Missing critical safety information or actively harmful advice.
""",
    },
    "empathy_tone": {
        "name": "Empathy & Tone",
        "description": "Appropriate emotional register and supportive communication",
        "weight": 0.10,
        "rubric": """
5 - Perfectly calibrated tone. Appropriate empathy, neither cold nor condescending. Supportive and reassuring.
4 - Very appropriate tone with minor imperfections.
3 - Generally appropriate but occasionally too clinical or too emotional.
2 - Tone mismatch - too cold, too alarming, or patronizing.
1 - Completely inappropriate tone - dismissive, frightening, or offensive.
""",
    },
}

# Audience-specific rubric adjustments
AUDIENCE_RUBRIC_ADJUSTMENTS = {
    "physician": {
        "terminological_appropriateness": "Technical medical terminology expected. Abbreviations acceptable. Clinical precision valued.",
        "empathy_tone": "Professional, clinical tone appropriate. Empathy less emphasized than precision.",
        "actionability": "Clinical decision points and evidence-based recommendations expected.",
    },
    "nurse": {
        "terminological_appropriateness": "Medical terminology acceptable but less specialized than physician-level.",
        "empathy_tone": "Practical, supportive tone. Balance of clinical and caring.",
        "actionability": "Specific nursing interventions and monitoring parameters expected.",
    },
    "patient": {
        "terminological_appropriateness": "Plain language essential. Medical terms should be explained or avoided.",
        "empathy_tone": "Warm, supportive, reassuring tone critical. Avoid anxiety-inducing language.",
        "actionability": "Simple, clear steps. What to do, when to call for help.",
    },
    "caregiver": {
        "terminological_appropriateness": "Plain language with some acceptable medical terms. More explanation than physician/nurse.",
        "empathy_tone": "Supportive, acknowledging caregiver stress. Practical and encouraging.",
        "actionability": "Clear caregiving tasks. Warning signs to watch for. When to seek help.",
    },
}


class EvaluationRubricBuilder:
    """Build evaluation prompts with G-Eval style rubrics."""
    
    SYSTEM_PROMPT = """You are an expert medical communication evaluator. Your task is to evaluate the quality of medical explanations tailored for specific audiences.

You will use a structured rubric to score each dimension from 1-5, providing brief reasoning for each score. Be rigorous but fair. Focus on whether the explanation serves its target audience effectively.

IMPORTANT: Your evaluation must consider the TARGET AUDIENCE. What works for a physician may fail for a patient, and vice versa.

Output your evaluation in the following JSON format:
{
    "dimensions": {
        "<dimension_name>": {
            "score": <1-5>,
            "reasoning": "<brief explanation for score>"
        },
        ...
    },
    "overall_reasoning": "<overall assessment>",
    "overall_score": <1-5>
}

Always output valid JSON. Do not include any text outside the JSON structure."""

    def build_evaluation_prompt(
        self,
        original_content: str,
        explanation: str,
        audience: str,
        persona: Optional[AudiencePersona] = None,
    ) -> List[Dict[str, str]]:
        """Build the evaluation prompt with rubrics.
        
        Args:
            original_content: Original medical content
            explanation: Generated explanation to evaluate
            audience: Target audience type
            persona: Optional detailed persona for evaluation
            
        Returns:
            Messages list for the judge model
        """
        # Build rubric section
        rubric_text = self._build_rubric_text(audience)
        
        # Build audience description
        audience_desc = self._build_audience_description(audience, persona)
        
        user_prompt = f"""## Evaluation Task

Evaluate the following medical explanation for a {audience.upper()} audience.

{audience_desc}

## Evaluation Rubric

{rubric_text}

## Original Medical Content

{original_content}

## Explanation to Evaluate (for {audience})

{explanation}

## Instructions

1. For each dimension, carefully assess the explanation against the rubric criteria.
2. Consider whether the explanation is appropriate for the TARGET AUDIENCE ({audience}).
3. Assign a score from 1-5 for each dimension with brief reasoning.
4. Calculate an overall score (1-5) based on dimension scores and their weights.

Provide your evaluation in JSON format as specified."""

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    
    def _build_rubric_text(self, audience: str) -> str:
        """Build rubric text with audience adjustments."""
        lines = []
        
        for dim_id, dim_config in EVALUATION_DIMENSIONS.items():
            lines.append(f"### {dim_config['name']} (Weight: {dim_config['weight']:.0%})")
            lines.append(f"{dim_config['description']}")
            lines.append("")
            lines.append(dim_config['rubric'].strip())
            
            # Add audience-specific adjustment if available
            adjustments = AUDIENCE_RUBRIC_ADJUSTMENTS.get(audience, {})
            if dim_id in adjustments:
                lines.append(f"\n**For {audience}:** {adjustments[dim_id]}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_audience_description(
        self,
        audience: str,
        persona: Optional[AudiencePersona]
    ) -> str:
        """Build audience description for context."""
        if persona:
            desc = f"""### Target Audience Profile
- **Type:** {persona.audience_type}
- **Health Literacy:** {persona.health_literacy}
- **Medical Familiarity:** {persona.medical_familiarity}
- **Preferred Detail Level:** {persona.preferred_detail_level}
- **Reading Level Target:** Grades {persona.reading_level_target[0]}-{persona.reading_level_target[1]}
"""
            if persona.description:
                desc += f"- **Description:** {persona.description}\n"
            return desc
        
        # Default descriptions
        descriptions = {
            "physician": "Medical doctor who expects clinical precision, evidence-based information, and appropriate medical terminology.",
            "nurse": "Healthcare professional who needs practical, actionable information with appropriate medical terminology.",
            "patient": "Non-medical person who needs clear, simple explanations without jargon, with empathetic and reassuring tone.",
            "caregiver": "Family member or professional caregiver who needs practical guidance on care tasks and warning signs.",
        }
        
        return f"### Target Audience\n{descriptions.get(audience, 'General audience')}"


class ResponseParser:
    """Parse judge model responses to extract scores."""
    
    def parse_response(self, response: str) -> Tuple[Dict[str, DimensionScore], float, str]:
        """Parse judge response to extract dimension scores.
        
        Args:
            response: Raw model response
            
        Returns:
            Tuple of (dimension_scores, overall_score, error_message)
        """
        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return self._parse_json_response(data)
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to extract scores manually
        return self._parse_freeform_response(response)
    
    def _parse_json_response(
        self,
        data: Dict[str, Any]
    ) -> Tuple[Dict[str, DimensionScore], float, str]:
        """Parse structured JSON response."""
        dimensions = {}
        
        dim_data = data.get("dimensions", {})
        
        for dim_id in EVALUATION_DIMENSIONS.keys():
            if dim_id in dim_data:
                score_data = dim_data[dim_id]
                dimensions[dim_id] = DimensionScore(
                    dimension=dim_id,
                    score=float(score_data.get("score", 3)),
                    reasoning=score_data.get("reasoning", ""),
                )
            else:
                # Check for alternative keys
                for key, value in dim_data.items():
                    if dim_id in key.lower().replace(" ", "_"):
                        dimensions[dim_id] = DimensionScore(
                            dimension=dim_id,
                            score=float(value.get("score", 3)),
                            reasoning=value.get("reasoning", ""),
                        )
                        break
        
        # Fill missing dimensions with neutral score
        for dim_id in EVALUATION_DIMENSIONS.keys():
            if dim_id not in dimensions:
                dimensions[dim_id] = DimensionScore(
                    dimension=dim_id,
                    score=3.0,
                    reasoning="Unable to parse score from response",
                    confidence=0.5,
                )
        
        overall = float(data.get("overall_score", 0))
        if overall == 0:
            # Calculate from dimensions
            overall = self._calculate_weighted_average(dimensions)
        
        return dimensions, overall, ""
    
    def _parse_freeform_response(
        self,
        response: str
    ) -> Tuple[Dict[str, DimensionScore], float, str]:
        """Parse unstructured response (fallback)."""
        dimensions = {}
        
        for dim_id, dim_config in EVALUATION_DIMENSIONS.items():
            # Try to find score patterns
            patterns = [
                rf"{dim_config['name']}.*?(\d)(?:\s*[/-]\s*5)?",
                rf"{dim_id}.*?(\d)(?:\s*[/-]\s*5)?",
                rf"(?:score|rating).*?{dim_id}.*?(\d)",
            ]
            
            score = 3.0  # Default
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    break
            
            dimensions[dim_id] = DimensionScore(
                dimension=dim_id,
                score=score,
                reasoning="Extracted from unstructured response",
                confidence=0.5,
            )
        
        overall = self._calculate_weighted_average(dimensions)
        
        return dimensions, overall, "Parsed from unstructured response"
    
    def _calculate_weighted_average(
        self,
        dimensions: Dict[str, DimensionScore]
    ) -> float:
        """Calculate weighted average of dimension scores."""
        total = 0.0
        weight_sum = 0.0
        
        for dim_id, dim_score in dimensions.items():
            weight = EVALUATION_DIMENSIONS[dim_id]["weight"]
            total += dim_score.score * weight
            weight_sum += weight
        
        return total / weight_sum if weight_sum > 0 else 3.0


class EnsembleLLMJudge:
    """Multi-model LLM judge ensemble for robust evaluation.
    
    Uses weighted ensemble of top reasoning models for evaluation:
    - Primary: GPT-5.1, Claude Opus 4.5, Gemini 3 Pro (85%)
    - Secondary: DeepSeek-V3, Qwen3-Max (15%)
    """
    
    # Default judge configuration
    DEFAULT_JUDGES = [
        JudgeConfig(model="gpt-5.1", provider="openai", weight=0.30),
        JudgeConfig(model="claude-opus-4.5", provider="anthropic", weight=0.30),
        JudgeConfig(model="gemini-3-pro", provider="google", weight=0.25),
        JudgeConfig(model="deepseek-v3", provider="deepseek", weight=0.10),
        JudgeConfig(model="qwen3-max", provider="alibaba", weight=0.05),
    ]
    
    def __init__(
        self,
        judges: Optional[List[JudgeConfig]] = None,
        client: Optional[UnifiedModelClient] = None,
        parallel: bool = True,
        min_judges: int = 2,
    ):
        """Initialize ensemble judge.
        
        Args:
            judges: List of judge configurations. Defaults to DEFAULT_JUDGES.
            client: Model client for API calls
            parallel: Whether to call judges in parallel
            min_judges: Minimum judges needed for valid ensemble
        """
        self.judges = judges or self.DEFAULT_JUDGES
        self.client = client or UnifiedModelClient()
        self.parallel = parallel
        self.min_judges = min_judges
        
        self.rubric_builder = EvaluationRubricBuilder()
        self.response_parser = ResponseParser()
        
        # Calibration data (can be updated via calibrate method)
        self.judge_biases: Dict[str, float] = {}
    
    def evaluate(
        self,
        original: str,
        explanation: str,
        audience: str,
        persona: Optional[AudiencePersona] = None,
    ) -> EnsembleScore:
        """Evaluate an explanation using the judge ensemble.
        
        Args:
            original: Original medical content
            explanation: Generated explanation to evaluate
            audience: Target audience type
            persona: Optional detailed persona
            
        Returns:
            EnsembleScore with weighted results from all judges
        """
        # Get persona if not provided
        if persona is None:
            try:
                persona = PersonaFactory.create_persona(audience)
            except Exception:
                persona = None
        
        # Build evaluation prompt
        messages = self.rubric_builder.build_evaluation_prompt(
            original, explanation, audience, persona
        )
        
        # Get results from all judges
        if self.parallel:
            results = self._evaluate_parallel(messages)
        else:
            results = self._evaluate_sequential(messages)
        
        # Aggregate results
        return self._aggregate_results(results, audience)
    
    def _evaluate_parallel(
        self,
        messages: List[Dict[str, str]]
    ) -> List[JudgeResult]:
        """Evaluate with all judges in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=len(self.judges)) as executor:
            futures = {
                executor.submit(self._evaluate_single, judge, messages): judge
                for judge in self.judges
                if judge.enabled
            }
            
            for future in as_completed(futures):
                judge = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Judge {judge.model} failed: {e}")
                    results.append(JudgeResult(
                        judge_model=judge.model,
                        dimensions={},
                        overall_score=0,
                        raw_response="",
                        latency_ms=0,
                        success=False,
                        error=str(e),
                    ))
        
        return results
    
    def _evaluate_sequential(
        self,
        messages: List[Dict[str, str]]
    ) -> List[JudgeResult]:
        """Evaluate with judges sequentially."""
        results = []
        
        for judge in self.judges:
            if not judge.enabled:
                continue
            
            try:
                result = self._evaluate_single(judge, messages)
                results.append(result)
            except Exception as e:
                logger.error(f"Judge {judge.model} failed: {e}")
                results.append(JudgeResult(
                    judge_model=judge.model,
                    dimensions={},
                    overall_score=0,
                    raw_response="",
                    latency_ms=0,
                    success=False,
                    error=str(e),
                ))
        
        return results
    
    def _evaluate_single(
        self,
        judge: JudgeConfig,
        messages: List[Dict[str, str]]
    ) -> JudgeResult:
        """Get evaluation from a single judge."""
        import time
        
        start_time = time.time()
        
        try:
            response = self.client.generate(
                model=judge.model,
                messages=messages,
                temperature=judge.temperature,
                max_tokens=2048,
            )
            
            latency = (time.time() - start_time) * 1000
            
            # Parse response
            dimensions, overall, error = self.response_parser.parse_response(
                response.content
            )
            
            # Apply bias correction if available
            bias = self.judge_biases.get(judge.model, 0)
            for dim_score in dimensions.values():
                dim_score.score = max(1, min(5, dim_score.score - bias))
            overall = max(1, min(5, overall - bias))
            
            return JudgeResult(
                judge_model=judge.model,
                dimensions=dimensions,
                overall_score=overall,
                raw_response=response.content,
                latency_ms=latency,
                success=True,
            )
            
        except Exception as e:
            return JudgeResult(
                judge_model=judge.model,
                dimensions={},
                overall_score=0,
                raw_response="",
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )
    
    def _aggregate_results(
        self,
        results: List[JudgeResult],
        audience: str
    ) -> EnsembleScore:
        """Aggregate results from multiple judges into final score."""
        # Filter successful results
        successful = [r for r in results if r.success]
        
        if len(successful) < self.min_judges:
            logger.warning(f"Only {len(successful)} judges succeeded, minimum is {self.min_judges}")
        
        if not successful:
            return EnsembleScore(
                overall=0,
                dimensions={},
                dimension_details={},
                judge_results=results,
                agreement_score=0,
                confidence=0,
                audience=audience,
                metadata={"error": "No successful judge evaluations"},
            )
        
        # Get judge weights
        judge_weights = {
            j.model: j.weight
            for j in self.judges
        }
        
        # Normalize weights for successful judges
        total_weight = sum(judge_weights.get(r.judge_model, 0) for r in successful)
        normalized_weights = {
            r.judge_model: judge_weights.get(r.judge_model, 0) / total_weight
            for r in successful
        }
        
        # Aggregate dimension scores
        dimension_scores: Dict[str, float] = {}
        dimension_details: Dict[str, Dict[str, Any]] = {}
        
        for dim_id in EVALUATION_DIMENSIONS.keys():
            scores = []
            reasonings = []
            
            for result in successful:
                if dim_id in result.dimensions:
                    weight = normalized_weights[result.judge_model]
                    dim_score = result.dimensions[dim_id]
                    scores.append((dim_score.score, weight))
                    reasonings.append({
                        "judge": result.judge_model,
                        "score": dim_score.score,
                        "reasoning": dim_score.reasoning,
                    })
            
            if scores:
                weighted_avg = sum(s * w for s, w in scores)
                dimension_scores[dim_id] = weighted_avg
                
                # Calculate score variance for disagreement
                raw_scores = [s for s, w in scores]
                variance = sum((s - weighted_avg) ** 2 for s in raw_scores) / len(raw_scores) if len(raw_scores) > 1 else 0
                
                dimension_details[dim_id] = {
                    "weighted_score": weighted_avg,
                    "variance": variance,
                    "individual_scores": reasonings,
                }
        
        # Calculate overall score
        overall = sum(
            dimension_scores.get(dim_id, 3) * config["weight"]
            for dim_id, config in EVALUATION_DIMENSIONS.items()
        )
        
        # Calculate agreement score (inverse of average variance)
        variances = [d.get("variance", 0) for d in dimension_details.values()]
        avg_variance = sum(variances) / len(variances) if variances else 0
        agreement = max(0, 1 - (avg_variance / 4))  # Normalize: max variance is 4 (range 1-5)
        
        # Calculate confidence
        confidence = (len(successful) / len(self.judges)) * agreement
        
        return EnsembleScore(
            overall=overall,
            dimensions=dimension_scores,
            dimension_details=dimension_details,
            judge_results=results,
            agreement_score=agreement,
            confidence=confidence,
            audience=audience,
            metadata={
                "successful_judges": len(successful),
                "total_judges": len(results),
                "total_weight": total_weight,
            },
        )
    
    def calibrate_judges(
        self,
        calibration_set: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calibrate judge weights using labeled data.
        
        Args:
            calibration_set: List of items with 'original', 'explanation',
                           'audience', and 'ground_truth_score'
                           
        Returns:
            Updated bias corrections per judge
        """
        logger.info(f"Calibrating judges with {len(calibration_set)} items")
        
        # Collect scores from each judge on calibration set
        judge_errors: Dict[str, List[float]] = {j.model: [] for j in self.judges}
        
        for item in calibration_set:
            ground_truth = item["ground_truth_score"]
            
            # Build prompt
            messages = self.rubric_builder.build_evaluation_prompt(
                item["original"],
                item["explanation"],
                item["audience"],
            )
            
            # Get each judge's score
            for judge in self.judges:
                try:
                    result = self._evaluate_single(judge, messages)
                    if result.success:
                        error = result.overall_score - ground_truth
                        judge_errors[judge.model].append(error)
                except Exception as e:
                    logger.warning(f"Calibration error for {judge.model}: {e}")
        
        # Calculate bias (mean error) for each judge
        for model, errors in judge_errors.items():
            if errors:
                bias = sum(errors) / len(errors)
                self.judge_biases[model] = bias
                logger.info(f"Judge {model} bias: {bias:.3f}")
        
        return self.judge_biases
    
    def get_disagreement_analysis(
        self,
        score: EnsembleScore
    ) -> Dict[str, Any]:
        """Analyze disagreement between judges.
        
        Args:
            score: Ensemble score to analyze
            
        Returns:
            Disagreement analysis
        """
        analysis = {
            "overall_agreement": score.agreement_score,
            "high_disagreement_dimensions": [],
            "judge_outliers": [],
        }
        
        # Find dimensions with high disagreement
        for dim_id, details in score.dimension_details.items():
            if details.get("variance", 0) > 0.5:  # More than 0.5 variance
                analysis["high_disagreement_dimensions"].append({
                    "dimension": dim_id,
                    "variance": details["variance"],
                    "scores": [s["score"] for s in details.get("individual_scores", [])],
                })
        
        # Find judge outliers
        for result in score.judge_results:
            if not result.success:
                continue
            
            deviation = abs(result.overall_score - score.overall)
            if deviation > 1.0:  # More than 1 point deviation
                analysis["judge_outliers"].append({
                    "judge": result.judge_model,
                    "score": result.overall_score,
                    "deviation": deviation,
                })
        
        return analysis


def create_single_judge(model: str = "gpt-5.1") -> EnsembleLLMJudge:
    """Create a single-model judge for faster evaluation.
    
    Args:
        model: Model to use
        
    Returns:
        EnsembleLLMJudge configured with single judge
    """
    return EnsembleLLMJudge(
        judges=[JudgeConfig(model=model, provider="openai", weight=1.0)],
        parallel=False,
        min_judges=1,
    )


def create_fast_ensemble() -> EnsembleLLMJudge:
    """Create a fast ensemble with efficient models.
    
    Returns:
        EnsembleLLMJudge with fast model configuration
    """
    return EnsembleLLMJudge(
        judges=[
            JudgeConfig(model="gpt-4o", provider="openai", weight=0.40),
            JudgeConfig(model="claude-haiku-4.5", provider="anthropic", weight=0.35),
            JudgeConfig(model="gemini-3-flash", provider="google", weight=0.25),
        ],
        parallel=True,
        min_judges=2,
    )


def create_full_ensemble() -> EnsembleLLMJudge:
    """Create full ensemble with all top-tier judges.
    
    Returns:
        EnsembleLLMJudge with comprehensive configuration
    """
    return EnsembleLLMJudge()  # Uses DEFAULT_JUDGES

