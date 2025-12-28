"""MEQ-Bench 2.0: Comprehensive Medical Explanation Quality Benchmark.

A research-grade benchmark for evaluating audience-adaptive explanation
quality in medical LLMs, featuring:

- Curated clinical scenarios across 12+ medical specialties
- Sophisticated audience modeling with 8+ personas
- Medical knowledge grounding with UMLS/RxNorm integration
- Multimodal evaluation (text + medical images)
- Ensemble LLM-as-judge with late-2025 frontier models
- Comprehensive validation framework

Late 2025 Model Support:
    - OpenAI: GPT-5.2, GPT-5.1, GPT-5, GPT-4o
    - Anthropic: Claude Opus 4.5, Claude Sonnet 4.5, Claude Haiku 4.5
    - Google: Gemini 3 Ultra/Pro/Flash
    - Meta: Llama 4 Behemoth/Maverick/Scout
    - DeepSeek: DeepSeek-V3
    - Alibaba: Qwen3-Max
    - Amazon: Nova Pro/Omni

Example:
    ```python
    from src import MEQBench, EnsembleLLMJudge, UnifiedModelClient
    
    # Initialize benchmark
    bench = MEQBench()
    
    # Create model client
    client = UnifiedModelClient()
    
    # Generate explanations
    response = client.generate(
        model="gpt-5.1",
        messages=[{"role": "user", "content": "Explain diabetes for a patient"}],
    )
    
    # Evaluate with ensemble judge
    judge = EnsembleLLMJudge()
    score = judge.evaluate(
        original="Diabetes mellitus...",
        explanation=response.content,
        audience="patient",
    )
    
    print(f"Overall score: {score.overall}")
    ```
"""

__version__ = "2.0.0"
__author__ = "MEQ-Bench Team"

# Core data structures
from .data_schema import (
    MEQBenchItemV2,
    MedicalEntity,
    ClinicalContext,
    MultimodalContent,
    ReferenceExplanation,
    MedicalSpecialty,
    ComplexityLevel,
    DatasetSource,
    save_benchmark_items_v2,
    load_benchmark_items_v2,
)

# Data loading
from .data_loaders_v2 import (
    load_pubmedqa,
    load_medmcqa,
    load_liveqa,
    load_healthsearchqa,
    load_mimic_discharge,
    load_vqa_rad,
    load_pathvqa,
    create_clinical_vignette,
    generate_sample_vignettes,
    DatasetCurator,
    detect_specialty,
    detect_safety_categories,
    calculate_complexity_v2,
)

# Audience personas
from .audience_personas import (
    AudiencePersona,
    AudienceType,
    HealthLiteracy,
    DetailLevel,
    AgeGroup,
    TerminologyExpectations,
    CommunicationPreferences,
    PersonaFactory,
    HealthLiteracyAssessor,
    PersonaBasedScorer,
    get_default_personas,
    get_comprehensive_personas,
)

# Knowledge grounding
from .knowledge_grounding import (
    MedicalKnowledgeGrounder,
    MedicalEntityExtractor,
    MedicalNLIVerifier,
    SemanticSimilarityScorer,
    UMLSClient,
    RxNormClient,
    GroundingScore,
    FactCheckResult,
    EntityType,
)

# Safety evaluation
from .safety_evaluator import (
    MedicalSafetyEvaluator,
    SafetyPatternDetector,
    DrugSafetyChecker,
    HarmClassifier,
    SafetyScore,
    SafetyIssue,
    SafetyCategory,
    SeverityLevel,
)

# Model clients
from .model_clients import (
    UnifiedModelClient,
    ModelConfig,
    GenerationResult,
    Provider,
    ModelTier,
    MODEL_REGISTRY,
    OpenAIClient,
    AnthropicClient,
    GoogleClient,
    DeepSeekClient,
    AlibabaClient,
    AmazonClient,
    LocalModelClient,
)

# Ensemble judge
from .ensemble_judge import (
    EnsembleLLMJudge,
    JudgeConfig,
    EnsembleScore,
    JudgeResult,
    DimensionScore,
    EvaluationRubricBuilder,
    EVALUATION_DIMENSIONS,
    AUDIENCE_RUBRIC_ADJUSTMENTS,
    create_single_judge,
    create_fast_ensemble,
    create_full_ensemble,
)

# Multimodal evaluation
from .multimodal_evaluator import (
    MultimodalMedicalEvaluator,
    ImageAnalyzer,
    VisualAlignmentScorer,
    ModalityTerminologyChecker,
    ImageContent,
    MultimodalScore,
    ImagingModality,
    MODALITY_CONFIG,
    get_supported_modalities,
    get_modality_info,
    get_multimodal_models,
)

# Validation
from .validation import (
    ValidationRunner,
    ValidationResult,
    ExpertAnnotation,
    SyntheticTestCase,
    SyntheticTestGenerator,
    CorrelationCalculator,
    InterRaterReliability,
    create_sample_annotations,
    save_validation_results,
    load_expert_annotations,
)

# Legacy compatibility - keep original exports working
from .benchmark import MEQBench, MEQBenchItem
from .evaluator import MEQBenchEvaluator

__all__ = [
    # Version
    "__version__",
    
    # Data Schema
    "MEQBenchItemV2",
    "MedicalEntity",
    "ClinicalContext",
    "MultimodalContent",
    "ReferenceExplanation",
    "MedicalSpecialty",
    "ComplexityLevel",
    "DatasetSource",
    "save_benchmark_items_v2",
    "load_benchmark_items_v2",
    
    # Data Loading
    "load_pubmedqa",
    "load_medmcqa",
    "load_liveqa",
    "load_healthsearchqa",
    "load_mimic_discharge",
    "load_vqa_rad",
    "load_pathvqa",
    "create_clinical_vignette",
    "generate_sample_vignettes",
    "DatasetCurator",
    "detect_specialty",
    "detect_safety_categories",
    "calculate_complexity_v2",
    
    # Audience Personas
    "AudiencePersona",
    "AudienceType",
    "HealthLiteracy",
    "DetailLevel",
    "AgeGroup",
    "TerminologyExpectations",
    "CommunicationPreferences",
    "PersonaFactory",
    "HealthLiteracyAssessor",
    "PersonaBasedScorer",
    "get_default_personas",
    "get_comprehensive_personas",
    
    # Knowledge Grounding
    "MedicalKnowledgeGrounder",
    "MedicalEntityExtractor",
    "MedicalNLIVerifier",
    "SemanticSimilarityScorer",
    "UMLSClient",
    "RxNormClient",
    "GroundingScore",
    "FactCheckResult",
    "EntityType",
    
    # Safety Evaluation
    "MedicalSafetyEvaluator",
    "SafetyPatternDetector",
    "DrugSafetyChecker",
    "HarmClassifier",
    "SafetyScore",
    "SafetyIssue",
    "SafetyCategory",
    "SeverityLevel",
    
    # Model Clients
    "UnifiedModelClient",
    "ModelConfig",
    "GenerationResult",
    "Provider",
    "ModelTier",
    "MODEL_REGISTRY",
    "OpenAIClient",
    "AnthropicClient",
    "GoogleClient",
    "DeepSeekClient",
    "AlibabaClient",
    "AmazonClient",
    "LocalModelClient",
    
    # Ensemble Judge
    "EnsembleLLMJudge",
    "JudgeConfig",
    "EnsembleScore",
    "JudgeResult",
    "DimensionScore",
    "EvaluationRubricBuilder",
    "EVALUATION_DIMENSIONS",
    "AUDIENCE_RUBRIC_ADJUSTMENTS",
    "create_single_judge",
    "create_fast_ensemble",
    "create_full_ensemble",
    
    # Multimodal Evaluation
    "MultimodalMedicalEvaluator",
    "ImageAnalyzer",
    "VisualAlignmentScorer",
    "ModalityTerminologyChecker",
    "ImageContent",
    "MultimodalScore",
    "ImagingModality",
    "MODALITY_CONFIG",
    "get_supported_modalities",
    "get_modality_info",
    "get_multimodal_models",
    
    # Validation
    "ValidationRunner",
    "ValidationResult",
    "ExpertAnnotation",
    "SyntheticTestCase",
    "SyntheticTestGenerator",
    "CorrelationCalculator",
    "InterRaterReliability",
    "create_sample_annotations",
    "save_validation_results",
    "load_expert_annotations",
    
    # Legacy
    "MEQBench",
    "MEQBenchItem",
    "MEQBenchEvaluator",
]
