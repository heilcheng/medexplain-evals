Ensemble Judge
==============

The ensemble judge module implements a multi-model LLM-as-Judge framework for robust, calibrated evaluation of medical explanations.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

MedExplain-Evals uses an ensemble of frontier LLMs to evaluate medical explanations, providing more reliable and calibrated scores than single-model evaluation.

**Default Ensemble Composition:**

- **Primary Judges (85% weight):**
  - GPT-5.2 (30%)
  - Claude Opus 4.5 (30%)
  - Gemini 3 Pro (25%)

- **Secondary Judges (15% weight):**
  - DeepSeek-V3.2 (10%)
  - Qwen3-Max (5%)

Quick Start
-----------

.. code-block:: python

   from src import EnsembleLLMJudge, UnifiedModelClient

   # Initialize
   client = UnifiedModelClient()
   judge = EnsembleLLMJudge(client)

   # Evaluate an explanation
   score = judge.evaluate(
       original_content="Patient has Type 2 DM with HbA1c 8.5%...",
       explanation="You have high blood sugar. This means...",
       audience="patient"
   )

   print(f"Overall: {score.overall:.2f}/5.0")
   print(f"Agreement: {score.agreement_score:.2f}")
   print(f"Dimensions: {score.dimensions}")

Core Classes
------------

EnsembleLLMJudge
~~~~~~~~~~~~~~~~

The main evaluation class that orchestrates multiple judge models.

.. code-block:: python

   class EnsembleLLMJudge:
       """Multi-model LLM judge ensemble for robust evaluation.
       
       Uses weighted ensemble of top reasoning models for evaluation:
       - Primary: GPT-5.2, Claude Opus 4.5, Gemini 3 Pro (85%)
       - Secondary: DeepSeek-V3.2, Qwen3-Max (15%)
       """
       
       def __init__(
           self,
           client: Optional[UnifiedModelClient] = None,
           judges: Optional[List[JudgeConfig]] = None,
           parallel: bool = True,
           min_judges: int = 2
       ):
           """Initialize the ensemble judge.
           
           Args:
               client: Model client for API calls
               judges: Custom judge configurations
               parallel: Execute judges in parallel
               min_judges: Minimum judges required for valid evaluation
           """

       def evaluate(
           self,
           original_content: str,
           explanation: str,
           audience: str,
           persona: Optional[AudiencePersona] = None
       ) -> EnsembleScore:
           """Evaluate an explanation with the ensemble.
           
           Args:
               original_content: Original medical content
               explanation: Generated explanation to evaluate
               audience: Target audience type
               persona: Optional detailed persona configuration
               
           Returns:
               EnsembleScore with overall and dimension scores
           """

       def evaluate_batch(
           self,
           items: List[Dict[str, Any]],
           max_workers: int = 4
       ) -> List[EnsembleScore]:
           """Evaluate multiple explanations in parallel."""

**Usage Examples:**

.. code-block:: python

   from src import EnsembleLLMJudge, PersonaFactory

   judge = EnsembleLLMJudge()

   # Basic evaluation
   score = judge.evaluate(
       original_content="Acute myocardial infarction...",
       explanation="You had a heart attack...",
       audience="patient"
   )

   # With detailed persona
   persona = PersonaFactory.get_predefined_persona("patient_low_literacy")
   score = judge.evaluate(
       original_content="Acute myocardial infarction...",
       explanation="Your heart muscle was hurt...",
       audience="patient",
       persona=persona
   )

   # Analyze disagreement
   analysis = judge.get_disagreement_analysis(score)
   print(analysis)

JudgeConfig
~~~~~~~~~~~

Configuration for individual judges in the ensemble.

.. code-block:: python

   @dataclass
   class JudgeConfig:
       """Configuration for an individual judge in the ensemble."""
       model: str           # Model identifier (e.g., "gpt-5.2")
       provider: str        # Provider name (e.g., "openai")
       weight: float        # Weight in ensemble (0.0-1.0)
       enabled: bool = True
       temperature: float = 0.1
       max_retries: int = 3

EnsembleScore
~~~~~~~~~~~~~

The comprehensive evaluation result.

.. code-block:: python

   @dataclass
   class EnsembleScore:
       """Final ensemble evaluation score."""
       overall: float                    # Weighted overall score (1-5)
       dimensions: Dict[str, float]      # Per-dimension scores
       dimension_details: Dict[str, Dict[str, Any]]  # Detailed breakdown
       judge_results: List[JudgeResult]  # Individual judge outputs
       agreement_score: float            # Inter-judge agreement (0-1)
       confidence: float                 # Ensemble confidence (0-1)
       audience: str                     # Target audience
       metadata: Dict[str, Any]          # Additional metadata

DimensionScore
~~~~~~~~~~~~~~

Score for a single evaluation dimension.

.. code-block:: python

   @dataclass
   class DimensionScore:
       """Score for a single evaluation dimension."""
       dimension: str      # Dimension name
       score: float        # Score (1-5)
       reasoning: str      # Justification text
       confidence: float = 1.0

Evaluation Dimensions
---------------------

The ensemble evaluates explanations across six weighted dimensions:

.. list-table:: Evaluation Dimensions
   :header-rows: 1
   :widths: 30 10 60

   * - Dimension
     - Weight
     - Description
   * - Factual Accuracy
     - 25%
     - Clinical correctness and evidence alignment
   * - Terminological Appropriateness
     - 15%
     - Language complexity matching audience needs
   * - Explanatory Completeness
     - 20%
     - Comprehensive yet accessible coverage
   * - Actionability
     - 15%
     - Clear, practical guidance
   * - Safety
     - 15%
     - Appropriate warnings and harm avoidance
   * - Empathy & Tone
     - 10%
     - Audience-appropriate communication style

Factory Functions
-----------------

Convenience functions for creating judge configurations:

.. code-block:: python

   from src import create_single_judge, create_fast_ensemble, create_full_ensemble

   # Single model judge (for testing)
   judge = create_single_judge(model="gpt-5.2")

   # Fast 2-model ensemble
   judge = create_fast_ensemble()

   # Full 5-model ensemble (default)
   judge = create_full_ensemble()

EvaluationRubricBuilder
-----------------------

Builds G-Eval style evaluation prompts with audience-specific rubrics.

.. code-block:: python

   from src.ensemble_judge import EvaluationRubricBuilder

   builder = EvaluationRubricBuilder()
   
   # Build evaluation prompt
   messages = builder.build_evaluation_prompt(
       original_content="Medical content...",
       explanation="Explanation...",
       audience="patient",
       persona=persona  # Optional
   )

Custom Judge Configuration
--------------------------

Create a custom ensemble configuration:

.. code-block:: python

   from src import EnsembleLLMJudge, JudgeConfig

   custom_judges = [
       JudgeConfig(model="gpt-5.2", provider="openai", weight=0.5),
       JudgeConfig(model="claude-opus-4-5", provider="anthropic", weight=0.5),
   ]

   judge = EnsembleLLMJudge(judges=custom_judges)
