API Reference
=============

Complete API documentation for MedExplain-Evals, the benchmark for evaluating audience-adaptive medical explanation quality in LLMs.

.. contents:: Table of Contents
   :local:
   :depth: 1

Core Modules
------------

.. toctree::
   :maxdepth: 2

   model_clients
   ensemble_judge
   audience_personas
   knowledge_grounding
   safety_evaluator

Quick Reference
---------------

Model Clients
~~~~~~~~~~~~~

The entry point for interacting with LLM providers.

.. code-block:: python

   from src import UnifiedModelClient

   client = UnifiedModelClient()
   result = client.generate(
       model="gpt-5.2",
       messages=[{"role": "user", "content": "Explain diabetes"}]
   )

See :doc:`model_clients` for full documentation.

Ensemble Judge
~~~~~~~~~~~~~~

Multi-model evaluation with weighted ensemble scoring.

.. code-block:: python

   from src import EnsembleLLMJudge

   judge = EnsembleLLMJudge()
   score = judge.evaluate(
       original_content="Type 2 DM with HbA1c 8.5%...",
       explanation="You have high blood sugar...",
       audience="patient"
   )
   print(f"Overall: {score.overall}/5.0")

See :doc:`ensemble_judge` for full documentation.

Audience Personas
~~~~~~~~~~~~~~~~~

Sophisticated audience modeling with 11 predefined personas.

.. code-block:: python

   from src import PersonaFactory

   persona = PersonaFactory.get_predefined_persona("patient_low_literacy")
   print(persona.health_literacy)  # "low"
   print(persona.reading_level_target)  # (6, 10)

See :doc:`audience_personas` for full documentation.

Knowledge Grounding
~~~~~~~~~~~~~~~~~~~

Medical knowledge base integration for factuality verification.

.. code-block:: python

   from src import MedicalKnowledgeGrounder

   grounder = MedicalKnowledgeGrounder()
   score = grounder.ground_explanation(
       original="Diabetes mellitus type 2...",
       explanation="You have high blood sugar..."
   )
   print(f"Factual accuracy: {score.factual_accuracy}")

See :doc:`knowledge_grounding` for full documentation.

Safety Evaluation
~~~~~~~~~~~~~~~~~

Comprehensive medical safety assessment.

.. code-block:: python

   from src import MedicalSafetyEvaluator

   evaluator = MedicalSafetyEvaluator()
   score = evaluator.evaluate(
       explanation="Stop your medication...",
       medical_context="Cardiovascular"
   )
   print(f"Passed: {score.passed}")

See :doc:`safety_evaluator` for full documentation.

Package Structure
-----------------

.. code-block:: text

   src/
   ├── __init__.py           # Package exports
   ├── model_clients.py      # LLM provider clients
   ├── ensemble_judge.py     # Multi-model judge ensemble
   ├── audience_personas.py  # Audience modeling
   ├── knowledge_grounding.py # Medical KB integration
   ├── safety_evaluator.py   # Safety assessment
   ├── evaluator.py          # Legacy evaluator
   ├── benchmark.py          # Benchmark runner
   ├── data_loaders_v2.py    # Dataset loading
   ├── validation.py         # Validation framework
   └── multimodal_evaluator.py # Image + text evaluation

Environment Variables
---------------------

Required API keys for full functionality:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``OPENAI_API_KEY``
     - OpenAI API key for GPT models
   * - ``ANTHROPIC_API_KEY``
     - Anthropic API key for Claude models
   * - ``GOOGLE_API_KEY``
     - Google AI API key for Gemini models
   * - ``DEEPSEEK_API_KEY``
     - DeepSeek API key
   * - ``UMLS_API_KEY``
     - UMLS Terminology Services key
   * - ``AWS_ACCESS_KEY_ID``
     - AWS credentials for Amazon Nova
   * - ``AWS_SECRET_ACCESS_KEY``
     - AWS secret key

Common Imports
--------------

.. code-block:: python

   # Core functionality
   from src import (
       # Model clients
       UnifiedModelClient,
       GenerationResult,
       
       # Ensemble judge
       EnsembleLLMJudge,
       EnsembleScore,
       JudgeConfig,
       
       # Personas
       PersonaFactory,
       AudiencePersona,
       AudienceType,
       
       # Knowledge grounding
       MedicalKnowledgeGrounder,
       MedicalEntityExtractor,
       UMLSClient,
       RxNormClient,
       
       # Safety
       MedicalSafetyEvaluator,
       SafetyScore,
       DrugSafetyChecker,
   )