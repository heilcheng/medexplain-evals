API Reference
=============

This section contains the complete API documentation for MedExplain-Evals.

.. toctree::
   :maxdepth: 2

   benchmark
   evaluator
   strategies
   config

Core Modules
------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   src.benchmark
   src.evaluator
   src.strategies
   src.config
   src.prompt_templates

Main Classes
------------

Benchmark
~~~~~~~~~

.. autoclass:: src.benchmark.MedExplain
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.benchmark.MedExplainItem
   :members:
   :undoc-members:
   :show-inheritance:

Evaluator
~~~~~~~~~

.. autoclass:: src.evaluator.MedExplainEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.evaluator.EvaluationScore
   :members:
   :undoc-members:
   :show-inheritance:

Strategies
~~~~~~~~~~

.. autoclass:: src.strategies.AudienceStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.strategies.StrategyFactory
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. autoclass:: src.config.Config
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation Components
---------------------

Calculators
~~~~~~~~~~~

.. autoclass:: src.evaluator.ReadabilityCalculator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.evaluator.TerminologyCalculator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.evaluator.SafetyChecker
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.evaluator.CoverageAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.evaluator.LLMJudge
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. autoclass:: src.evaluator.EvaluationError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.config.ConfigurationError
   :members:
   :undoc-members:
   :show-inheritance: