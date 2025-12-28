Quick Start Guide
=================

This guide will help you get started with MedExplain-Evals quickly.

Basic Usage
-----------

1. **Import the main classes:**

.. code-block:: python

   from src.benchmark import MedExplain
   from src.evaluator import MedExplainEvaluator
   from src.config import config

2. **Initialize the benchmark:**

.. code-block:: python

   # Initialize with default configuration
   bench = MedExplain()
   
   # Initialize evaluator
   evaluator = MedExplainEvaluator()

3. **Define your model function:**

.. code-block:: python

   def your_model_function(prompt: str) -> str:
       """
       Your LLM function that takes a prompt and returns a response.
       This should generate explanations for all four audiences.
       """
       # Call your LLM here
       return model_response

4. **Generate and evaluate explanations:**

.. code-block:: python

   # Sample medical content
   medical_content = "Hypertension is a condition where blood pressure is elevated..."
   
   # Generate audience-adaptive explanations
   explanations = bench.generate_explanations(medical_content, your_model_function)
   
   # Evaluate the explanations
   results = evaluator.evaluate_all_audiences(medical_content, explanations)
   
   # Print results
   for audience, score in results.items():
       print(f"{audience}: {score.overall:.3f}")

Working with Sample Data
------------------------

MedExplain-Evals includes sample data for testing:

.. code-block:: python

   # Create sample dataset
   sample_items = bench.create_sample_dataset()
   
   # Add to benchmark
   for item in sample_items:
       bench.add_benchmark_item(item)
   
   # Run evaluation on sample data
   results = bench.evaluate_model(your_model_function, max_items=3)

Configuration
-------------

Customize behavior using the configuration system:

.. code-block:: python

   from src.config import config
   
   # View current configuration
   print(config.get('llm_judge.default_model'))
   
   # Get audience list
   audiences = config.get_audiences()
   print(audiences)  # ['physician', 'nurse', 'patient', 'caregiver']

Custom Evaluation Components
----------------------------

Use dependency injection for custom components:

.. code-block:: python

   from src.evaluator import MedExplainEvaluator, LLMJudge
   from src.strategies import StrategyFactory
   
   # Custom LLM judge with different model
   custom_judge = LLMJudge(model="gpt-4o")
   
   # Initialize evaluator with custom components
   evaluator = MedExplainEvaluator(llm_judge=custom_judge)

Batch Evaluation
----------------

Evaluate multiple items efficiently:

.. code-block:: python

   # Load data
   bench = MedExplain(data_path="data/")
   
   # Run full evaluation
   results = bench.evaluate_model(
       your_model_function,
       max_items=100  # Limit for testing
   )
   
   # Save results
   bench.save_results(results, "evaluation_results.json")

Error Handling
--------------

MedExplain-Evals includes comprehensive error handling:

.. code-block:: python

   from src.evaluator import EvaluationError
   
   try:
       results = evaluator.evaluate_explanation(
           original="medical content",
           generated="explanation",
           audience="patient"
       )
   except EvaluationError as e:
       print(f"Evaluation failed: {e}")

Logging
-------

Enable detailed logging:

.. code-block:: python

   import logging
   from src.config import config
   
   # Set up logging from configuration
   config.setup_logging()
   
   # Set log level
   logging.getLogger('medexplain').setLevel(logging.DEBUG)

Next Steps
----------

* Read the :doc:`api/index` for detailed API documentation
* Explore :doc:`examples` for more use cases
* Learn about the :doc:`evaluation` methodology
* Contribute to the project following :doc:`contributing` guidelines