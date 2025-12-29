MedExplain-Evals Documentation
=======================

Welcome to MedExplain-Evals, a resource-efficient benchmark for evaluating audience-adaptive explanation quality in medical Large Language Models.

This project is developed as part of `Google Summer of Code 2025 <https://summerofcode.withgoogle.com/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Core Functionality:

   data_loading
   evaluation_metrics
   leaderboard

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index

Overview
--------

MedExplain-Evals addresses a critical gap in medical AI evaluation by providing the first benchmark specifically designed to assess an LLM's ability to generate audience-adaptive medical explanations for four key stakeholders:

- **Physicians** - Technical, evidence-based explanations
- **Nurses** - Practical care implications and monitoring
- **Patients** - Simple, empathetic, jargon-free language
- **Caregivers** - Concrete tasks and warning signs

Key Features
------------

- Novel evaluation framework for audience-adaptive medical explanations
- Support for MedQA-USMLE, iCliniq, and Cochrane Reviews datasets
- Advanced safety metrics including contradiction and hallucination detection
- Automated complexity stratification using Flesch-Kincaid Grade Level
- Interactive HTML leaderboards for result visualization
- Multi-dimensional scoring with LLM-as-a-judge paradigm
- Optimized for open-weight models on consumer hardware

Quick Start
-----------

.. code-block:: bash

   pip install -r requirements.txt

.. code-block:: python

   from src.benchmark import MedExplain
   from src.evaluator import MedExplainEvaluator

   # Initialize benchmark
   bench = MedExplain()
   
   # Generate audience-adaptive explanations
   explanations = bench.generate_explanations(medical_content, model)
   
   # Evaluate explanations
   evaluator = MedExplainEvaluator()
   scores = evaluator.evaluate_all_audiences(explanations)

Architecture
------------

MedExplain-Evals is built with SOLID principles:

- Strategy Pattern for audience-specific scoring
- Dependency Injection for flexible component management
- Configuration-driven design with YAML configuration
- Comprehensive logging for debugging and monitoring

Getting Help
------------

**Documentation**

- Primary documentation: This comprehensive guide covers installation, usage, and advanced topics
- API Reference: Detailed function and class documentation with examples
- Quickstart Guide: :doc:`quickstart`
- Installation Guide: :doc:`installation`

**Support Channels**

- Bug Reports: `GitHub Issues <https://github.com/heilcheng/MedExplain-Evals/issues>`_
- Questions: `GitHub Discussions <https://github.com/heilcheng/MedExplain-Evals/discussions>`_

**Troubleshooting**

.. code-block:: bash

   # Verify installation
   python -c "import src; print('MedExplain-Evals is working')"
   
   # Run basic test
   python run_benchmark.py --model_name dummy --max_items 2

Contributing
------------

We welcome contributions:

- Code contributions via Pull Requests
- Bug reports and feature requests via Issues
- Documentation improvements
- Research collaborations

See our `Contributing Guidelines <https://github.com/heilcheng/MedExplain-Evals/blob/main/CONTRIBUTING.md>`_.

Citation
--------

.. code-block:: bibtex

   @software{medexplain-evals-2025,
     title={MedExplain-Evals: A Resource-Efficient Benchmark for Evaluating Audience-Adaptive Explanation Quality in Medical Large Language Models},
     author={Cheng Hei Lam},
     year={2025},
     url={https://github.com/heilcheng/medexplain-evals}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`