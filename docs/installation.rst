Installation
============

Requirements
------------

* Python 3.8 or higher
* pip package manager
* Git (for development installation)

Basic Installation
------------------

Install MedExplain-Evals using pip:

.. code-block:: bash

   git clone https://github.com/heilcheng/MedExplain-Evals.git
   cd MedExplain-Evals
   pip install .

This command uses the setup.py file to install the package and its dependencies correctly. The setup.py file automatically handles all core dependencies defined in the install_requires section.

Development Installation
------------------------

For development, install in editable mode with additional dependencies:

.. code-block:: bash

   git clone https://github.com/heilcheng/MedExplain-Evals.git
   cd MedExplain-Evals
   pip install -e .[dev]

This installs additional tools for development:

* pytest for testing
* black for code formatting
* flake8 for linting
* mypy for type checking

Optional Dependencies
---------------------

Some features require additional packages:

**Machine Learning Components:**

.. code-block:: bash

   pip install torch transformers sentence-transformers

**Natural Language Processing:**

.. code-block:: bash

   pip install spacy scispacy
   python -m spacy download en_core_web_sm

**Documentation Building:**

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme myst-parser

Configuration
-------------

1. **Create logs directory:**

.. code-block:: bash

   mkdir logs

2. **Set up API keys** (required for LLM-as-a-judge evaluation):

.. code-block:: bash

   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"

3. **Verify installation:**

.. code-block:: bash

   python -c "from src.benchmark import MedExplain; print('Installation successful!')"

Docker Installation (Optional)
-------------------------------

For a containerized environment:

.. code-block:: bash

   # Build the Docker image
   docker build -t medexplain-evals .
   
   # Run the container
   docker run -it medexplain-evals

Troubleshooting
---------------

**Common Issues:**

1. **Import errors**: Ensure all dependencies are installed
2. **API key errors**: Set environment variables correctly
3. **Permission errors**: Use virtual environment or user installation

**Getting Help:**

* Check the GitHub Issues page
* Review the documentation
* Contact the development team