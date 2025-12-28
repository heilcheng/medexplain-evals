Public Leaderboard
==================

MedExplain-Evals includes a comprehensive leaderboard system that generates static HTML pages displaying model performance across different audiences and complexity levels.

Overview
--------

The leaderboard system processes evaluation results from multiple models and creates an interactive HTML dashboard featuring:

* **Overall Model Rankings**: Comprehensive performance comparison
* **Audience-Specific Performance**: Breakdown by physician, nurse, patient, and caregiver
* **Complexity-Level Analysis**: Performance across basic, intermediate, and advanced content
* **Interactive Visualizations**: Charts and graphs for performance analysis

Quick Start
-----------

Generate a leaderboard from evaluation results:

.. code-block:: bash

   # Basic usage
   python -m src.leaderboard --input results/ --output docs/index.html

   # With custom options
   python -m src.leaderboard \
       --input evaluation_results/ \
       --output leaderboard.html \
       --title "Custom MedExplain-Evals Results" \
       --verbose

Command Line Interface
----------------------

.. list-table:: Leaderboard CLI Options
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--input PATH``
     - Directory containing JSON evaluation result files (required)
   * - ``--output PATH``
     - Output path for HTML leaderboard (default: docs/index.html)
   * - ``--title TEXT``
     - Custom title for the leaderboard page
   * - ``--verbose``
     - Enable verbose logging during generation

Input Data Format
-----------------

The leaderboard expects JSON files containing evaluation results in the following format:

.. code-block:: json

   {
     "model_name": "GPT-4",
     "total_items": 1000,
     "audience_scores": {
       "physician": [0.85, 0.90, 0.88, ...],
       "nurse": [0.82, 0.85, 0.83, ...],
       "patient": [0.75, 0.80, 0.78, ...],
       "caregiver": [0.80, 0.82, 0.81, ...]
     },
     "complexity_scores": {
       "basic": [0.85, 0.90, ...],
       "intermediate": [0.80, 0.85, ...], 
       "advanced": [0.75, 0.80, ...]
     },
     "detailed_results": [...],
     "summary": {
       "overall_mean": 0.82,
       "physician_mean": 0.88,
       "nurse_mean": 0.83,
       "patient_mean": 0.78,
       "caregiver_mean": 0.81
     }
   }

Programmatic Usage
------------------

Basic Leaderboard Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.leaderboard import LeaderboardGenerator
   from pathlib import Path

   # Initialize generator
   generator = LeaderboardGenerator()

   # Load results from directory
   results_dir = Path("evaluation_results/")
   generator.load_results(results_dir)

   # Generate HTML leaderboard
   output_path = Path("docs/leaderboard.html")
   generator.generate_html(output_path)

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   # Get leaderboard statistics
   stats = generator.calculate_leaderboard_stats()
   print(f"Total models: {stats['total_models']}")
   print(f"Total evaluations: {stats['total_evaluations']}")
   print(f"Best score: {stats['best_score']:.3f}")

   # Get model rankings
   ranked_models = generator.rank_models()
   for model in ranked_models[:3]:  # Top 3
       print(f"{model['rank']}. {model['model_name']}: {model['overall_score']:.3f}")

   # Get audience-specific breakdowns
   audience_breakdown = generator.generate_audience_breakdown(ranked_models)
   for audience, models in audience_breakdown.items():
       print(f"\n{audience.title()} Rankings:")
       for model in models[:3]:
           print(f"  {model['rank']}. {model['model_name']}: {model['score']:.3f}")

Leaderboard Features
--------------------

Overall Rankings
~~~~~~~~~~~~~~~~

The main leaderboard table displays:

* Model rankings by overall performance
* Total items evaluated per model
* Audience-specific average scores
* Interactive sorting and filtering

.. image:: _static/leaderboard_overall.png
   :alt: Overall Rankings Table
   :width: 800px

**Ranking Highlights:**

* 🥇 **1st Place**: Gold highlighting with special styling
* 🥈 **2nd Place**: Silver highlighting  
* 🥉 **3rd Place**: Bronze highlighting

Audience-Specific Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Dedicated sections for each audience showing:

* Rankings specific to that audience type
* Performance differences across audiences
* Top performers for each professional group

**Audience Categories:**

* **Physician**: Technical medical explanations
* **Nurse**: Clinical care and monitoring focus
* **Patient**: Simple, empathetic communication
* **Caregiver**: Practical instructions and warnings

Complexity-Level Breakdown
~~~~~~~~~~~~~~~~~~~~~~~~~~

Analysis by content difficulty:

* **Basic**: Elementary/middle school reading level
* **Intermediate**: High school reading level
* **Advanced**: College/professional reading level

This helps identify models that excel at different complexity levels.

Interactive Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~

The leaderboard includes Chart.js-powered visualizations:

**Performance Comparison Chart**
   Bar chart showing overall scores for top models

**Audience Performance Radar**
   Radar chart displaying average performance across all audiences

.. code-block:: javascript

   // Example chart configuration (automatically generated)
   {
     type: 'bar',
     data: {
       labels: ['GPT-4', 'Claude-3', 'PaLM-2', ...],
       datasets: [{
         label: 'Overall Score',
         data: [0.85, 0.82, 0.79, ...],
         backgroundColor: 'rgba(59, 130, 246, 0.8)'
       }]
     }
   }

Responsive Design
-----------------

The leaderboard is fully responsive and works on:

* **Desktop**: Full feature set with side-by-side comparisons
* **Tablet**: Optimized layout with collapsible sections  
* **Mobile**: Touch-friendly interface with stacked content

CSS Grid and Flexbox ensure optimal viewing across all devices.

Customization
-------------

Styling
~~~~~~~

Customize the leaderboard appearance by modifying the CSS:

.. code-block:: python

   class CustomLeaderboardGenerator(LeaderboardGenerator):
       def _get_css_styles(self):
           # Override with custom styles
           return custom_css_content

Color Schemes
~~~~~~~~~~~~~

The default color scheme uses:

* **Primary**: Blue (#3b82f6) for highlights and buttons  
* **Success**: Green (#059669) for scores and positive indicators
* **Warning**: Gold (#ffd700) for first place highlighting
* **Neutral**: Gray scale for general content

Branding
~~~~~~~~

Customize titles, logos, and contact information:

.. code-block:: python

   # Modify the HTML template generation
   def _generate_html_template(self, ...):
       return f"""
       <header>
           <h1>🏆 {custom_title}</h1>
           <img src="your_logo.png" alt="Logo">
       </header>
       ...
       """

Performance Optimization
------------------------

Large Dataset Handling
~~~~~~~~~~~~~~~~~~~~~~

For leaderboards with many models:

.. code-block:: python

   # Pagination for large leaderboards
   def generate_paginated_leaderboard(models, page_size=50):
       pages = [models[i:i+page_size] for i in range(0, len(models), page_size)]
       return pages

   # Top-N filtering
   top_models = ranked_models[:20]  # Show only top 20

Caching
~~~~~~~

Cache expensive calculations:

.. code-block:: python

   import functools

   @functools.lru_cache(maxsize=100)
   def cached_statistics_calculation(self, data_hash):
       return self.calculate_leaderboard_stats()

CDN Assets
~~~~~~~~~~

For better performance, load external assets from CDN:

.. code-block:: html

   <!-- Chart.js from CDN -->
   <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

   <!-- Custom fonts -->
   <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

Deployment
----------

Static Hosting
~~~~~~~~~~~~~~

The generated HTML is completely self-contained and can be hosted on:

* **GitHub Pages**: Perfect for open source projects
* **Netlify**: Easy deployment with automatic builds
* **AWS S3**: Scalable static hosting
* **Apache/Nginx**: Traditional web servers

GitHub Pages Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # .github/workflows/leaderboard.yml
   name: Update Leaderboard
   on:
     schedule:
       - cron: '0 0 * * *'  # Daily updates
     workflow_dispatch:

   jobs:
     update-leaderboard:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Setup Python
           uses: actions/setup-python@v2
           with:
             python-version: '3.9'
         - name: Install dependencies
           run: pip install -r requirements.txt
         - name: Generate leaderboard
           run: python -m src.leaderboard --input results/ --output docs/index.html
         - name: Deploy to GitHub Pages
           uses: peaceiris/actions-gh-pages@v3
           with:
             github_token: ${{ secrets.GITHUB_TOKEN }}
             publish_dir: ./docs

Automated Updates
~~~~~~~~~~~~~~~~~

Set up automated leaderboard updates:

.. code-block:: bash

   #!/bin/bash
   # update_leaderboard.sh

   # Download latest results
   rsync -av results_server:/path/to/results/ ./results/

   # Regenerate leaderboard
   python -m src.leaderboard \
       --input results/ \
       --output docs/index.html \
       --verbose

   # Deploy to hosting
   aws s3 sync docs/ s3://your-bucket/ --delete

SEO and Analytics
-----------------

Search Engine Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The generated HTML includes SEO-friendly features:

.. code-block:: html

   <head>
       <title>MedExplain-Evals Leaderboard - Medical LLM Evaluation Results</title>
       <meta name="description" content="Comprehensive evaluation results for medical language models on audience-adaptive explanation quality.">
       <meta name="keywords" content="medical AI, language models, evaluation, leaderboard">
       <meta property="og:title" content="MedExplain-Evals Leaderboard">
       <meta property="og:description" content="Medical LLM evaluation results">
   </head>

Analytics Integration
~~~~~~~~~~~~~~~~~~~~

Add analytics tracking:

.. code-block:: python

   def add_analytics_tracking(self, html_content, tracking_id):
       analytics_code = f"""
       <!-- Google Analytics -->
       <script async src="https://www.googletagmanager.com/gtag/js?id={tracking_id}"></script>
       <script>
         window.dataLayer = window.dataLayer || [];
         function gtag(){{dataLayer.push(arguments);}}
         gtag('js', new Date());
         gtag('config', '{tracking_id}');
       </script>
       """
       return html_content.replace('</head>', analytics_code + '</head>')

API Reference
-------------

.. automodule:: src.leaderboard
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Complete Evaluation to Leaderboard Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.benchmark import MedExplain
   from src.leaderboard import LeaderboardGenerator
   from pathlib import Path

   # 1. Run evaluations for multiple models
   models = ['gpt-4', 'claude-3-opus', 'llama-2-70b']
   bench = MedExplain()

   for model_name in models:
       model_func = get_model_function(model_name)  # Your model interface
       results = bench.evaluate_model(model_func, max_items=1000)
       
       # Save individual results
       output_path = f"results/{model_name}_evaluation.json"
       bench.save_results(results, output_path)

   # 2. Generate leaderboard from all results
   generator = LeaderboardGenerator()
   generator.load_results(Path("results/"))
   generator.generate_html(Path("docs/index.html"))

   print("✅ Leaderboard generated successfully!")

Multi-Language Support
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiLanguageLeaderboard(LeaderboardGenerator):
       def __init__(self, language='en'):
           super().__init__()
           self.language = language
           self.translations = self._load_translations()

       def _load_translations(self):
           # Load language-specific strings
           return {
               'en': {'title': 'MedExplain-Evals Leaderboard', ...},
               'es': {'title': 'Tabla de Clasificación MedExplain-Evals', ...},
               # ... other languages
           }

Best Practices
--------------

1. **Regular Updates**: Update leaderboards regularly as new results become available

2. **Data Validation**: Validate result files before generating leaderboards

   .. code-block:: python

      def validate_results_directory(results_dir):
          required_fields = ['model_name', 'total_items', 'audience_scores', 'summary']
          for file_path in results_dir.glob("*.json"):
              with open(file_path) as f:
                  data = json.load(f)
              assert all(field in data for field in required_fields)

3. **Version Control**: Track leaderboard versions and source data

4. **Accessibility**: Ensure leaderboards are accessible to users with disabilities

5. **Mobile Testing**: Test leaderboard display across different screen sizes

6. **Performance Monitoring**: Monitor page load times and optimize as needed