Data Loading and Processing
===========================

MedExplain-Evals provides comprehensive data loading functionality for popular medical datasets, with automatic complexity stratification and standardized conversion to the MedExplain-Evals format.

Supported Datasets
------------------

The following medical datasets are currently supported:

* **MedQA-USMLE**: Medical question answering based on USMLE exam format
* **iCliniq**: Real clinical questions from patients with professional answers
* **Cochrane Reviews**: Evidence-based systematic reviews and meta-analyses

Each dataset loader handles the specific format and field mappings of its source data, converting everything to standardized :class:`~src.benchmark.MedExplainItem` objects.

Basic Usage
-----------

Load Individual Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.data_loaders import load_medqa_usmle, load_icliniq, load_cochrane_reviews

   # Load MedQA-USMLE dataset with automatic complexity stratification
   medqa_items = load_medqa_usmle(
       'data/medqa_usmle.json', 
       max_items=300,
       auto_complexity=True
   )

   # Load iCliniq dataset
   icliniq_items = load_icliniq(
       'data/icliniq.json',
       max_items=400,
       auto_complexity=True
   )

   # Load Cochrane Reviews
   cochrane_items = load_cochrane_reviews(
       'data/cochrane.json',
       max_items=300,
       auto_complexity=True
   )

Combine Multiple Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.data_loaders import save_benchmark_items

   # Combine all datasets
   all_items = medqa_items + icliniq_items + cochrane_items

   # Save as unified benchmark dataset
   save_benchmark_items(all_items, 'data/benchmark_items.json')

Complexity Stratification
--------------------------

MedExplain-Evals automatically categorizes content complexity using Flesch-Kincaid Grade Level scores:

* **Basic**: FK score ≤ 8 (elementary/middle school level)
* **Intermediate**: FK score 9-12 (high school level)  
* **Advanced**: FK score > 12 (college/professional level)

.. code-block:: python

   from src.data_loaders import calculate_complexity_level

   # Calculate complexity for any text
   text = "Hypertension is high blood pressure that can damage your heart."
   complexity = calculate_complexity_level(text)
   print(complexity)  # "basic"

   # More complex medical text
   complex_text = ("Pharmacokinetic interactions involving cytochrome P450 enzymes "
                  "can significantly alter therapeutic drug concentrations.")
   complexity = calculate_complexity_level(complex_text)
   print(complexity)  # "advanced"

Fallback Complexity Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the ``textstat`` library is unavailable, MedExplain-Evals uses a fallback method based on:

* Average sentence length
* Average syllables per word
* Medical terminology density

.. code-block:: python

   # The fallback method is automatically used when textstat is not available
   # No changes needed in your code - it's handled transparently

Data Processing Script
----------------------

MedExplain-Evals includes a comprehensive command-line script for processing and combining datasets:

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Process all three datasets with default settings
   python scripts/process_datasets.py \
       --medqa data/medqa_usmle.json \
       --icliniq data/icliniq.json \
       --cochrane data/cochrane.json \
       --output data/benchmark_items.json

Advanced Options
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Custom item limits per dataset
   python scripts/process_datasets.py \
       --medqa data/medqa_usmle.json \
       --icliniq data/icliniq.json \
       --cochrane data/cochrane.json \
       --output data/benchmark_items.json \
       --max-items 1500 \
       --medqa-items 600 \
       --icliniq-items 500 \
       --cochrane-items 400 \
       --balance-complexity \
       --validate \
       --stats \
       --verbose

Script Options
~~~~~~~~~~~~~~

.. list-table:: Command Line Options
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``--medqa PATH``
     - Path to MedQA-USMLE JSON file
   * - ``--icliniq PATH``
     - Path to iCliniq JSON file  
   * - ``--cochrane PATH``
     - Path to Cochrane Reviews JSON file
   * - ``--output PATH``
     - Output path for combined dataset (default: data/benchmark_items.json)
   * - ``--max-items N``
     - Maximum total items in final dataset (default: 1000)
   * - ``--medqa-items N``
     - Maximum items from MedQA-USMLE
   * - ``--icliniq-items N``
     - Maximum items from iCliniq
   * - ``--cochrane-items N``
     - Maximum items from Cochrane Reviews
   * - ``--auto-complexity``
     - Enable automatic complexity calculation (default: True)
   * - ``--no-auto-complexity``
     - Disable automatic complexity calculation
   * - ``--balance-complexity``
     - Balance dataset across complexity levels (default: True)
   * - ``--validate``
     - Validate final dataset and show report
   * - ``--stats``
     - Show detailed statistics about created dataset
   * - ``--seed N``
     - Random seed for reproducible dataset creation (default: 42)
   * - ``--verbose``
     - Enable verbose logging

Dataset Validation
------------------

The data processing includes comprehensive validation:

.. code-block:: python

   from scripts.process_datasets import validate_dataset

   # Validate any list of MedExplainItem objects
   validation_report = validate_dataset(items)

   if validation_report['valid']:
       print("✅ Dataset validation passed")
   else:
       print("❌ Dataset validation failed")
       for issue in validation_report['issues']:
           print(f"  Issue: {issue}")

   for warning in validation_report['warnings']:
       print(f"  Warning: {warning}")

Validation Checks
~~~~~~~~~~~~~~~~~

* **Duplicate IDs**: Ensures all item IDs are unique
* **Content Length**: Validates minimum content length (≥20 characters)
* **Complexity Distribution**: Warns if not all complexity levels are represented
* **Data Integrity**: Checks for valid field types and required fields

Dataset Statistics
------------------

Generate comprehensive statistics about your dataset:

.. code-block:: python

   from scripts.process_datasets import print_dataset_statistics

   # Print detailed statistics
   print_dataset_statistics(items)

The statistics include:

* Total item count
* Complexity level distribution (percentages)
* Source dataset distribution
* Content length statistics (min, max, average)

Custom Dataset Loading
----------------------

For datasets not directly supported, use the custom loader:

.. code-block:: python

   from src.data_loaders import load_custom_dataset

   # Define field mapping for your dataset
   field_mapping = {
       'q': 'question',          # Your field -> standard field
       'a': 'answer',
       'medical_text': 'medical_content',
       'item_id': 'id'
   }

   items = load_custom_dataset(
       'path/to/your/dataset.json',
       field_mapping=field_mapping,
       max_items=500,
       complexity_level='intermediate'  # Or use auto_complexity=True
   )

Error Handling
--------------

All data loaders include comprehensive error handling:

.. code-block:: python

   try:
       items = load_medqa_usmle('data/medqa.json')
   except FileNotFoundError:
       print("Dataset file not found")
   except json.JSONDecodeError:
       print("Invalid JSON format")
   except ValueError as e:
       print(f"Data validation error: {e}")

The loaders will:

* Skip invalid items with detailed logging
* Continue processing when individual items fail
* Provide informative error messages
* Return partial results when possible

Performance Considerations
--------------------------

For large datasets:

* Use ``max_items`` to limit memory usage during development
* Enable ``auto_complexity`` only when needed (adds processing time)
* Consider processing datasets separately and combining later
* Use the ``--verbose`` flag to monitor progress

.. code-block:: python

   # Process large dataset in chunks
   chunk_size = 1000
   all_items = []

   for i in range(0, total_items, chunk_size):
       chunk_items = load_medqa_usmle(
           'large_dataset.json',
           max_items=chunk_size,
           offset=i  # If your loader supports offset
       )
       all_items.extend(chunk_items)

Best Practices
--------------

1. **Reproducible Datasets**: Always use the same random seed for consistent results

   .. code-block:: bash

      python scripts/process_datasets.py --seed 42 [other options]

2. **Validation**: Always validate your final dataset

   .. code-block:: bash

      python scripts/process_datasets.py --validate [other options]

3. **Backup**: Keep backup copies of your original datasets

4. **Documentation**: Document your dataset processing pipeline

   .. code-block:: python

      # Document your processing steps
      processing_config = {
          'medqa_items': 300,
          'icliniq_items': 400, 
          'cochrane_items': 300,
          'complexity_stratification': True,
          'balance_complexity': True,
          'seed': 42
      }

5. **Version Control**: Track your dataset versions and processing scripts

API Reference
-------------

.. automodule:: src.data_loaders
   :members:
   :undoc-members:
   :show-inheritance: