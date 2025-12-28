# Data Directory

This directory contains benchmark datasets and evaluation data for MedExplain-Evals.

## Structure

```
data/
├── raw/                    # Raw source datasets
├── processed/             # Processed benchmark items
├── reference/             # Reference explanations
└── sample/               # Sample data for testing
```

## Data Sources

MedExplain-Evals leverages existing, validated medical datasets:

- **MedQuAD**: 47,000+ Q&A pairs from NIH websites (patient/caregiver level)
- **HealthSearchQA**: 3,000+ consumer health questions
- **MedQA-USMLE**: Exam questions with detailed rationales (physician level)
- **iCliniq**: 30,000+ doctor-provided explanations
- **Cochrane Reviews**: Evidence-based summaries

## Usage

To load benchmark data:

```python
from src.benchmark import MedExplain

bench = MedExplain(data_path="data/")
stats = bench.get_benchmark_stats()
```

## Data Format

Each benchmark item follows this structure:

```json
{
  "id": "unique_identifier",
  "medical_content": "original medical information",
  "complexity_level": "basic|intermediate|advanced",
  "source_dataset": "dataset_name",
  "reference_explanations": {
    "physician": "technical explanation",
    "nurse": "practical care explanation",
    "patient": "patient-friendly explanation",
    "caregiver": "caregiver-focused explanation"
  }
}
```