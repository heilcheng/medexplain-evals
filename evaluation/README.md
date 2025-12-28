# Evaluation Framework

This directory contains the evaluation methodology and scoring components for MedExplain-Evals.

## Framework Overview

MedExplain-Evals uses a dual evaluation approach:

1. **Automated Metrics Suite** - Fast, objective scoring
2. **LLM-as-a-Judge** - Nuanced, rubric-based evaluation

## Directory Structure

```
evaluation/
├── metrics/              # Automated evaluation metrics
├── rubrics/             # Detailed scoring rubrics
├── judges/              # LLM-as-a-judge implementations
└── validation/          # Validation studies and correlations
```

## Evaluation Pipeline

```python
from src.evaluator import MedExplainEvaluator

evaluator = MedExplainEvaluator()
scores = evaluator.evaluate_explanation(
    original="medical content",
    generated="model explanation", 
    audience="patient"
)
```

## Scoring Dimensions

### 1. Readability Assessment
- **Flesch-Kincaid Grade Level**
- **SMOG Index** 
- **Average sentence length**
- **Audience-appropriate complexity**

### 2. Terminology Appropriateness
- **Medical term density analysis**
- **Audience-specific expectations**
- **Jargon appropriateness scoring**

### 3. Safety & Factual Consistency
- **Contradiction detection**
- **Information preservation**
- **Hallucination detection**
- **Safety compliance checking**

### 4. Information Coverage
- **BERTScore semantic similarity**
- **Key concept matching**
- **Completeness assessment**

### 5. LLM-as-a-Judge Quality
- **Multi-dimensional rubric scoring**
- **Audience-specific criteria**
- **Expert-validated prompts**

## Validation Strategy

### 1. Synthetic Agreement Testing
- Unambiguous test cases
- Rubric application verification
- Quality control checks

### 2. Cross-Model Agreement  
- Multiple LLM judges (GPT-4, Claude, DeepSeek)
- Inter-rater reliability (Krippendorff's Alpha)
- Consistency verification

### 3. Correlation Analysis
- Established quality indicators
- Source dataset alignment
- Metric validation

## Usage Examples

See [examples/evaluation_examples.py](../examples/evaluation_examples.py) for detailed usage patterns.