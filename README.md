# MedExplain-Evals

**A Benchmark for Audience-Adaptive Medical Explanation Quality in LLMs**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Overview

MedExplain-Evals is a comprehensive evaluation framework for measuring how effectively large language models generate medical explanations tailored to different audiences. The benchmark assesses explanations across six dimensions: factual accuracy, terminological appropriateness, explanatory completeness, actionability, safety, and empathy.

**Key Capabilities:**
- Multi-audience evaluation (physicians, nurses, patients, caregivers)
- Ensemble LLM-as-Judge with weighted scoring
- Medical knowledge grounding via UMLS/RxNorm
- Comprehensive safety evaluation
- Support for 20+ frontier models (December 2025)

---

## Installation

```bash
git clone https://github.com/heilcheng/medexplain-evals.git
cd medexplain-evals
pip install -r requirements.txt
```

**Configure API Keys:**

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"        # Optional
export DEEPSEEK_API_KEY="your-key"      # Optional
```

---

## Quick Start

### 1. Validate Environment

```bash
make validate-env
```

### 2. Estimate API Costs

```bash
make estimate
```

### 3. Run Evaluation

```bash
# Smoke test (10 items, quick validation)
make smoke-test

# Full benchmark
make evaluate MODELS="gpt-5.1 claude-opus-4.5"
```

### 4. Generate Reports

```bash
make report
make view-report
```

---

## Project Structure

```
medexplain-evals/
├── src/                          # Core library
│   ├── model_clients.py          # Unified API client for all providers
│   ├── ensemble_judge.py         # Multi-LLM evaluation ensemble
│   ├── audience_personas.py      # Audience modeling with health literacy
│   ├── safety_evaluator.py       # Medical safety assessment
│   ├── knowledge_grounding.py    # UMLS/RxNorm integration
│   └── rubrics/                  # G-Eval style scoring rubrics
│
├── scripts/                      # Researcher toolkit
│   ├── curate_dataset.py         # Dataset curation pipeline
│   ├── run_evaluation.py         # Main evaluation runner
│   ├── generate_explanations.py  # Batch explanation generation
│   ├── compute_scores.py         # Score computation
│   ├── estimate_cost.py          # API cost estimator
│   └── run_full_benchmark.sh     # End-to-end script
│
├── analysis/                     # Analysis tools
│   ├── analyzer.py               # Score aggregation and statistics
│   ├── visualizations.py         # Charts and figures
│   ├── error_analysis.py         # Failure case identification
│   ├── report_generator.py       # HTML/Markdown reports
│   └── statistical_tests.py      # Significance testing
│
├── configs/                      # Configuration templates
│   ├── evaluation_config.yaml
│   └── models_config.yaml
│
├── data/
│   └── benchmark_v2/             # Curated benchmark dataset
│
└── results/                      # Evaluation outputs
```

---

## Evaluation Framework

### Dimensions

| Dimension | Description |
|-----------|-------------|
| **Factual Accuracy** | Clinical correctness and evidence alignment |
| **Terminological Appropriateness** | Language complexity matching audience needs |
| **Explanatory Completeness** | Comprehensive yet accessible coverage |
| **Actionability** | Clear, practical guidance for the reader |
| **Safety** | Appropriate warnings and harm avoidance |
| **Empathy & Tone** | Audience-appropriate communication style |

### Ensemble Judge

Evaluation uses a weighted ensemble of frontier models:

| Model | Provider | Weight |
|-------|----------|--------|
| GPT-5.1 | OpenAI | 0.30 |
| Claude Opus 4.5 | Anthropic | 0.30 |
| Gemini 3 Pro | Google | 0.25 |
| DeepSeek-V3 | DeepSeek | 0.10 |
| Qwen3-Max | Alibaba | 0.05 |

### Target Audiences

- **Physicians**: Specialist and general practice
- **Nurses**: Various specializations
- **Patients**: Low, medium, and high health literacy
- **Caregivers**: Family members and professional caregivers

---

## Supported Models

| Provider | Models | Multimodal |
|----------|--------|------------|
| OpenAI | GPT-5.2, GPT-5.1, GPT-5, GPT-4o | ✓ |
| Anthropic | Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 | ✓ |
| Google | Gemini 3 Ultra, Pro, Flash | ✓ |
| Meta | Llama 4 Behemoth, Maverick, Scout | ✓ |
| DeepSeek | DeepSeek-V3 | |
| Alibaba | Qwen3-Max | ✓ |
| Amazon | Nova Pro, Nova Omni | ✓ |

---

## Usage

### Programmatic API

```python
from src import UnifiedModelClient, EnsembleLLMJudge

# Initialize
client = UnifiedModelClient()
judge = EnsembleLLMJudge(client)

# Generate explanation
explanation = await client.generate_text(
    prompt="Explain hypertension for a patient with low health literacy",
    model="gpt-5.1"
)

# Evaluate
score = await judge.evaluate(
    original_medical_content="Hypertension management guidelines...",
    generated_explanation=explanation,
    audience_persona=persona,
    rubric=rubric
)

print(f"Overall: {score.overall_quality:.2f}")
print(f"By dimension: {score.dimension_scores}")
```

### Command Line

```bash
# Full pipeline
./scripts/run_full_benchmark.sh --models gpt-5.1,claude-opus-4.5 --items 100

# Individual steps
python scripts/curate_dataset.py --output data/benchmark_v2/full_dataset.json
python scripts/run_evaluation.py --benchmark data/benchmark_v2/test.json --models gpt-5.1
python scripts/compute_scores.py --explanations results/gpt-5.1/explanations/
```

### Make Targets

```bash
make validate-env    # Check environment setup
make estimate        # Estimate API costs
make curate          # Curate benchmark dataset
make evaluate        # Run evaluation
make analyze         # Analyze results
make report          # Generate HTML/Markdown reports
make smoke-test      # Quick validation run
```

---

## Docker

```bash
# Build image
docker build -t medexplain-evals:2.0 .

# Run with Docker Compose
docker-compose up -d medexplain

# With local LLM (Llama 4 via vLLM)
docker-compose --profile gpu up -d
```

---

## Validation

MedExplain-Evals includes a comprehensive validation framework:

1. **Synthetic Agreement Testing**: Unambiguous test cases with known scores
2. **Human Correlation**: Comparison with expert annotations (target ρ > 0.80)
3. **Inter-Rater Reliability**: Cross-model agreement (target α > 0.75)
4. **Statistical Testing**: Paired t-tests with bootstrap confidence intervals

---

## Citation

```bibtex
@inproceedings{medexplain2025,
  title     = {MedExplain-Evals: A Benchmark for Audience-Adaptive 
               Medical Explanation Quality in LLMs},
  author    = {Cheng, Heil and others},
  year      = {2025},
  url       = {https://github.com/heilcheng/medexplain-evals}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

MedExplain-Evals is designed for **research evaluation purposes only**. Generated explanations should not be used for actual medical advice. Always consult qualified healthcare professionals for medical decisions.
