# MedExplain-Evals

A benchmark for evaluating audience-adaptive medical explanation quality in large language models.

## Overview

MedExplain-Evals provides infrastructure for:

- Evaluating how well LLMs generate medical explanations for different audiences (physicians, nurses, patients, caregivers)
- Measuring explanation quality across six dimensions (accuracy, terminology, completeness, actionability, safety, empathy)
- Using ensemble LLM-as-Judge with weighted scoring from multiple frontier models
- Grounding evaluations against medical knowledge bases (UMLS, RxNorm, SNOMED-CT)
- Generating publication-ready analysis reports and visualizations

## Requirements

### Software

- Python 3.10+
- API keys for at least one LLM provider (OpenAI, Anthropic, Google)
- 10 GB+ disk space (datasets and results)

### Hardware (for local models)

| Model Size | Minimum VRAM | Recommended | With Quantization |
|------------|--------------|-------------|-------------------|
| 7-9B       | 8 GB         | 16 GB       | 5 GB              |
| 13-14B     | 16 GB        | 24 GB       | 8 GB              |
| 70B+       | 40 GB        | 80 GB+      | 20 GB             |

## Installation

```bash
git clone https://github.com/heilcheng/medexplain-evals.git
cd medexplain-evals

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Supported Models

| Provider   | Models                                        | Multimodal |
|------------|-----------------------------------------------|------------|
| OpenAI     | GPT-5.2, GPT-5.1, GPT-5, GPT-4o              | ✓          |
| Anthropic  | Claude Opus 4.5, Sonnet 4.5, Haiku 4.5       | ✓          |
| Google     | Gemini 3 Ultra, Pro, Flash                   | ✓          |
| Meta       | Llama 4 Behemoth, Maverick, Scout            | ✓          |
| DeepSeek   | DeepSeek-V3                                  |            |
| Alibaba    | Qwen3-Max, Qwen3 family                      | ✓          |
| Amazon     | Nova Pro, Nova Omni                          | ✓          |

## Evaluation Dimensions

| Dimension                      | Weight | Description                                      |
|--------------------------------|--------|--------------------------------------------------|
| Factual Accuracy               | 25%    | Clinical correctness and evidence alignment      |
| Terminological Appropriateness | 15%    | Language complexity matching audience needs      |
| Explanatory Completeness       | 20%    | Comprehensive yet accessible coverage            |
| Actionability                  | 15%    | Clear, practical guidance                        |
| Safety                         | 15%    | Appropriate warnings and harm avoidance          |
| Empathy & Tone                 | 10%    | Audience-appropriate communication style         |

## Target Audiences

| Audience   | Variants                          | Health Literacy     |
|------------|-----------------------------------|---------------------|
| Physicians | Specialist, Generalist            | Expert              |
| Nurses     | ICU, General Ward, Specialty      | Professional        |
| Patients   | Low, Medium, High literacy        | Variable            |
| Caregivers | Family, Professional, Pediatric   | Variable            |

## Quick Start

```bash
# Set API keys
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key

# Validate environment
python scripts/validate_env.py

# Run evaluation
python scripts/run_evaluation.py \
  --config configs/evaluation_config.yaml \
  --models gpt-5.1 claude-opus-4.5 \
  --audiences patient_low_literacy physician_specialist \
  --output results/
```

## Configuration

```yaml
models:
  gpt-5.1:
    provider: openai
    temperature: 0.3
    max_tokens: 2048

  claude-opus-4.5:
    provider: anthropic
    temperature: 0.3

audiences:
  - patient_low_literacy
  - patient_medium_literacy
  - physician_specialist
  - nurse_general

evaluation:
  dimensions:
    factual_accuracy: 0.25
    terminological_appropriateness: 0.15
    explanatory_completeness: 0.20
    actionability: 0.15
    safety: 0.15
    empathy_tone: 0.10

judge:
  ensemble:
    - model: gpt-5.1
      weight: 0.30
    - model: claude-opus-4.5
      weight: 0.30
    - model: gemini-3-pro
      weight: 0.25
    - model: deepseek-v3
      weight: 0.15

output:
  path: ./results
  formats: [json, csv, html]
```

## Python API

```python
from src import UnifiedModelClient, EnsembleLLMJudge, PersonaFactory

# Initialize
client = UnifiedModelClient()
judge = EnsembleLLMJudge(client)
persona = PersonaFactory.get_predefined_persona("patient_low_literacy")

# Generate explanation
explanation = client.generate(
    model="gpt-5.1",
    messages=[{"role": "user", "content": "Explain type 2 diabetes simply"}]
)

# Evaluate
score = judge.evaluate(
    original="Type 2 diabetes mellitus with HbA1c 8.5%...",
    explanation=explanation.content,
    audience=persona
)

print(f"Overall: {score.overall:.2f}/5.0")
print(f"Agreement: {score.agreement_score:.2f}")
```

## Web Platform

An interactive web interface is available for browser-based evaluation:

```bash
# Backend
cd web/backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# Frontend
cd web/frontend
npm install && npm run dev
```

Access at http://localhost:3000

## Output Structure

```
results/
├── 20251229_143022/
│   ├── scores.json
│   ├── explanations/
│   │   ├── gpt-5.1/
│   │   └── claude-opus-4.5/
│   ├── analysis/
│   │   ├── dimension_breakdown.png
│   │   ├── audience_comparison.png
│   │   └── model_rankings.png
│   └── report.html
```

## Testing

```bash
pytest tests/
pytest --cov=src tests/
```

## Project Structure

```
medexplain-evals/
├── src/
│   ├── model_clients.py       # Unified LLM API client
│   ├── ensemble_judge.py      # Multi-model evaluation
│   ├── audience_personas.py   # Audience modeling
│   ├── safety_evaluator.py    # Safety assessment
│   ├── knowledge_grounding.py # UMLS/RxNorm integration
│   └── rubrics/               # G-Eval scoring rubrics
├── scripts/                   # CLI tools
├── analysis/                  # Visualization and reporting
├── web/                       # Web platform (FastAPI + Next.js)
├── configs/                   # Configuration templates
├── tests/                     # Test suite
└── examples/                  # Usage examples
```

## Citation

```bibtex
@software{medexplain2025,
  author = {Hailey Cheng},
  title = {MedExplain-Evals: A Benchmark for Audience-Adaptive Medical Explanation Quality in LLMs},
  year = {2025},
  url = {https://github.com/heilcheng/medexplain-evals}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

MedExplain-Evals is designed for research evaluation purposes only. Generated explanations should not be used for actual medical advice. Always consult qualified healthcare professionals for medical decisions.
