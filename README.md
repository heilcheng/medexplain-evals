# MedExplain-Evals

> This project is developed as part of [Google Summer of Code 2025](https://summerofcode.withgoogle.com/) at [Google DeepMind](https://deepmind.google/).
>
> See [my GSoC work record](https://github.com/heilcheng/DeepMind) for documentation of my contributions during the program.

A benchmark for evaluating audience-adaptive medical explanation quality in large language models.

## Overview

MedExplain-Evals provides infrastructure for:

- Evaluating how well LLMs generate medical explanations for different audiences (physicians, nurses, patients, caregivers)
- Measuring explanation quality across six dimensions (accuracy, terminology, completeness, actionability, safety, empathy)
- Using ensemble LLM-as-Judge with weighted scoring from multiple frontier models
- Grounding evaluations against medical knowledge bases (UMLS, RxNorm, SNOMED-CT)
- Generating publication-ready analysis reports and visualizations

**[Read the full documentation](https://heilcheng.github.io/medexplain-evals/)**

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

## Benchmark Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MedExplain-Evals Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────────┐│
│  │   Input     │    │   LLM Under     │    │     Generated Explanation   ││
│  │  Clinical   │───▶│     Test        │───▶│     for Target Audience     ││
│  │  Scenario   │    │  (API/Local)    │    │                             ││
│  └─────────────┘    └─────────────────┘    └──────────────┬───────────────┘│
│                                                            │                │
│                                                            ▼                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Evaluation Engine                                    ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐││
│  │  │ Knowledge      │  │  Ensemble      │  │     Safety                 │││
│  │  │ Grounding      │  │  LLM-as-Judge  │  │     Evaluator              │││
│  │  │                │  │                │  │                            │││
│  │  │ • UMLS Lookup  │  │ • GPT-5.x      │  │ • Drug interactions       │││
│  │  │ • RxNorm Match │  │ • Claude 4.5   │  │ • Contraindications       │││
│  │  │ • SNOMED-CT    │  │ • Gemini 3     │  │ • Harm classification     │││
│  │  │ • NLI Verify   │  │ • DeepSeek-V3  │  │ • Warning detection       │││
│  │  └────────────────┘  └────────────────┘  └────────────────────────────┘││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                            │                │
│                                                            ▼                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Dimension Scoring (Weighted)                         ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │  Factual Accuracy (25%) │ Terminology (15%) │ Completeness (20%)       ││
│  │  Actionability (15%)    │ Safety (15%)      │ Empathy & Tone (10%)     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Medical Knowledge Grounding

MedExplain-Evals grounds factual claims against established medical ontologies:

### UMLS (Unified Medical Language System)

The UMLS integrates over 200 source vocabularies and provides a comprehensive framework for medical concept normalization.

| Resource | Description | Usage in MedExplain |
|----------|-------------|---------------------|
| Metathesaurus | Unified concepts from 200+ sources | Entity linking, concept validation |
| Semantic Network | Semantic types and relations | Relationship verification |
| SPECIALIST Lexicon | Biomedical language tools | Term normalization |

**Citation:**
```bibtex
@article{bodenreider2004umls,
  author = {Bodenreider, Olivier},
  title = {The Unified Medical Language System (UMLS): integrating biomedical terminology},
  journal = {Nucleic Acids Research},
  volume = {32},
  number = {suppl\_1},
  pages = {D267--D270},
  year = {2004},
  doi = {10.1093/nar/gkh061},
  url = {https://www.nlm.nih.gov/research/umls/}
}
```

### RxNorm

RxNorm provides normalized names and identifiers for clinical drugs, enabling drug safety verification.

| Component | Description | Usage in MedExplain |
|-----------|-------------|---------------------|
| RxCUI | Unique drug identifiers | Drug entity validation |
| Drug classes | Therapeutic categories | Drug-condition matching |
| Ingredient links | Active ingredient mapping | Interaction checking |

**Citation:**
```bibtex
@article{nelson2011rxnorm,
  author = {Nelson, Stuart J and Zeng, Kelly and Kilbourne, John and Powell, Tammy and Moore, Robin},
  title = {Normalized names for clinical drugs: RxNorm at 6 years},
  journal = {Journal of the American Medical Informatics Association},
  volume = {18},
  number = {4},
  pages = {441--448},
  year = {2011},
  doi = {10.1136/amiajnl-2011-000116},
  url = {https://www.nlm.nih.gov/research/umls/rxnorm/}
}
```

### SNOMED CT (Clinical Terms)

SNOMED CT is the most comprehensive clinical terminology, covering diseases, procedures, and clinical findings.

| Hierarchy | Description | Usage in MedExplain |
|-----------|-------------|---------------------|
| Clinical finding | Disorders, symptoms | Diagnosis validation |
| Procedure | Medical interventions | Treatment verification |
| Body structure | Anatomical concepts | Anatomical accuracy |
| Pharmaceutical product | Medications | Drug reference checking |

**Citation:**
```bibtex
@article{donnelly2006snomed,
  author = {Donnelly, Kevin},
  title = {SNOMED-CT: The advanced terminology and coding system for eHealth},
  journal = {Studies in Health Technology and Informatics},
  volume = {121},
  pages = {279--290},
  year = {2006},
  url = {https://www.snomed.org/}
}
```

### Additional Medical Resources

| Resource | Purpose | Citation |
|----------|---------|----------|
| **MedDRA** | Adverse event terminology | [ICH MedDRA](https://www.meddra.org/) |
| **ICD-10/11** | Disease classification | [WHO ICD](https://www.who.int/standards/classifications/icd) |
| **LOINC** | Lab/clinical observations | [Regenstrief LOINC](https://loinc.org/) |
| **DrugBank** | Drug interaction data | [DrugBank 5.0](https://go.drugbank.com/) |

## Supported Models

| Provider   | Models                                        | Multimodal |
|------------|-----------------------------------------------|------------|
| OpenAI     | GPT-5.2, GPT-5.1, GPT-5, GPT-4o              | ✓          |
| Anthropic  | Claude Opus 4.5, Sonnet 4.5, Haiku 4.5       | ✓          |
| Google     | Gemini 3 Pro, Flash                          | ✓          |
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
python scripts/validate_environment.py

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

Features:
- **Dashboard**: Overview of evaluation runs and statistics
- **Playground**: Interactive testing of medical explanations
- **Models**: Configure API and local model providers
- **Audiences**: Browse 11 medical audience personas
- **Results**: Visualize evaluation scores and rankings

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
├── src/                       # Core Python library
│   ├── clients/               # LLM API clients
│   ├── evaluation/            # Scoring and judging
│   ├── personas/              # Audience modeling
│   ├── data/                  # Data loading
│   ├── knowledge/             # UMLS/RxNorm grounding
│   ├── core/                  # Shared utilities
│   └── rubrics/               # G-Eval scoring rubrics
├── scripts/                   # CLI tools
├── web/                       # Web platform (Next.js + FastAPI)
├── docs/                      # Sphinx documentation
├── analysis/                  # Visualization and reporting
├── configs/                   # Configuration templates
├── data/                      # Sample datasets
├── tests/                     # Test suite
└── examples/                  # Usage examples
```

## Documentation

Full documentation is available at: https://heilcheng.github.io/medexplain-evals/

## Citation

If you use MedExplain-Evals in your research, please cite:

```bibtex
@software{medexplain2025,
  author = {Cheng Hei Lam},
  title = {MedExplain-Evals: A Benchmark for Audience-Adaptive Medical Explanation Quality in LLMs},
  year = {2025},
  url = {https://github.com/heilcheng/medexplain-evals}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

MedExplain-Evals is designed for research evaluation purposes only. Generated explanations should not be used for actual medical advice. Always consult qualified healthcare professionals for medical decisions.
