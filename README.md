# Plainline

Fidelity under audience adaptation. When a model explains a medical source to a nurse, a parent, or a reader at a sixth-grade level, which safety-critical facts does it drop, and does it add false ones?

**Status: pilot.** The repository contains 30 open-licensed sources and 339 draft facts. The fact lists are machine-drafted and have **not been reviewed by clinicians** (`status: draft, needs clinician review` on every fact set). No model scores are published here. Numbers produced from this data are for checking the pipeline, not for ranking models. This repository was previously named MedExplain-Evals; the design reasoning for the rebuild is in [docs/design.md](docs/design.md).

## What it measures

Each item is a source text (an FDA label excerpt or a MedlinePlus topic summary) and a reader. The model writes an explanation for that reader. The same source is always also explained with no reader specified (the neutral condition), so every comparison is paired.

A fact list per source says what must survive: facts whose omission could cause harm (`must_keep`), facts that matter for correct use (`should_keep`), and background (`optional`). A `must_not_say` list holds plausible wrong simplifications. A judge answers one narrow question per fact ("is this stated, contradicted, or absent?"). It does not rate quality, tone or empathy, and there is no weighted composite score.

Two views of the same question:

- `adapt` mode: the prompt names the reader ("an adult at about a sixth-grade level"). Readers: clinician, nurse, grade6_adult, limited_english, caregiver_child, plus neutral.
- `cue` mode: the user states a cue about themselves (profession, literacy, language, age, caregiver role) and asks for an explanation, with identical task text across cues. This is the paired reader-cue slice: does a model change safety content because of who it thinks is asking?

## Metrics

All are computed per model and per condition, with 95 percent percentile bootstrap intervals that resample items (so pairing is preserved).

| Metric | Definition |
|---|---|
| Critical Fact Retention (CFR) | Share of `must_keep` facts judged "stated", averaged over items. |
| Should-keep retention (SFR) | Same for `should_keep` facts. |
| Adaptation Tax | Paired per item: CFR(neutral) minus CFR(condition). Positive means the model keeps fewer critical facts for that reader. Reported in points. |
| Distortion rate | Share of facts of any tier judged "contradicted". |
| Must-not-say violations | Share of `must_not_say` statements the explanation asserts. |
| Readability band | Share of outputs whose Flesch-Kincaid grade falls in the condition's band (pass or fail gate, not a quality score). Bands live in `plainline/data/readers.yaml` and are working values. |
| Worst-reader CFR | Lowest mean CFR over the non-neutral conditions. The one headline number. |
| CFR spread | Max minus min mean CFR across conditions. In `cue` mode this is the audience-invariance disparity. |

Length control: every report includes mean output words per condition, the tax recomputed on items where the reader output is within a factor of 1.33 of the neutral length (`Tax at matched length`), CFR by output-length tercile, and the Spearman correlation between CFR and length. A model that wins only by writing more will show up there.

Not yet measured: unsupported claims that the model adds beyond the source (only contradiction of listed facts and must-not-say statements are checked). It is on the roadmap.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                  # all model calls are mocked
plainline validate-data

# offline smoke run, measures nothing about any model
plainline run --mock --out runs/smoke

# any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, OpenRouter, ...)
export OPENAI_API_KEY=...
plainline run --base-url https://api.openai.com/v1 --model gpt-4o-mini \
    --judge-base-url https://api.openai.com/v1 --judge-model gpt-4o \
    --mode adapt --out runs/gpt-4o-mini

plainline report --run runs/gpt-4o-mini           # recompute tables and CIs
```

Use a judge from a different family than the model under test. The CLI warns when they match. Output goes to `runs/<name>/` as `scored.jsonl` (one row per generation, with the text and every verdict), `summary.json` and `results.md`. Runs resume if interrupted.

Inspect AI:

```bash
inspect eval plainline/inspect_task.py --model openai/gpt-4o-mini \
    -T judge_model=openai/gpt-4o -T mode=adapt
```

The task reports `worst_reader_cfr`, `adaptation_tax`, and means of CFR, SFR, distortion, must-not-say and band pass.

Rebuild the pilot data (needs network): `plainline build-pilot`. Draft facts with a model instead of the rule-based pilot drafter: `plainline draft --base-url ... --model ...`. Either way the output is marked as a draft.

## Layout

```
plainline/          package: schema, readers, prompts, judge, metrics, runner, report, cli, inspect_task
plainline/pipeline/ DailyMed and MedlinePlus fetchers, chunking, draft fact extraction
data/               sources.jsonl and facts.jsonl (pilot, dev split)
docs/design.md      the design reasoning and landscape review behind this rebuild
docs/judge_validation.md   how to validate the judge against human labels
DATA_LICENSES.md    terms found for each source
```

## Limitations

- Fact lists are drafts. The pilot ones come from a deterministic rule-based drafter (`rule-v0`): facts are whole source sentences, not truly atomic, and the `must_not_say` lists are three generic templates for drug labels only. The model-based drafter exists but was not run, to avoid spending on unreviewed output.
- The judge is unvalidated. No human labels exist yet.
- All sources are public web text and probably in training data. Everything is a `dev` split. There is no held-out test split.
- Reading bands and the cue wording are unvalidated against real readers.
- Simulated readers are prompts. This benchmark says nothing about whether real patients understand an explanation (Bean et al., 2025, show such gaps exist).
- DailyMed redistribution terms are unconfirmed (see DATA_LICENSES.md).

## Roadmap

1. Clinician annotation: two or more clinicians review and tier every fact, add missed facts and distortions, and adjudicate. Fact sets move to `clinician_reviewed` only with recorded reviewers and a reported kappa.
2. Judge validation on at least 300 clinician-labelled facts (see docs/judge_validation.md); rule-based checks for numeric limits where the judge is weak.
3. Held-out split: freshly written clinician vignettes never published as raw text, plus rotating label snapshots, with canary strings and private test labels.
4. Scale from 30 to about 150 sources with difficulty enrichment, and add CDC pages after a per-page license check.
5. Add the unsupported-claim check against the source.
6. Teachback, a human-calibration study: real readers answer comprehension questions on model explanations, to test whether a simulated reader or this judge tracks human understanding. Needs an IRB determination and a crowd or recruited sample.

## Citation

No paper yet. If you use this code or data, cite the repository:

```bibtex
@misc{plainline2026,
  author = {Cheng Hei Lam},
  title = {Plainline: fidelity under audience adaptation for medical explanations},
  year = {2026},
  note = {Pilot, fact lists are unreviewed drafts},
  url = {https://github.com/heilcheng/medexplain-evals}
}
```

MIT license for the code. Source text keeps its original terms (see DATA_LICENSES.md).
