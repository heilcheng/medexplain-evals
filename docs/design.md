# MedExplain-Evals redesign proposal

Date: 2026-11-24. Status: design only, nothing pushed or renamed.

Conventions: "verified" means I opened a source this session (search result or API call). "Unverified" means I am working from memory or an aggregator. Cost and effort figures are my own estimates, not sourced. Several of the 2026 arXiv items below came from search snippets only, so open them before citing them in a paper.

## 1. Honest audit of the current repo

What it is (read from the repo at commit 80ed0e5, history 2025-07-01 to 2025-12-29, 44 commits):

- Task: given a piece of medical content, produce an explanation for one of four audiences (physician, nurse, patient, caregiver), with 11 persona variants.
- Data: `data/sample_dataset.json` has 3 items and `data/clinical_vignettes/sample_vignettes.json` has 5. That is the whole shipped dataset. `data/README.md` says the benchmark will draw on MedQuAD, HealthSearchQA, MedQA-USMLE, iCliniq and Cochrane, but those are loaders and scripts, not a released benchmark. There is no results directory, no leaderboard data, no published scores.
- Metrics: six weighted dimensions (accuracy 25, terminology 15, completeness 20, actionability 15, safety 15, empathy 10), scored 1 to 5 by an LLM ensemble (GPT, Claude, Gemini, DeepSeek, Qwen with hand-set weights), plus readability formulas, UMLS/RxNorm grounding, and a keyword-style safety evaluator.
- Engineering: about 27.7k lines of Python, but the same modules exist two or three times (`src/ensemble_judge.py` and `src/evaluation/ensemble_judge.py`, `src/rubrics/` and `src/evaluation/rubrics/`, `src/data_loaders.py`, `src/data_loaders_v2.py`, `src/data/data_loaders.py`). Plus a Next.js and FastAPI web app, Sphinx docs, Docker. It is a framework, not a dataset.

What is weak, most serious first:

1. There is no benchmark yet. Eight items cannot rank models. Nothing has been run and reported.
2. The judge is the measurement, and it is unvalidated. `evaluation/validate_judge.py` computes a correlation against human ratings, but no human ratings exist in the repo. The weights (0.30/0.30/0.25/0.15) and the dimension weights are arbitrary. Judges come from the same families as the models under test, so self-preference and style bias are uncontrolled. The scores are 1 to 5 holistic ratings, which is the format that saturates fastest and discriminates least.
3. Six weighted dimensions collapse safety and accuracy into an average. A response that omits a lethal contraindication but is warm and readable still scores well. Averaging hides the failures that matter.
4. The ground truth is the source text plus an LLM opinion. There is no list of facts that must survive adaptation, so "completeness" and "accuracy" are judged by vibes.
5. No clinician or lay-reader validation. The "audiences" are prompt personas. Nobody checked that a patient persona output is understood by a real patient.
6. The data design invites contamination. MedQA, MedQuAD, HealthSearchQA, iCliniq and Cochrane text are all heavily scraped. Public static items with no held-out split will be in training data.
7. The task is mostly solved at the surface level. Frontier models write fluent patient-friendly text; MedHELM already reports Patient Communication and Education as the strongest category (0.78 to 0.83 per its abstract). A benchmark whose headline is "is the explanation nice" will saturate on the judge.
8. Reuse problem: it needs four API keys, a bespoke harness, and a web stack. Labs run Inspect, lm-eval-harness or simple-evals style tasks, not a FastAPI server.
9. Licensing: README lists UMLS, SNOMED CT and DrugBank grounding, which are license-gated. That blocks a clean open release.
10. Minor: README claims "Google Summer of Code 2025 mentored by Google DeepMind" framing; keep that factual and out of the benchmark's pitch.

The one good idea in the repo is the right question, not the implementation: models must change how they communicate for different readers without changing the facts that matter. Nobody has a hard, clean measure of that.

## 2. Landscape (what labs report, what got adopted)

What labs and the field use now:

- HealthBench (OpenAI, May 2025): 5,000 multi-turn conversations, 48,562 rubric criteria written by 262 physicians, graded by a model, with Consensus and Hard subsets; released through simple-evals. [arXiv 2505.08775](https://arxiv.org/pdf/2505.08775). simple-evals is MIT licensed (verified via GitHub API); the data license itself I could not confirm. Third-party reimplementations exist (for example `m42-health/healthbench`).
- HealthBench Professional (OpenAI, April 2026): 525 tasks from a pool of 15,079, rubrics adjudicated by three or more physicians, difficulty-enriched, with a physician reference baseline. [arXiv 2604.27470](https://arxiv.org/abs/2604.27470), [OpenAI PDF](https://cdn.openai.com/dd128428-0184-4e25-b155-3a7686c7d744/HealthBench-Professional.pdf). The Claude Opus 5 system card reports it with both raw and length-adjusted scores ([system card](https://www-cdn.anthropic.com/c5fbac3f0b1280a933ebd26d3cb8bb9f5bdeaf48/Claude%20Opus%205%20System%20Card.pdf); numbers came via a search summary, unverified). I found no Gemini or Llama card reporting it. So HealthBench is the de facto lab-reported health eval, and length-adjustment is now a norm labs care about.
- MedHELM (Stanford, Nature Medicine 2026 per project site): clinician-validated taxonomy of 5 categories, 22 subcategories, 121 tasks, 35 benchmarks, LLM-jury grading with reported ICC 0.47 against clinicians (authors' own claim). [arXiv 2505.23802](https://arxiv.org/abs/2505.23802), [site](https://medhelm.org/).
- MedQA and MMLU-medical: saturated. Vals.ai archived its MedQA leaderboard with most new models above 95 percent ([Vals](https://www.vals.ai/benchmarks/medqa)); Inspect Evals itself notes MedQA is close to saturation ([docs](https://ukgovernmentbeis.github.io/inspect_evals/evals/medqa/index.html)).
- Agentic and interactive: MedAgentBench (300 physician-written EHR tasks over a FHIR environment, best original model 69.67 percent) [arXiv 2501.14654](https://arxiv.org/abs/2501.14654); AgentClinic (simulated encounters, static accuracy overstates interactive performance; I did not verify its authors in this session).
- Safety and calibration: Health-ORSC-Bench on over-refusal, 31,920 benign boundary prompts, ACL 2026 Findings [arXiv 2601.17642](https://arxiv.org/abs/2601.17642); MedQAbstain, models almost never abstain under medical uncertainty [ACL 2026](https://aclanthology.org/2026.acl-long.1365/).
- Humans in the loop: Bean et al., a 1,298 person randomized trial where models alone identified conditions in 94.9 percent of cases but people using them did worse than controls (odds 1.76 times higher for control) [arXiv 2504.18919](https://arxiv.org/abs/2504.18919). This is the strongest argument that communication, not knowledge, is the bottleneck, and that simulated users do not predict real ones.
- Patient communication work (closest prior art to this repo):
  - DischargeSim, EMNLP 2025, simulated patients plus a downstream comprehension exam [arXiv 2509.07188](https://arxiv.org/pdf/2509.07188).
  - DischargeBench, 477 MIMIC-derived cases, persona axes including health literacy, judge validated against physicians, authors say it is not yet validated against real patients [arXiv 2609.20827](https://arxiv.org/abs/2609.20827). Note it uses MIMIC, so data access is credentialed.
  - RepQA, readability of LLM answers to lay questions, 25 LLMs, models miss target reading levels [arXiv 2509.16360](https://arxiv.org/html/2509.16360).
  - PlainFact and PlainQAFact, factual consistency for plain-language summaries of biomedical abstracts [arXiv 2503.08890](https://arxiv.org/pdf/2503.08890); FactPICO, 345 LLM plain-language summaries of 115 RCT abstracts with expert factuality ratings [arXiv 2402.11456](https://arxiv.org/pdf/2402.11456).
  - MedLayBench-V and MedLayXPlain on expert-lay alignment for vision-language models [arXiv 2604.05738](https://arxiv.org/abs/2604.05738), [arXiv 2606.21194](https://arxiv.org/html/2606.21194) (snippet level only).
  - A 2026 JMIR scoping review found few LLM patient-education outputs reach a sixth-grade reading level [JMIR](https://www.jmir.org/2026/1/e98084) (snippet level).

Harness landscape: Inspect Evals (UK AISI) and lm-evaluation-harness both carry medical tasks; I did not find a standalone HealthBench in Inspect Evals from the search, so check before claiming a gap. simple-evals hosts HealthBench.

Why adopted benchmarks got adopted (my synthesis, partly from the above):

1. Expert-written rubrics with a named physician count (HealthBench, HealthBench Professional, MedAgentBench). Labs cite "262 physicians", not "an LLM judge".
2. Hard and not saturated, often with an enriched hard subset, so there is headroom for the next model.
3. One command to run. HealthBench shipped inside simple-evals. A benchmark that needs a bespoke server will not be run.
4. Clean license and no gated data (no UMLS, SNOMED, MIMIC dependency for the core split).
5. A held-out part. Public dev set, private or rotating test set.
6. Reports a single headline number plus a few interpretable axes. Labs put one or two numbers in a system card.
7. Backed by an institution and a paper at a recognised venue.

Gaps I see that this repo could occupy:

- No benchmark measures whether adapting to a reader preserves the facts that matter. HealthBench has communication axes and a context-seeking theme, but it is not built around a controlled same-content, different-reader comparison. (I read abstracts and search summaries, not the full HealthBench rubric tagging, so confirm before claiming it is absent.)
- Length-adjusted scoring is needed, since longer answers win rubric graders. A fidelity benchmark with a fixed must-keep fact list gives a length-robust check.
- Audience cues as a safety variable: whether a model drops warnings, doses or escalation advice when it thinks the reader is low literacy, young, non-native, or a professional. I did not find a benchmark doing counterfactual audience pairs on safety content (not exhaustive, one search).
- Validation of simulated readers against real readers. DischargeBench states it lacks this.

## 3. Three redesign directions

### Direction A (recommended core): Plainline, a fidelity-under-adaptation benchmark

Name: Plainline (working name). "Plainline" returned no benchmark or paper hits on arXiv search and the GitHub hits are unrelated (a Neovim statusline, a coding syllabus). Not a guarantee; run a final check before announcing. Fallback names: AudienceDrift, KeepTheFacts. "PlainFacts" is taken in spirit (PlainFact/PlainQAFact) and "KeepFacts" is a GitHub repo.

Research question: when a model rewrites medical information for a reader with less (or different) expertise, how many of the safety-critical facts survive, how many false facts get added, and does that "simplification tax" differ by audience cue, model, or length?

Task format: each item is a source document plus a reader spec. Model sees `(source, reader)` and writes the explanation. Reader specs: expert clinician, general nurse, adult at grade 6 reading level, adult with limited English, caregiver of a child, plus a neutral no-cue control. The same source appears under all readers, so the comparison is paired.

Data source and license: open public-domain or permissively licensed text only.
- FDA drug labels from DailyMed (US government work, public). Boxed warnings, contraindications, dosing limits, and interaction sections give natural must-keep facts.
- MedlinePlus and CDC/NIH patient and clinician pages (mostly US government, but some third-party content on MedlinePlus is licensed; filter by source, unverified).
- Guideline summaries with permissive licenses (check each; NICE and many society guidelines are not open).
- A smaller set of newly written vignettes by clinicians, never published in text form (held-out test).
Avoid UMLS, SNOMED, DrugBank, MIMIC in the core pipeline.

Ground truth: for each source, atomic facts are extracted with an LLM, then reviewed and tiered by clinicians into three tiers: must keep (omission could cause harm, for example "do not exceed 4 g per day", "stop and seek care if..."), should keep, and optional. Also a list of must-not-say distortions (common wrong simplifications, for example "safe to double a missed dose"). Tiering by at least two clinicians per item with adjudication, reporting kappa. This is the physician-rubric pattern that made HealthBench credible, but applied to fact preservation.

Metrics (headline first):
- Critical Fact Retention (CFR): share of must-keep facts present and correct, per reader. Judged by a narrow entailment question per fact ("does the answer state X?"), a far easier and more reliable judge task than a 1 to 5 holistic score. Verified against clinician labels on a sample.
- Adaptation Tax: CFR(neutral control) minus CFR(reader). Paired, so source difficulty cancels.
- Distortion rate: must-not-say statements asserted, plus unsupported added claims checked against source (PlainQAFact style).
- Reader fit: reading grade (Flesch-Kincaid, SMOG), jargon rate against a word list, checked against the target. A pass or fail band, not a quality score.
- Length-controlled reporting: all scores also reported at matched length buckets, since graders favor long answers.
No weighted average of "empathy" and "safety". Report the vector and one headline: worst-reader CFR.

Contamination protection: public dev split of about 20 percent; test split built from freshly written vignettes plus label text snapshots dated after a stated cutoff, rotated each release (LiveMedBench-style idea, see [arXiv 2602.10367](https://arxiv.org/pdf/2602.10367), unverified details). Include canary strings. Optionally keep test labels private and score via a hosted checker.

Why a lab would use it: it is cheap (single-turn, short outputs), measures a failure the lab cares about (dropping warnings when simplifying, which is a real product risk for consumer health features), has headroom, and yields a per-fact failure list a safety team can act on. It ships as an Inspect task and a simple-evals style script, so adding it to an eval suite is one line.

Effort and cost (estimates): 300 sources, 6 readers, about 6 to 10 must-keep facts each. Clinician tiering: about 2,000 to 3,000 facts reviewed by 2 clinicians, perhaps 40 to 80 clinician hours. Judge calls: low thousands of dollars at most for 15 models. Engineering: 3 to 4 weeks, mostly deleting code. Clinician time is the real cost and the bottleneck.

Risks: (1) fact extraction by an LLM biases the fact list toward what LLMs notice, so clinician add-ons are required. (2) Entailment judge errors, mitigated by validating on a clinician-labelled sample. (3) Models may saturate CFR for easy sources; solve with difficulty enrichment (long labels, interacting warnings). (4) Prior art on plain-language factuality exists, so novelty must rest on the paired audience design and tiered criticality, not on "we check facts".

Minimal publishable version: 150 sources, 4 readers, 2 clinicians, 10 models, human validation of the entailment judge on 300 facts, plus the adaptation-tax finding. Fits a NeurIPS Datasets and Benchmarks submission or a workshop paper (for example a health or evaluation workshop). Deadlines are not checked here; verify them.

### Direction B: Teachback, human-calibrated comprehension

Name: Teachback (working). GitHub searches for "teachback medical benchmark" and variants returned nothing, but "teach-back" is a standard clinical term, so expect search collisions in papers. Fallbacks: LandsRight, ReadersCheck. Needs a final check.

Research question: do explanations that LLM judges rate highly actually produce understanding in real lay readers, and can a simulated reader be calibrated to predict human comprehension?

Task format: model explains a source document to a reader spec; then comprehension is measured with a short set of questions (multiple choice plus a free-text teach-back summary). Two arms: simulated readers (LLM persona) and real readers (crowd workers recruited for specified literacy bands).

Data source and license: same open sources as A (DailyMed, MedlinePlus, CDC/NIH), plus lay questions from a licensed or consented source. Avoid scraping Reddit or iCliniq text (terms and privacy concerns, unverified). Comprehension questions written and checked by clinicians and a health-literacy specialist.

Ground truth: real human comprehension scores. This is the distinguishing feature. Real readers answer questions after reading a model explanation; the benchmark ships both human scores and a calibrated simulated reader, with reported correlation.

Metrics: comprehension accuracy on critical questions (human and simulated), teach-back recall of must-keep points, calibration (rank correlation between simulated and human model rankings), reading time, and a confusion rate for misinterpretations that would cause harm.

Contamination protection: question sets rotated, test questions unpublished, human data collected fresh per release.

Why a lab would use it: it ties to the strongest real-world result in the field, that model knowledge does not transfer to user outcomes (Bean et al.). A validated cheap simulated reader would be genuinely useful and none exists yet. DischargeBench says plainly it lacks validation against real patient encounters, so the gap is stated by the competitor.

Effort and cost (estimates): human data is the cost. For example 100 sources by 5 models by 6 to 8 readers per cell gives thousands of paid reader-sessions, likely several thousand dollars on a crowd platform, plus an IRB or exempt determination from Columbia, plus recruitment quality control. Engineering 4 to 6 weeks, study 6 to 10 weeks.

Risks: IRB timeline; crowd workers are not patients and may be higher literacy than the target population; simulated reader may fail to track humans (a known issue, one study found LLMs overestimate human comprehension on long discharge summaries, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13690339), snippet level). A negative calibration result is still publishable but is a weaker benchmark. High prior-art density (DischargeSim, DischargeBench).

Minimal publishable version: 60 sources, 4 models, 3 literacy bands, 40 human readers per cell, report simulated versus human correlation. A short paper or workshop paper; D&B only if human data is large enough.

### Direction C: Audience-invariance safety audit

Name: Equalcare (working). Not searched exhaustively; GitHub search for it was not run, so treat as unchecked. Fallback: SameFacts.

Research question: do models change safety-critical content (red-flag warnings, dose limits, advice to seek urgent care) based on who they think the reader is (literacy, age, language, claimed profession, demographic cues), holding the clinical situation fixed?

Task format: counterfactual pairs. The same clinical question or situation with a different reader cue prepended or embedded in the user message. Pairs differ only in the cue.

Data source and license: newly written clinician vignettes (the unpublishable test set is the point) plus open label and guideline text. Fully synthetic cues, no real patient data.

Ground truth: clinician-written checklist of safety-critical actions per vignette (must warn, must escalate, must not recommend). Gold lists are the same across cues, so any gap is a disparity.

Metrics: safety-checklist recall per cue, disparity (max minus min across cues), rate of over-disclosure to claimed professionals versus under-disclosure to lay cues, refusal asymmetry across cues, with bootstrap confidence intervals over vignettes.

Contamination protection: vignettes never published in raw text, rotating cues, a private test split.

Why a lab would use it: it maps onto fairness and safety reporting labs already do, gives a crisp pass or fail audit, and counterfactual pairs are hard to game by learning a general style. It also supplies a subscore for Plainline.

Effort and cost (estimates): about 200 vignettes by 8 cues, clinician checklists 30 to 50 hours. Cheapest of the three. Engineering 2 to 3 weeks.

Risks: if models are already invariant, the result is "no disparity", which is useful but less exciting; cue-based findings need care to avoid stereotyping claims; a narrower story than A; some safety-audit and bias-in-medical-LLM work exists (for example [The Biased Oracle, arXiv 2511.00924](https://arxiv.org/html/2511.00924), snippet level), so novelty has to be checked.

Minimal publishable version: 100 vignettes, 6 cues, 8 models. Workshop paper or a section of Plainline.

## 4. Recommendation

Build Direction A, Plainline, with Direction C folded in as a "reader cue" slice (the neutral control plus paired cues already give disparity metrics), and keep Direction B as a later validation study rather than the launch.

Why:

- It repairs the repo's central flaw. The old design measured quality by LLM opinion. A fact checklist tiered by clinicians gives a verifiable ground truth and replaces the weighted six-dimension average.
- It is the cheapest direction that still has physician-built rubrics, the property that made HealthBench and MedAgentBench credible to labs.
- It has headroom. Models are fluent; the open question is whether they drop warnings when simplifying, and lab consumer health products sit exactly on that risk.
- It is judge-light. Per-fact entailment checks on short outputs are more reliable and cheaper than holistic scoring, and can be validated against a clinician-labelled sample, answering the circularity criticism directly.
- It runs as a normal Inspect task, with no web stack, no gated terminologies, and no human-subjects approval needed for v1.
- B is the more exciting science but costs weeks of IRB and several thousand dollars of crowd work, and sits in the most crowded prior-art area. A produces the clinician-labelled fact sets that B would later need anyway, so A does not block B.

What not to do: do not pitch it as "a better judge ensemble" or "empathy scoring". Do not claim novelty over PlainQAFact or DischargeBench in factual checking in general; the claim is the paired audience design and criticality tiers.

## 5. What can be reused from the current code

Keep (after cleanup):
- `src/audience_personas.py` and `src/personas/`: starting point for reader specs, trimmed from 11 personas to about 6 with explicit reading-grade targets.
- Readability and terminology metrics in `src/evaluator.py` (Flesch-Kincaid, SMOG, term density): reuse as the "reader fit" check.
- `src/data_schema.py` and validation: adapt to the item schema (source, reader, fact list, tiers).
- `scripts/generate_splits.py`, `scripts/compute_scores.py`: adapt for dev/test splitting and aggregation.
- `analysis/statistical_tests.py` and `analysis/visualizations.py`: bootstrap CIs and plots.
- Model client abstraction (`src/clients/model_clients.py`): thin it, or replace with Inspect's model API.
- The rubric prompts in `src/rubrics/`: only as raw material for the entailment prompts.

Delete or archive:
- All duplicate module trees (root `src/*.py` versus `src/{evaluation,rubrics,personas,knowledge,data,core}/`).
- The weighted six-dimension scoring and the 5-model judge ensemble with hand-set weights.
- `web/` (Next.js and FastAPI), Docker, Sphinx site, `docs/leaderboard.rst` machinery. A static results page is enough.
- UMLS, RxNorm, SNOMED and DrugBank grounding from the core path. Make them optional extras at most, because of license gating.
- `evaluation/validate_judge.py`: keep the idea, rewrite it to validate the per-fact entailment judge against clinician labels with Cohen's kappa.

Repo hygiene: rename is allowed. If Plainline is chosen, rename to `plainline` (or keep `medexplain-evals` as a redirect note), restructure as a pip-installable package with an `inspect_ai` task, and add a 1 page datasheet and a model-score table that was actually run.

## 6. Plan and milestones (estimates)

1. Week 1: collapse the repo, write the item schema, pull 150 open sources, auto-extract candidate facts.
2. Weeks 2 to 3: recruit 2 clinicians (ask the Columbia medical network or a paid platform); tier facts; compute kappa. Decide the budget before starting; clinician time is the main expense.
3. Week 3: implement the entailment judge, validate on 300 clinician-labelled facts, record judge-versus-clinician agreement. If agreement is poor, switch to rule or string-match checks for the subset where possible.
4. Week 4: run 10 to 15 models, produce adaptation-tax and disparity tables with CIs, write the datasheet and the Inspect task.
5. Then: public dev split, private test split, a short paper. Decide venue by the then-current D&B deadline (not checked here).

Open questions for the owner: budget for clinicians and any crowd work; whether a Columbia IRB exempt determination is wanted for a later Teachback study; whether to keep the GSoC and DeepMind attribution in the new README; and which model families to run first, given API spend.
