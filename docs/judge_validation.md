# Validating the fact judge

The judge answers one narrow question per fact: is this fact stated in the explanation, contradicted by it, or absent. Its prompt is `JUDGE_SYSTEM` in `plainline/prompts.py` (version `judge-v1`, stored with every judged output). It never sees the source or the tiers.

Status: not yet validated against human labels. No human labels exist in this repository.

## Workflow for a validation round

1. Run the benchmark to get `runs/<name>/scored.jsonl`.
2. `plainline export-labels --run runs/<name> --out labels.jsonl` writes one row per (output, fact) with the explanation, the fact, and the judge's verdict. `human_stated` is null.
3. Two or more labelers fill `human_stated` (true or false) without seeing `judge_stated`. For adjudication, keep one file per labeler and merge by majority, then resolve ties by discussion.
4. `plainline agreement --labels labels.jsonl` prints the number of rows, raw agreement and Cohen's kappa between the judge and the human labels.

The same function (`plainline.judge.agreement`) works on any pair of lists of `{key, stated}`, so inter-annotator agreement between two humans uses the same code.

## Targets before treating CFR as a measurement

- At least 300 labelled (output, fact) pairs, stratified over tiers and over reader conditions.
- Report kappa overall and for must_keep facts alone. Report the judge's false-stated rate on must_keep facts separately, since that error hides omissions.
- If kappa is low for a fact type (for example numeric limits), switch that type to a rule or string check.
- Re-run with a second judge model from a different family and report the spread, to bound self-preference.
