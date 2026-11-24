# Golden files

Recorded from the original Python implementation of Plainline (v0.1.0) and used by the Rust tests
as parity references. They were produced by running the Python code and the Rust core side by side
on the pilot data (see the pull request "Port core to Rust"); the generator script lives in git
history at commit `a086521` (`tests/golden/generate_from_python.py`) and is not part of the tree.

- `rng.json`: NumPy `default_rng(seed).integers(0, n, size)` draws (NumPy 2.4.6), for the bootstrap RNG.
- `metrics_cases.json`, `score_output.json`, `readability.json`, `verdicts.json`, `agreement.json`,
  `pipeline_text.json`: inputs with the Python outputs (NaN recorded as null).
- `mock_*_results.md`, `mock_*_summary.json`, `mock_runs.json`: full mock runs over the pilot data
  (`plainline run --mock`), with a SHA-256 of the Python `scored.jsonl`.
