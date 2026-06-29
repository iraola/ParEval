# generate/tools

Maintenance and diagnostic utilities for the generation pipeline. 
Run them from `generate/` with the repo venv, e.g.:

```
cd generate
../.venv/bin/python tools/clean-outputs.py --help
```

| Script | Purpose |
|--------|---------|
| `clean-outputs.py` | Re-apply `InferenceConfig.clean_output` to existing `raw_outputs` without re-running inference. |
| `audit-outputs.py` | Run `check_output_integrity` over every `output-*.json` plus extra raw-text checks; writes `audit-report.md`. |
| `analyze-truncation.py` | Classify truncated-reasoning empties as degenerate loops vs. budget-starved progress (loopiness ratio). |
| `throughput.py` | Benchmark model inference throughput. |
