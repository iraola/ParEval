# drivers/tools

Maintenance and diagnostic utilities for the evaluation pipeline. 
Most of these take JSON paths as arguments and can be run from anywhere; the
SLURM scripts invoke them as `tools/<script>.py` from `drivers/`.

| Script | Purpose |
|--------|---------|
| `filter-correct.py` | Prepare a correctness output JSON for a scaling re-run (passing → raw string, failing → carried over). |
| `sync-scaling-with-correctness.py` | Re-sync `output_scaling_*.json` for selected problems from an updated correctness JSON. |
| `reset-problem-outputs.py` | Reset evaluated outputs of selected problems back to raw strings so a resumed `run-all.py` re-evaluates only those. |
| `generate_resource_slots.py` | Generate SLURM resource-slot definitions for the parallel/scaling launchers. |
| `list-array-models.py` | Print the SLURM array-index → model mapping for a `generate/outputs/<DIR>`. |
| `update-problem-sizes.py` | Interpolate per-model problem sizes in `problem-sizes.json`. |
| `create-driver-template.py` | Scaffold a new problem's driver template. |
| `generate-test-outputs.py` | Generate reference/test outputs for the harness. |
