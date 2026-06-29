#!/usr/bin/env python
"""List the array index → model mapping for the SLURM driver scripts.

The slurm scripts pick a model by SLURM_ARRAY_TASK_ID from:
    ls generate/outputs/<DIR>/output-*.json | sort -f | sed 's|.*/output-||;s|\\.json$||'

This prints that same enumeration so you know which --array indices to pass.

Usage: python list-array-models.py DIR_NAME
"""
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {Path(sys.argv[0]).name} DIR_NAME")
    dir_name = sys.argv[1]

    # generate/outputs/<DIR> relative to the repo root (this file is in drivers/tools/)
    outputs_dir = Path(__file__).resolve().parents[2] / "generate" / "outputs" / dir_name
    if not outputs_dir.is_dir():
        sys.exit(f"Directory not found: {outputs_dir}")

    # Match the shell pipeline: strip "output-" prefix and ".json" suffix, sort -f
    models = [p.name[len("output-"):-len(".json")]
              for p in outputs_dir.glob("output-*.json")]
    models.sort(key=str.lower)  # sort -f (case-insensitive)

    if not models:
        sys.exit(f"No output-*.json files in {outputs_dir}")

    width = len(str(len(models) - 1))
    for idx, model in enumerate(models):
        print(f"{idx:>{width}}  {model}")
    print(f"\n{len(models)} models  ->  --array=0-{len(models) - 1}")

if __name__ == "__main__":
    main()
