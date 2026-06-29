#!/usr/bin/env python3
"""Prepare a correctness output JSON for a scaling re-run.

For each output in the JSON:
  - Passed (are_any_valid == True): reset to the raw generated string so
    run-all.py will re-evaluate it with scaling configs.
  - Failed: keep the existing result dict so run-all.py skips it.

This means the scaling output JSON preserves failure data from the correctness
run while only re-executing the outputs that are known to be correct.

Usage:
    python3 filter-correct.py <correctness_output.json> <scaling_output.json>
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Filter correctness output JSON to re-run only passing outputs."
    )
    parser.add_argument("input", help="Correctness driver output JSON.")
    parser.add_argument("output", help="Filtered JSON path for the scaling run.")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    n_reset = 0
    n_kept = 0

    for prompt in data:
        new_outputs = []
        for o in prompt.get("outputs", []):
            if isinstance(o, dict) and o.get("are_any_valid", False):
                if "generated_output" not in o:
                    raise ValueError(
                        f"Output in prompt '{prompt.get('name')}' has are_any_valid=True "
                        f"but is missing 'generated_output'."
                    )
                new_outputs.append(o["generated_output"])
                n_reset += 1
            else:
                new_outputs.append(o)
                n_kept += 1
        prompt["outputs"] = new_outputs

    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)

    n_prompts = len(data)
    print(f"Processed {n_prompts} prompts.")
    print(f"  {n_reset} output(s) reset to string  → will be re-run with scaling configs")
    print(f"  {n_kept} output(s) kept as dict      → skipped (failed correctness or already string)")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
