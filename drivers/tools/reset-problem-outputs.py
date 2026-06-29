#!/usr/bin/env python3
"""Reset evaluated outputs of selected problems back to raw strings so a
resumed run-all.py re-evaluates only those problems.

Two JSON shapes are handled, based on filename or --json-type:
  - correctness (output_drivers_*.json): each dict output is a full
    correctness result (1 run). ALL dict outputs for matching prompts are
    reset -- the whole result is considered stale.
  - scaling (output_scaling_*.json): dict outputs are either
    scaling-evaluated (>1 run / has 'problem_size') or correctness-failure
    carryovers from filter-correct.py (1 run, no 'problem_size'). By default
    only scaling-evaluated outputs are reset, preserving carryovers so
    run-all.py keeps skipping known-failed outputs. Pass --full to also
    reset carryovers (e.g. after correctness itself was redone for these
    problems and the carryover results are now stale too).

Targets can be given as:
  - explicit file paths (positional FILES), type inferred from filename
  - --dir DIR --json-type {correctness,scaling}  (expands to
    DIR/output_drivers_*.json or DIR/output_scaling_*.json)

Problems are matched by exact prompt name, name prefix (e.g. '45' matches
'45_sparse_la_sparse_solve'), or problem_type family (e.g. 'sparse_la').

Usage:
    # Reset correctness results for 05/45 across all kernel models
    python3 reset-problem-outputs.py --dir outputs/kernel --json-type correctness \\
        --problems 05_fft_inverse_fft 45_sparse_la_sparse_solve

    # Dry run on a single model's scaling file, full reset
    python3 reset-problem-outputs.py outputs/kernel/output_scaling_CodeLlama-13b-hf.json \\
        --problems 45 --full --dry-run
"""
import argparse
import glob
import json
import os


def matches(prompt: dict, selectors) -> bool:
    """True if the prompt's name/prefix or problem_type matches a selector."""
    name = prompt.get("name", "")
    ptype = prompt.get("problem_type", "")
    return any(name == s or name.startswith(s + "_") or ptype == s for s in selectors)


def is_scaling_evaluated(output) -> bool:
    if not isinstance(output, dict):
        return False
    return "problem_size" in output or len(output.get("runs") or []) > 1


def reset_prompt_outputs(prompt: dict, json_type: str, full: bool) -> int:
    new_outputs = []
    n_reset = 0
    for o in prompt.get("outputs", []):
        if not isinstance(o, dict):
            new_outputs.append(o)
            continue
        if json_type == "correctness" or full or is_scaling_evaluated(o):
            if "generated_output" not in o:
                raise ValueError(f"Output in prompt '{prompt.get('name')}' is missing 'generated_output'.")
            new_outputs.append(o["generated_output"])
            n_reset += 1
        else:
            new_outputs.append(o)
    prompt["outputs"] = new_outputs
    return n_reset


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Output JSON file(s), edited in place.")
    parser.add_argument("--dir", help="Directory to glob files from (combined with --json-type).")
    parser.add_argument("--json-type", choices=["correctness", "scaling"],
                        help="Selects output_drivers_*.json or output_scaling_*.json when using --dir, "
                             "and overrides type inference for explicit FILES.")
    parser.add_argument("--problems", nargs="+", required=True,
                        help="Prompt names, name prefixes (e.g. '45'), or problem_type families (e.g. 'sparse_la').")
    parser.add_argument("--full", action="store_true",
                        help="For scaling files, also reset correctness-failure carryover outputs "
                             "(1 run, no 'problem_size'), not just scaling-evaluated ones.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be reset without writing.")
    args = parser.parse_args()

    files = list(args.files)
    if args.dir:
        if not args.json_type:
            parser.error("--dir requires --json-type")
        pattern = "output_drivers_*.json" if args.json_type == "correctness" else "output_scaling_*.json"
        files += sorted(glob.glob(os.path.join(args.dir, pattern)))
    if not files:
        parser.error("no input files (pass FILES or --dir/--json-type)")

    for path in files:
        json_type = args.json_type
        if json_type is None:
            base = os.path.basename(path)
            if base.startswith("output_drivers_"):
                json_type = "correctness"
            elif base.startswith("output_scaling_"):
                json_type = "scaling"
            else:
                parser.error(f"cannot infer json type for {path}; pass --json-type")

        with open(path) as f:
            data = json.load(f)

        n_reset = 0
        per_problem = {}
        for prompt in data:
            if not matches(prompt, args.problems):
                continue
            n = reset_prompt_outputs(prompt, json_type, args.full)
            if n:
                per_problem[prompt["name"]] = n
                n_reset += n

        if not args.dry_run and n_reset:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)

        action = "would reset" if args.dry_run else "reset"
        print(f"{path} [{json_type}]: {action} {n_reset} output(s)")
        for name, n in sorted(per_problem.items()):
            print(f"    {name}: {n}")


if __name__ == "__main__":
    main()
