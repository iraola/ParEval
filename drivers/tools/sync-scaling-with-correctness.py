#!/usr/bin/env python3
"""Re-sync output_scaling_*.json entries for selected problems from an
updated output_drivers_*.json (correctness).

output_scaling_*.json is derived from a correctness run via filter-correct.py:
each output becomes a raw string (are_any_valid == True, to be scaling-
evaluated) or stays the correctness result dict (failure carryover, skipped
by run-all.py). When correctness is re-run for specific problems (e.g. after
fixing a harness/baseline bug), the scaling JSON's entries for those problems
become stale. This re-applies filter-correct.py's per-output rule using the
new correctness results, but only for the selected problems -- every other
problem's scaling entries (including any real scaling results) are left
untouched.

Targets can be given as:
  - an explicit (correctness, scaling) file pair
  - --dir DIR, which pairs DIR/output_drivers_<model>.json with
    DIR/output_scaling_<model>.json for every model found

Problems are matched by exact prompt name, name prefix (e.g. '45' matches
'45_sparse_la_sparse_solve'), or problem_type family (e.g. 'sparse_la').

Usage:
    # Single model
    python3 sync-scaling-with-correctness.py \\
        outputs/kernel/output_drivers_CodeLlama-13b-hf.json \\
        outputs/kernel/output_scaling_CodeLlama-13b-hf.json \\
        --problems 05_fft_inverse_fft 45_sparse_la_sparse_solve

    # All models in a directory
    python3 sync-scaling-with-correctness.py --dir outputs/kernel \\
        --problems 05_fft_inverse_fft 45_sparse_la_sparse_solve --dry-run
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


def filtered_output(correctness_output):
    """Apply filter-correct.py's per-output rule to a correctness output."""
    if isinstance(correctness_output, dict) and correctness_output.get("are_any_valid", False):
        if "generated_output" not in correctness_output:
            raise ValueError("are_any_valid=True output is missing 'generated_output'.")
        return correctness_output["generated_output"]
    return correctness_output


def is_scaling_evaluated(output) -> bool:
    if not isinstance(output, dict):
        return False
    return "problem_size" in output or len(output.get("runs") or []) > 1


def sync_pair(correctness_path, scaling_path, selectors, dry_run):
    with open(correctness_path) as f:
        corr_data = json.load(f)
    with open(scaling_path) as f:
        scal_data = json.load(f)

    corr_by_name = {p["name"]: p for p in corr_data}

    per_problem = {}
    for scal_prompt in scal_data:
        if not matches(scal_prompt, selectors):
            continue
        name = scal_prompt["name"]
        corr_prompt = corr_by_name.get(name)
        if corr_prompt is None:
            raise ValueError(f"Prompt '{name}' not found in {correctness_path}")

        corr_outputs = corr_prompt.get("outputs", [])
        scal_outputs = scal_prompt.get("outputs", [])
        if len(corr_outputs) != len(scal_outputs):
            raise ValueError(f"Prompt '{name}': output count mismatch "
                             f"({len(corr_outputs)} correctness vs {len(scal_outputs)} scaling)")

        new_outputs = []
        n_promoted = n_demoted = n_discarded_scaling = 0
        for corr_o, scal_o in zip(corr_outputs, scal_outputs):
            new_o = filtered_output(corr_o)
            was_str = isinstance(scal_o, str)
            is_str = isinstance(new_o, str)
            if is_str and not was_str:
                n_promoted += 1
            elif was_str and not is_str:
                n_demoted += 1
            elif not was_str and not is_str and is_scaling_evaluated(scal_o):
                n_discarded_scaling += 1
            new_outputs.append(new_o)

        scal_prompt["outputs"] = new_outputs
        per_problem[name] = (n_promoted, n_demoted, n_discarded_scaling)

    if not dry_run and per_problem:
        with open(scaling_path, "w") as f:
            json.dump(scal_data, f, indent=4)

    return per_problem


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("correctness", nargs="?", help="output_drivers_<model>.json (new correctness results).")
    parser.add_argument("scaling", nargs="?", help="output_scaling_<model>.json, edited in place.")
    parser.add_argument("--dir", help="Directory containing paired output_drivers_*.json / output_scaling_*.json.")
    parser.add_argument("--problems", nargs="+", required=True,
                        help="Prompt names, name prefixes (e.g. '45'), or problem_type families (e.g. 'sparse_la').")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()

    pairs = []
    if args.dir:
        for corr_path in sorted(glob.glob(os.path.join(args.dir, "output_drivers_*.json"))):
            model = os.path.basename(corr_path)[len("output_drivers_"):-len(".json")]
            scal_path = os.path.join(args.dir, f"output_scaling_{model}.json")
            if not os.path.exists(scal_path):
                print(f"{scal_path}: skipped (no scaling file)")
                continue
            pairs.append((corr_path, scal_path))
    if args.correctness and args.scaling:
        pairs.append((args.correctness, args.scaling))
    if not pairs:
        parser.error("no input pairs (pass correctness+scaling FILES or --dir)")

    for corr_path, scal_path in pairs:
        per_problem = sync_pair(corr_path, scal_path, args.problems, args.dry_run)
        action = "would update" if args.dry_run else "updated"
        total_promoted = sum(v[0] for v in per_problem.values())
        total_demoted = sum(v[1] for v in per_problem.values())
        total_discarded = sum(v[2] for v in per_problem.values())
        print(f"{scal_path} [{action}]: "
              f"{total_promoted} promoted to scaling, "
              f"{total_demoted} demoted to fail-carryover, "
              f"{total_discarded} existing scaling results discarded")
        for name, (n_p, n_d, n_disc) in sorted(per_problem.items()):
            print(f"    {name}: promoted={n_p} demoted={n_d} discarded_scaling={n_disc}")


if __name__ == "__main__":
    main()
