#!/usr/bin/env python3
"""
Update problem sizes for a given programming model in problem-sizes.json by
linearly interpolating from the serial size, with bounds:
  serial (1<<LOW_EXP)  -> LOW_VAL
  serial (1<<HIGH_EXP) -> HIGH_VAL
Values are rounded to the nearest multiple of STEP.

Usage: python3 update-pycompss-sizes.py <model> [--dry-run]
         [--low-exp N] [--high-exp N] [--low-val N] [--high-val N] [--step N]

Examples:
  python3 update-pycompss-sizes.py pycompss --dry-run
  python3 update-pycompss-sizes.py kokkos --low-exp 9 --high-exp 24 --low-val 10 --high-val 50
"""

import json
import re
import argparse

DEFAULT_LOW_EXP  = 9
DEFAULT_HIGH_EXP = 24
DEFAULT_LOW_VAL  = 10
DEFAULT_HIGH_VAL = 50
DEFAULT_STEP     = 5


def serial_exponent(size_str: str) -> int | None:
    """Extract the exponent from a size string like '(1<<17)'. Returns None if not parseable."""
    m = re.search(r'1<<(\d+)', size_str)
    return int(m.group(1)) if m else None


def interpolate(exp: int, low_exp: int, high_exp: int, low_val: int, high_val: int, step: int) -> int:
    """Map a serial exponent to a model size, clamped and rounded to the nearest step."""
    ratio = (exp - low_exp) / (high_exp - low_exp)
    raw = low_val + ratio * (high_val - low_val)
    raw = max(low_val, min(high_val, raw))
    return round(raw / step) * step


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help='Programming model field to update (e.g. pycompss, kokkos)')
    parser.add_argument('--dry-run', action='store_true', help='Print changes without writing')
    parser.add_argument('--low-exp',  type=int, default=DEFAULT_LOW_EXP,  help=f'Serial exponent mapped to low-val (default: {DEFAULT_LOW_EXP})')
    parser.add_argument('--high-exp', type=int, default=DEFAULT_HIGH_EXP, help=f'Serial exponent mapped to high-val (default: {DEFAULT_HIGH_EXP})')
    parser.add_argument('--low-val',  type=int, default=DEFAULT_LOW_VAL,  help=f'Value assigned at low-exp (default: {DEFAULT_LOW_VAL})')
    parser.add_argument('--high-val', type=int, default=DEFAULT_HIGH_VAL, help=f'Value assigned at high-exp (default: {DEFAULT_HIGH_VAL})')
    parser.add_argument('--step',     type=int, default=DEFAULT_STEP,     help=f'Round to nearest multiple of this (default: {DEFAULT_STEP})')
    args = parser.parse_args()

    if args.model == 'serial':
        parser.error("'serial' is the reference field and cannot be updated.")

    path = 'problem-sizes.json'
    with open(path) as f:
        data = json.load(f)

    # Validate that the model field exists in at least one entry
    known_models = {k for sizes in data.values() for k in sizes if k != 'serial'}
    if args.model not in known_models:
        parser.error(f"Unknown model '{args.model}'. Known fields: {sorted(known_models)}")

    changes = []
    for problem, sizes in data.items():
        if args.model not in sizes:
            continue
        serial_str = sizes.get('serial', '')
        exp = serial_exponent(serial_str)
        if exp is None:
            print(f"WARNING: could not parse serial size for {problem}: {serial_str!r}")
            continue
        new_val = str(interpolate(exp, args.low_exp, args.high_exp, args.low_val, args.high_val, args.step))
        old_val = sizes[args.model]
        if old_val != new_val:
            changes.append((problem, old_val, new_val))
            sizes[args.model] = new_val

    if not changes:
        print("No changes needed.")
        return

    print(f"{'Problem':<55} {'Old':>10} -> {'New':>4}")
    print('-' * 75)
    for problem, old, new in changes:
        print(f"{problem:<55} {old:>10} -> {new:>4}")

    if args.dry_run:
        print(f"\nDry run: {len(changes)} change(s) not written.")
        return

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    print(f"\nWrote {len(changes)} change(s) to {path}.")


if __name__ == '__main__':
    main()
