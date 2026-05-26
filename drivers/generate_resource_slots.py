#!/usr/bin/env python3
"""Generate resource slot definitions for parallel runcompss execution.

Divides the local node's CPUs into fixed-size slots and outputs a JSON slot
manifest. XML files are are written per-run by python_driver_wrapper.py into
TMPDIR.

Usage:
    python3 generate_resource_slots.py \\
        --cpus-per-node 112 \\
        --cpus-per-slot 5 \\
        --output slots.json

    # Scaling mode (one slot = whole node):
    python3 generate_resource_slots.py \\
        --cpus-per-node 112 \\
        --cpus-per-slot 112 \\
        --output slots_scaling.json
"""
import argparse
import json
import sys

_BASE_PORT = 43001


def generate_slots(cpus_per_node: int, cpus_per_slot: int) -> list[dict]:
    """Generate slot definitions. Returns the slot manifest."""
    slots = []
    slot_index = 0
    cpu = 0
    while cpu + cpus_per_slot <= cpus_per_node:
        slots.append({
            "cpu_start": cpu,
            "slot_cpus": cpus_per_slot,
            "base_port": _BASE_PORT + 2 * slot_index,
        })
        cpu += cpus_per_slot
        slot_index += 1
    return slots


def main():
    parser = argparse.ArgumentParser(description="Generate resource slots for parallel runcompss.")
    parser.add_argument("--cpus-per-node", type=int, required=True, help="Total CPUs available on this node.")
    parser.add_argument("--cpus-per-slot", type=int, default=5, help="CPUs assigned to each runcompss instance (default: 5).")
    parser.add_argument("--output", required=True, help="Path for output slots JSON file.")
    args = parser.parse_args()

    if args.cpus_per_node % args.cpus_per_slot != 0:
        print(f"Warning: {args.cpus_per_node} CPUs not evenly divisible by {args.cpus_per_slot}; "
              f"{args.cpus_per_node % args.cpus_per_slot} CPUs will be unused.", file=sys.stderr)

    slots = generate_slots(args.cpus_per_node, args.cpus_per_slot)
    n = len(slots)
    print(f"Generated {n} slot(s) ({n} × {args.cpus_per_slot} CPUs, base_port {_BASE_PORT}).")

    with open(args.output, "w") as f:
        json.dump(slots, f, indent=2)
    print(f"Wrote slot manifest to {args.output}.")


if __name__ == "__main__":
    main()
