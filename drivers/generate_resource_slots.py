#!/usr/bin/env python3
"""Generate resource slot definitions for parallel runcompss execution.

Divides the local node's CPUs into fixed-size slots, writes one COMPSs
resources XML per slot, and outputs a JSON slot manifest.

Usage:
    python3 generate_resource_slots.py \\
        --cpus-per-node 100 \\
        --cpus-per-slot 5 \\
        --resources-dir resources/ \\
        --output slots.json
"""
import argparse
import json
import os
import sys

_XML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<ResourcesList>
    <ComputeNode Name="localhost">
        <Processor Name="MainProcessor">
            <ComputingUnits>{computing_units}</ComputingUnits>
        </Processor>
        <Adaptors>
            <Adaptor Name="es.bsc.compss.nio.master.NIOAdaptor">
                <SubmissionSystem>
                    <Interactive/>
                </SubmissionSystem>
                <Ports>
                    <MinPort>{min_port}</MinPort>
                    <MaxPort>{max_port}</MaxPort>
                </Ports>
            </Adaptor>
            <Adaptor Name="es.bsc.compss.gat.master.GATAdaptor">
                <SubmissionSystem>
                    <Batch>
                        <Queue>sequential</Queue>
                    </Batch>
                    <Interactive/>
                </SubmissionSystem>
                <BrokerAdaptor>sshtrilead</BrokerAdaptor>
            </Adaptor>
        </Adaptors>
    </ComputeNode>
</ResourcesList>
"""

def generate_slots(cpus_per_node: int, cpus_per_slot: int, resources_dir: str) -> list[dict]:
    """Generate slot definitions and write XML files. Returns the slot manifest."""
    os.makedirs(resources_dir, exist_ok=True)
    slots = []
    slot_index = 0
    cpu = 0
    while cpu + cpus_per_slot <= cpus_per_node:
        cpu_affinity = f"{cpu}-{cpu + cpus_per_slot - 1}"
        min_port = 43001 + 2 * slot_index
        max_port = 43002 + 2 * slot_index

        xml_name = f"resources_slot{slot_index}.xml"
        xml_path = os.path.abspath(os.path.join(resources_dir, xml_name))
        with open(xml_path, "w") as f:
            f.write(_XML_TEMPLATE.format(
                computing_units=cpus_per_slot,
                min_port=min_port,
                max_port=max_port,
            ))

        slots.append({
            "cpu_affinity": cpu_affinity,
            "resources_xml": xml_path,
        })

        cpu += cpus_per_slot
        slot_index += 1

    return slots


def main():
    parser = argparse.ArgumentParser(description="Generate resource slots for parallel runcompss.")
    parser.add_argument("--cpus-per-node", type=int, required=True, help="Total CPUs available on this node.")
    parser.add_argument("--cpus-per-slot", type=int, default=5, help="CPUs assigned to each runcompss instance (default: 5).")
    parser.add_argument("--resources-dir", default="resources/", help="Directory for generated XML files (default: resources/).")
    parser.add_argument("--output", required=True, help="Path for output slots JSON file.")
    args = parser.parse_args()

    if args.cpus_per_node % args.cpus_per_slot != 0:
        print(f"Warning: {args.cpus_per_node} CPUs not evenly divisible by {args.cpus_per_slot}; "
              f"{args.cpus_per_node % args.cpus_per_slot} CPUs will be unused.", file=sys.stderr)

    slots = generate_slots(args.cpus_per_node, args.cpus_per_slot, args.resources_dir)
    print(f"Generated {len(slots)} slots ({args.cpus_per_node // args.cpus_per_slot} slots × {args.cpus_per_slot} CPUs each).")

    with open(args.output, "w") as f:
        json.dump(slots, f, indent=2)
    print(f"Wrote slot manifest to {args.output}.")


if __name__ == "__main__":
    main()
