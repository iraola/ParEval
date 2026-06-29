#!/usr/bin/env python
"""Integrity audit across all generated output JSONs.

Runs the repo's own check_output_integrity over every output-*.json under
generate/outputs/kernel and generate/outputs/kernel-guided, plus a few extra
raw-text checks (literal fence leakage, truncated reasoning), and writes a
markdown report. Run from generate/ with the repo venv:

    cd generate
    ../.venv/bin/python tools/audit-outputs.py
"""
import glob
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

# Allow running from generate/ now that this script lives in generate/tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import check_output_integrity, get_inference_config

SUITES = ["outputs/kernel", "outputs/kernel-guided"]
REPORT = "audit-report.md"


def model_name_from_file(path):
    base = os.path.basename(path)
    return base[len("output-"):-len(".json")]


def raw_list(entry):
    for k in ("raw_outputs", "raw_output", "raw"):
        if k in entry:
            v = entry[k]
            return v if isinstance(v, list) else [v]
    return []


def clean_list(entry):
    for k in ("outputs", "cleaned_outputs", "output", "cleaned"):
        if k in entry:
            v = entry[k]
            return v if isinstance(v, list) else [v]
    return []


def prompt_id(entry):
    for k in ("name", "problem", "problem_name", "id", "prompt_name"):
        if k in entry:
            return entry[k]
    return "?"


def extra_raw_checks(entries):
    """Cheap text-level checks independent of the cleaner."""
    fence_leak = []        # cleaned output still contains a literal ``` line
    odd_fence = []         # raw has an odd number of ``` (unclosed fence)
    truncated_think = []   # raw has <think> but no </think>
    empty_clean = []       # cleaned output is empty/whitespace
    for entry in entries:
        pid = prompt_id(entry)
        for i, c in enumerate(clean_list(entry)):
            c = c or ""
            if any(ln.strip().startswith("```") for ln in c.splitlines()):
                fence_leak.append(f"{pid}#{i}")
            if not c.strip():
                empty_clean.append(f"{pid}#{i}")
        for i, r in enumerate(raw_list(entry)):
            r = r or ""
            if r.count("```") % 2 == 1:
                odd_fence.append(f"{pid}#{i}")
            if "<think>" in r and "</think>" not in r:
                truncated_think.append(f"{pid}#{i}")
    return {
        "fence_leak_in_cleaned": fence_leak,
        "odd_raw_fence_count": odd_fence,
        "truncated_think": truncated_think,
        "empty_cleaned": empty_clean,
    }


def main():
    lines = ["# Output integrity audit\n"]
    grand = {}
    for suite in SUITES:
        files = sorted(glob.glob(os.path.join(suite, "output-*.json")))
        lines.append(f"\n## {suite}  ({len(files)} models)\n")
        if not files:
            lines.append("_no files_\n")
            continue
        lines.append("| model | prompts | samples | builtin flags | fence-leak | odd-fence | trunc-think | empty |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for f in files:
            model = model_name_from_file(f)
            with open(f) as fh:
                data = json.load(fh)
            entries = data if isinstance(data, list) else data.get("results", data.get("prompts", []))
            n_samples = sum(len(clean_list(e)) for e in entries)

            buf = io.StringIO()
            builtin = ""
            try:
                with redirect_stdout(buf):
                    check_output_integrity(entries)
                printed = buf.getvalue().strip().replace("\n", " ⏎ ")
                builtin = printed[-400:] if printed else "ok"
            except Exception as exc:
                builtin = f"ERROR: {exc!r}"

            extra = extra_raw_checks(entries)
            grand.setdefault(suite, {})[model] = extra
            lines.append(
                f"| {model} | {len(entries)} | {n_samples} | {builtin} "
                f"| {len(extra['fence_leak_in_cleaned'])} "
                f"| {len(extra['odd_raw_fence_count'])} "
                f"| {len(extra['truncated_think'])} "
                f"| {len(extra['empty_cleaned'])} |"
            )

    # Detail section: only non-empty findings
    lines.append("\n## Detail (non-zero findings only)\n")
    any_finding = False
    for suite, models in grand.items():
        for model, extra in models.items():
            hits = {k: v for k, v in extra.items() if v}
            if not hits:
                continue
            any_finding = True
            lines.append(f"\n### {suite} / {model}")
            for k, v in hits.items():
                shown = ", ".join(v[:25]) + (f" … (+{len(v)-25})" if len(v) > 25 else "")
                lines.append(f"- **{k}** ({len(v)}): {shown}")
    if not any_finding:
        lines.append("\n_No fence-leak / odd-fence / truncated-think / empty-cleaned findings._\n")

    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {REPORT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
