#!/usr/bin/env python
"""Quantify whether truncated reasoning outputs are degenerate loops or
genuinely-progressing reasoning that ran out of budget.

For each model+suite, find samples whose cleaned output is empty AND whose raw
output is a truncated reasoning block (open <think> without close, or Harmony
with no `assistantfinal`). For those, measure a loopiness ratio:

  loop_ratio = 1 - (unique non-empty lines / total non-empty lines)

High ratio (-> 1.0) = lots of verbatim repetition (degenerate loop).
Low ratio  (-> 0.0) = mostly novel lines (progressing reasoning, budget-starved).

Run from generate/ with the repo venv.
"""
import glob
import json
import os
from collections import Counter

SUITES = ["outputs/kernel", "outputs/kernel-guided"]


def is_truncated_reasoning(raw):
    if "<think>" in raw and "</think>" not in raw:
        return True
    # Harmony: reasoning then `assistantfinal`; if final marker never reached
    if ("<|channel|>analysis" in raw or "<|start|>assistant" in raw) and "assistantfinal" not in raw:
        return True
    return False


def loop_ratio(raw):
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return 0.0, 0, 0
    uniq = len(set(lines))
    return 1.0 - uniq / len(lines), len(lines), uniq


def top_repeat(raw):
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return 0, ""
    ln, n = Counter(lines).most_common(1)[0]
    return n, ln


def model_name(path):
    return os.path.basename(path)[len("output-"):-len(".json")]


def main():
    for suite in SUITES:
        for f in sorted(glob.glob(os.path.join(suite, "output-*.json"))):
            data = json.load(open(f))
            trunc = []
            for e in data:
                outs = e.get("outputs", [])
                raws = e.get("raw_outputs", [])
                for i, (o, r) in enumerate(zip(outs, raws)):
                    if (o or "").strip():
                        continue
                    if not is_truncated_reasoning(r or ""):
                        continue
                    lr, tot, uq = loop_ratio(r)
                    n, ln = top_repeat(r)
                    trunc.append((lr, tot, uq, n, len(r), f"{e['name']}#{i}", ln))
            if not trunc:
                continue
            trunc.sort(reverse=True)
            ratios = [t[0] for t in trunc]
            avg = sum(ratios) / len(ratios)
            looped = sum(1 for r in ratios if r >= 0.5)
            print(f"\n### {suite} / {model_name(f)}")
            print(f"  truncated-reasoning empties: {len(trunc)} | "
                  f"avg loop_ratio={avg:.2f} | loopy(>=0.5): {looped}/{len(trunc)}")
            # Show the 3 LEAST loopy (best case for "needs more tokens")
            print("  least-loopy (would benefit most from more tokens):")
            for lr, tot, uq, n, rl, tag, ln in sorted(trunc)[:3]:
                print(f"    loop={lr:.2f} lines={tot} uniq={uq} maxrep={n} rawlen={rl} {tag}")
                print(f"        top-line: {ln[:80]!r}")


if __name__ == "__main__":
    main()
