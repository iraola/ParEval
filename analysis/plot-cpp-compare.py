"""Compare PyCOMPSs pass@1 against the original C++ ParEval paper.

Draws a grouped bar per shared model: the paper's *parallel* pass@1 (digitized
into data/pareval_cpp_pass1.csv) next to our PyCOMPSs pass@1, read from a
metrics dir (default outputs/kernel) and averaged across problem types, the same
reduction plot-compare.py uses. Only the models in MODEL_MAP (present in both
benchmarks) are plotted; the paper's StarCoderBase is intentionally left out, as
it does not match a model in our set.

Usage:
    python plot-cpp-compare.py
    python plot-cpp-compare.py kernel-guided -o plots/cpp-compare-guided
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

import plotstyle as ps

# (paper label in the CSV, our metrics model id, display label). Order = bar order.
MODEL_MAP = [
    ("CL-13B", "CodeLlama-13b-hf",         "CL-13B"),
    ("Phind-V2", "Phind-CodeLlama-34B-v2", "Phind-V2"),
    ("GPT-4", "openai-gpt-4-turbo-2024-04-09", "GPT-4"),
]

CPP_COLOR = "#D9822B"       # orange, echoing the paper figure
PYCOMPSS_COLOR = "#1BB863"  # green
CPP_LABEL = "C++"
PYCOMPSS_LABEL = "task-based (PyCOMPSs)"

_DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "pareval_cpp_pass1.csv")


def _our_pass1(metrics_dir: str) -> pd.Series:
    """pass@1 per model, averaged across problem types (fraction in [0, 1])."""
    df = ps.load_csvs(metrics_dir, "metrics_")
    return df.groupby("model")["pass@1"].mean()


def _cpp_pass1() -> pd.Series:
    """Paper parallel pass@1 per label, converted from percent to fraction."""
    df = pd.read_csv(_DATA_CSV, comment="#")
    return df.set_index("model")["parallel_pass1"] / 100.0


def plot(metrics_dir: str, output_dir: str) -> None:
    cpp = _cpp_pass1()
    ours = _our_pass1(metrics_dir)

    rows = []
    for cpp_label, model_id, display in MODEL_MAP:
        if cpp_label not in cpp.index:
            raise KeyError(f"'{cpp_label}' missing from {_DATA_CSV}")
        if model_id not in ours.index:
            raise KeyError(f"'{model_id}' has no metrics in {metrics_dir}")
        rows.append((display, cpp[cpp_label], ours[model_id]))

    labels = [r[0] for r in rows]
    cpp_vals = [r[1] for r in rows]
    our_vals = [r[2] for r in rows]

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2.0, 4.5))

    b1 = ax.bar([i - width / 2 for i in x], cpp_vals, width, label=CPP_LABEL,
                color=CPP_COLOR, hatch="xx", edgecolor="black", linewidth=0.5)
    b2 = ax.bar([i + width / 2 for i in x], our_vals, width, label=PYCOMPSS_LABEL,
                color=PYCOMPSS_COLOR, hatch="//", edgecolor="black", linewidth=0.5)

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h * 100:.1f}", (rect.get_x() + rect.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("pass@1")
    ax.set_ylim(0, max(cpp_vals + our_vals) * 1.18)
    ax.set_title("Parallel pass@1: C++ ParEval vs PyCOMPSs")
    ps.style_axis(ax, pct_axis="y")
    ax.legend(frameon=False, loc="upper left")

    ps.save_both(fig, output_dir, "cpp_vs_pycompss_pass1")


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", nargs="?", default="kernel",
        help="Metrics run name or path for our pass@1 (default: kernel).")
    parser.add_argument("-o", "--output", default=None,
        help="Output dir for the figure. Defaults to plots/cpp-compare.")
    return parser.parse_args()


def main():
    args = get_args()
    metrics_dir = ps.resolve_metrics_dir(args.dir)
    output_dir = args.output or os.path.join(
        os.path.dirname(__file__), "plots", "cpp-compare")
    plot(metrics_dir, output_dir)


if __name__ == "__main__":
    main()
