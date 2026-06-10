""" Plot speedup/efficiency scaling curves from analysis/outputs/<dir>/curve_*.csv files.

The curve_*.csv files are produced by metrics-scaling.py and hold one row per
(model, problem type) with speedup_{n}@{k} / efficiency_{n}@{k} columns for each
resource count n. This script draws speedup-vs-n and efficiency-vs-n, one line per
model (mean across problem types).
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

# ── Model filter presets ──────────────────────────────────────────────────────

def _is_reasoning(name: str) -> bool:
    return bool(re.search(r'R1|Qwen3|Magistral', name, re.IGNORECASE))

def _is_instruct(name: str) -> bool:
    return bool(re.search(r'Instruct|Codestral|Mistral|Mixtral|Magistral|gpt-oss', name, re.IGNORECASE))

PRESETS = {
    "all":          lambda models: models,
    "instruct":     lambda models: [m for m in models if _is_instruct(m)],
    "no-reasoning": lambda models: [m for m in models if not _is_reasoning(m)],
    "reasoning":    lambda models: [m for m in models if _is_reasoning(m)],
    "base":         lambda models: [m for m in models if not _is_instruct(m) and not _is_reasoning(m)],
}

# ── Style ─────────────────────────────────────────────────────────────────────

def _style(ax):
    """Linear-axis styling (these plots show ratios/counts, not percentages)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_metrics_dir(value: str) -> str:
    if os.path.isdir(value):
        return value
    candidate = os.path.join(os.path.dirname(__file__), "outputs", value)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"'{value}' is not a valid directory and was not found under outputs/")

def default_output_dir(metrics_dir: str) -> str:
    name = os.path.basename(os.path.normpath(metrics_dir))
    return os.path.join(os.path.dirname(__file__), "plots", name)

def _short_name(name: str) -> str:
    name = re.sub(r'-Instruct$', '', name)
    name = re.sub(r'-v\d+\.\d+$', '', name)
    return name

def _save_both(fig, output_dir, base_fname):
    """Save twice: with titles (_with_title suffix) and without titles (no suffix)."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, base_fname + "_with_title.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
        for ax in fig.axes:
            ax.set_title("")
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
        fig.tight_layout()
        path = os.path.join(output_dir, base_fname + ".png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_curves(metrics_dir: str) -> pd.DataFrame:
    frames = []
    for fname in sorted(os.listdir(metrics_dir)):
        if not fname.startswith("curve_") or not fname.endswith(".csv"):
            continue
        frames.append(pd.read_csv(os.path.join(metrics_dir, fname)))
    if not frames:
        raise FileNotFoundError(f"No curve_*.csv files found in {metrics_dir}")
    return pd.concat(frames, ignore_index=True)

def to_long(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, list]:
    """Melt the wide speedup_{n}@{k} / efficiency_{n}@{k} columns into long form.

    Returns (long_df with columns [model, problem type, n, speedup, efficiency], ns).
    """
    ns = sorted({
        int(m.group(1))
        for c in df.columns
        for m in [re.fullmatch(rf'speedup_(\d+)@{k}', c)] if m
    })
    if not ns:
        raise ValueError(f"No speedup_*@{k} columns found; available: {list(df.columns)}")

    parts = []
    for n in ns:
        scol, ecol = f"speedup_{n}@{k}", f"efficiency_{n}@{k}"
        sub = df[["model", "problem type", scol, ecol]].rename(
            columns={scol: "speedup", ecol: "efficiency"})
        sub["n"] = n
        parts.append(sub)
    return pd.concat(parts, ignore_index=True), ns

def filter_models(df: pd.DataFrame, args, ns: list) -> pd.DataFrame:
    all_models = sorted(df["model"].unique())
    if args.models:
        selected = [m for m in args.models if m in all_models]
    elif args.filter in PRESETS:
        selected = PRESETS[args.filter](all_models)
    else:
        raise ValueError(f"Unknown --filter value '{args.filter}'. Valid: {list(PRESETS.keys())}.")
    if args.exclude:
        selected = [m for m in selected if m not in args.exclude]
    df = df[df["model"].isin(selected)].copy()
    if args.top:
        # rank by mean speedup at the largest resource count
        max_n = max(ns)
        top_models = (df[df["n"] == max_n].groupby("model")["speedup"].mean()
                        .nlargest(args.top).index.tolist())
        df = df[df["model"].isin(top_models)].copy()
    return df

# ── Plot ──────────────────────────────────────────────────────────────────────

def _model_lines(ax, pivot, ns):
    n_models = pivot.shape[1]
    palette = sns.color_palette("tab10" if n_models <= 10 else "tab20", n_colors=n_models)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", ">", "<", "p"]
    for i, model in enumerate(pivot.columns):
        ax.plot(ns, pivot[model].reindex(ns).values,
                marker=markers[i % len(markers)],
                color=palette[i],
                linewidth=1.8, markersize=6,
                label=_short_name(model))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("resources (n)")

def plot_speedup_vs_n(long_df, ns, output_dir, suffix=""):
    agg = long_df.groupby(["model", "n"])["speedup"].mean().reset_index()
    pivot = agg.pivot(index="n", columns="model", values="speedup")
    # order models by speedup at the largest n (best on top of legend)
    order = pivot.loc[max(ns)].sort_values(ascending=False).index.tolist()
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    _model_lines(ax, pivot, ns)
    ax.plot(ns, ns, color="#555555", linestyle="--", linewidth=1.0,
            alpha=0.7, label="ideal (linear)")
    ax.set_ylabel("speedup")
    # cap to observed speedups so the linear-ideal reference doesn't squash the data
    ax.set_ylim(0, max(1.0, float(pivot.max().max())) * 1.3)
    ax.set_title("Speedup vs resources  (mean across problem types)")
    _style(ax)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    _save_both(fig, output_dir, f"speedup_vs_n{suffix}")

def plot_efficiency_vs_n(long_df, ns, output_dir, suffix=""):
    agg = long_df.groupby(["model", "n"])["efficiency"].mean().reset_index()
    pivot = agg.pivot(index="n", columns="model", values="efficiency")
    order = pivot.loc[min(ns)].sort_values(ascending=False).index.tolist()
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    _model_lines(ax, pivot, ns)
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0,
               alpha=0.7, label="ideal (100%)")
    ax.set_ylabel("efficiency")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Efficiency vs resources  (mean across problem types)")
    _style(ax)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    _save_both(fig, output_dir, f"efficiency_vs_n{suffix}")

# ── Argument parsing ──────────────────────────────────────────────────────────

def fname_suffix(args) -> str:
    parts = []
    if args.models:
        parts.append("custom")
    elif args.filter != "all":
        parts.append(args.filter.replace("-", ""))
    if args.exclude:
        parts.append("excl")
    if args.top:
        parts.append(f"top{args.top}")
    return ("_" + "_".join(parts)) if parts else ""

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_dir", type=str,
        help="Directory with curve_*.csv files, or a run name like 'kernel-scaling'.")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
        help="Directory to save plots. Defaults to plots/<run-name>.")
    parser.add_argument("--filter", type=str, default="all",
        help=f"Named model subset: {list(PRESETS.keys())}. Default: all.")
    parser.add_argument("--models", nargs="+", type=str, default=None,
        help="Explicit list of model names to include (overrides --filter).")
    parser.add_argument("--exclude", nargs="+", type=str, default=None,
        help="Model names to exclude (applied after --filter / --models).")
    parser.add_argument("--top", type=int, default=None,
        help="Keep only the top-N models by speedup at the largest resource count.")
    parser.add_argument("-k", type=int, default=1,
        help="K value of the speedup_{n}@k / efficiency_{n}@k columns to plot (default: 1).")
    return parser.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    args.metrics_dir = resolve_metrics_dir(args.metrics_dir)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.metrics_dir)

    df = load_curves(args.metrics_dir)
    long_df, ns = to_long(df, args.k)
    long_df = filter_models(long_df, args, ns)

    if long_df.empty:
        print("No models left after filtering.")
        return

    models_used = sorted(long_df["model"].unique())
    print(f"Plotting scaling for {len(models_used)} models over n={ns}: {models_used}")

    suffix = fname_suffix(args)
    plot_speedup_vs_n(long_df, ns, args.output_dir, suffix)
    plot_efficiency_vs_n(long_df, ns, args.output_dir, suffix)


if __name__ == "__main__":
    main()
