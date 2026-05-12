""" Plot metrics from the analysis/outputs/<dir>/metrics_*.csv files. """
import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

# ── Model filter presets ──────────────────────────────────────────────────────
# Each entry is a callable: list[str] -> list[str]

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

HATCHES = ['/', '\\', 'x', '-', '+', 'o', '//', '\\\\', '||', '--', '++', 'xx']

def _style(ax, *, xgrid=False, ygrid=True, pct_axis="y"):
    """Apply consistent styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        if pct_axis == "y":          # only force 20%-step ticks on pct axes
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.6, zorder=0)
    if xgrid:
        if pct_axis == "x":
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(axis="x", which="major", color="#cccccc", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if pct_axis == "y":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    elif pct_axis == "x":
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
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

# ── Argument parsing ──────────────────────────────────────────────────────────

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

def fname_suffix(args) -> str:
    parts = []
    if args.models:
        parts.append("custom")
    elif args.filter != "all":
        parts.append(args.filter.replace("-", ""))   # e.g. no-reasoning → noreasoning
    if args.exclude:
        parts.append("excl")
    if args.top and not re.fullmatch(r'top\d+', args.filter):
        parts.append(f"top{args.top}")
    return ("_" + "_".join(parts)) if parts else ""

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_dir", type=str,
        help="Directory with metrics_*.csv files, or a run name like 'kernel-20'.")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
        help="Directory to save plots. Defaults to plots/<run-name>.")
    parser.add_argument("--filter", type=str, default="all",
        help=(f"Named model subset: {list(PRESETS.keys())} or 'topN' (e.g. top5). "
              "Default: all."))
    parser.add_argument("--models", nargs="+", type=str, default=None,
        help="Explicit list of model names to include (overrides --filter).")
    parser.add_argument("--exclude", nargs="+", type=str, default=None,
        help="Model names to exclude (applied after --filter / --models).")
    parser.add_argument("--top", type=int, default=None,
        help="Keep only the top-N models by overall pass@1 (applied last).")
    parser.add_argument("-k", nargs="+", type=int, default=[1, 5, 10, 20],
        help="K values for the k-vs-pass@k plot (default: 1 5 10 20).")
    return parser.parse_args()

# ── Data loading ──────────────────────────────────────────────────────────────

def load_metrics(metrics_dir: str) -> pd.DataFrame:
    frames = []
    for fname in sorted(os.listdir(metrics_dir)):
        if not fname.startswith("metrics_") or not fname.endswith(".csv"):
            continue
        frames.append(pd.read_csv(os.path.join(metrics_dir, fname)))
    if not frames:
        raise FileNotFoundError(f"No metrics_*.csv files found in {metrics_dir}")
    return pd.concat(frames, ignore_index=True)

def filter_models(df: pd.DataFrame, args) -> pd.DataFrame:
    all_models = sorted(df["model"].unique())

    # Resolve --filter, including dynamic "topN" shorthand
    top_n = args.top
    m = re.fullmatch(r'top(\d+)', args.filter)
    if m:
        top_n = int(m.group(1))
        selected = all_models
    elif args.models:
        selected = [m for m in args.models if m in all_models]
    elif args.filter in PRESETS:
        selected = PRESETS[args.filter](all_models)
    else:
        raise ValueError(f"Unknown --filter value '{args.filter}'. "
                         f"Valid: {list(PRESETS.keys())} or topN (e.g. top5).")

    if args.exclude:
        selected = [m for m in selected if m not in args.exclude]
    df = df[df["model"].isin(selected)].copy()
    if top_n:
        top_models = (df.groupby("model")["pass@1"].mean()
                        .nlargest(top_n).index.tolist())
        df = df[df["model"].isin(top_models)].copy()
    return df

# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_or_show(fig, output_dir, fname):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fname)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)

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

def _short_name(name: str) -> str:
    name = re.sub(r'-Instruct$', '', name)       # Llama-3.3-70B-Instruct → Llama-3.3-70B
    name = re.sub(r'-v\d+\.\d+$', '', name)      # Codestral-22B-v0.1 → Codestral-22B
    return name

def _model_order(df: pd.DataFrame, ascending=False) -> list:
    """Models sorted by mean pass@1. ascending=True puts worst on left."""
    return (df.groupby("model")["pass@1"].mean()
              .sort_values(ascending=ascending).index
              .map(_short_name).tolist())

def _problem_type_order(df: pd.DataFrame) -> list:
    """Problem types sorted by mean pass@1 ascending (hardest on left within each group)."""
    return (df.groupby("problem type")["pass@1"].mean()
              .sort_values(ascending=True).index.tolist())

# ── Plot 1: k vs pass@k ───────────────────────────────────────────────────────

def plot_k_vs_passk(df: pd.DataFrame, k_values: list, output_dir, suffix=""):
    cols = [f"pass@{k}" for k in k_values if f"pass@{k}" in df.columns]
    agg = df.groupby("model")[cols].mean().reset_index()
    agg["_short"] = agg["model"].map(_short_name)
    agg = agg.sort_values("pass@1", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10" if len(agg) <= 10 else "tab20", n_colors=len(agg))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", ">", "<", "p"]

    for i, (_, row) in enumerate(agg.iterrows()):
        ys = [row[c] for c in cols]
        ax.plot(k_values[:len(ys)], ys,
                marker=markers[i % len(markers)],
                label=row["_short"],
                color=palette[i],
                linewidth=1.8, markersize=6)

    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("pass@k vs k  (mean across problem types)")
    ax.set_xticks(k_values)
    ax.set_ylim(bottom=0)
    _style(ax)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    _save_both(fig, output_dir, f"k_vs_passk{suffix}")

# ── Plot 2: pass@1 per problem type, grouped by model ────────────────────────

def plot_passk1_by_problem_type(df: pd.DataFrame, output_dir, suffix=""):
    agg = df[["model", "problem type", "pass@1"]].copy()
    agg["model"] = agg["model"].map(_short_name)

    # Global orderings shared across all models
    models        = _model_order(df, ascending=True)   # worst → best, left → right
    problem_types = _problem_type_order(df)            # hardest → easiest within each group
    n_models = len(models)
    n_types  = len(problem_types)

    palette = dict(zip(problem_types,
                       sns.color_palette("tab20", n_colors=n_types)))
    hatch_map = dict(zip(problem_types, HATCHES))

    bar_width = 0.75 / n_types
    x = list(range(n_models))

    fig, ax = plt.subplots(figsize=(max(10, 1.6 * n_models), 5))

    for i, ptype in enumerate(problem_types):
        sub = (agg[agg["problem type"] == ptype]
               .set_index("model")["pass@1"]
               .reindex(models)
               .fillna(0))
        offsets = [xi + (i - n_types / 2 + 0.5) * bar_width for xi in x]
        ax.bar(offsets, sub.values, width=bar_width * 0.92,
               label=ptype,
               color=palette[ptype],
               hatch=hatch_map[ptype],
               edgecolor="black", linewidth=0.5,
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("pass@1")
    ax.set_ylim(0, 1.05)
    ax.set_title("pass@1 by problem type  (models sorted by avg pass@1; "
                 "bars sorted hardest → easiest)")
    ax.legend(title="problem type", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    _style(ax)
    fig.tight_layout()
    _save_both(fig, output_dir, f"pass1_by_problem_type{suffix}")

# ── Plot 3: speedup@1 and efficiency@1 per model ─────────────────────────────

def plot_speedup_efficiency(df: pd.DataFrame, output_dir, suffix=""):
    agg = df.groupby("model")[["speedup@1", "efficiency@1"]].mean().reset_index()
    agg["model"] = agg["model"].map(_short_name)
    agg = agg.sort_values("speedup@1", ascending=True)
    n = len(agg)
    palette = sns.color_palette("Blues_d", n_colors=n)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.45 * n)), sharey=True)

    for ax, col, title in zip(
        axes,
        ["speedup@1", "efficiency@1"],
        ["speedup@1", "efficiency@1"],
    ):
        for j, (_, row) in enumerate(agg.iterrows()):
            ax.barh(row["model"], row[col],
                    color=palette[j],
                    hatch=HATCHES[j % len(HATCHES)],
                    edgecolor="black", linewidth=0.5,
                    height=0.65, zorder=3)
        ax.set_title(f"{title}  (mean across problem types)")
        ax.axvline(1.0, color="#555555", linewidth=0.9, linestyle="--", alpha=0.6, zorder=4)
        _style(ax, xgrid=True, ygrid=False, pct_axis=None)
        ax.tick_params(axis="y", labelsize=9)

    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle("Speedup and efficiency @ k=1  (mean across problem types)", fontsize=12)
    fig.tight_layout()
    _save_both(fig, output_dir, f"speedup_efficiency_1{suffix}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    args.metrics_dir = resolve_metrics_dir(args.metrics_dir)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.metrics_dir)

    df = load_metrics(args.metrics_dir)
    df = filter_models(df, args)

    if df.empty:
        print("No models left after filtering.")
        return

    models_used = sorted(df["model"].unique())
    print(f"Plotting {len(models_used)} models: {models_used}")

    suffix = fname_suffix(args)
    plot_k_vs_passk(df, args.k, args.output_dir, suffix)
    plot_passk1_by_problem_type(df, args.output_dir, suffix)
    plot_speedup_efficiency(df, args.output_dir, suffix)


if __name__ == "__main__":
    main()
