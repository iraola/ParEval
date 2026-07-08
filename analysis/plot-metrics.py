""" Plot metrics from the analysis/outputs/<dir>/metrics_*.csv files. """
import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

import plotstyle as ps

# ── Model filter presets come from plotstyle (shared across all plot scripts) ──
PRESETS = ps.PRESETS

# Hatches keep the per-problem-type bars distinguishable in grayscale print.
HATCHES = ['/', '\\', 'x', '-', '+', 'o', '//', '\\\\', '||', '--', '++', 'xx']

# ── Argument parsing ──────────────────────────────────────────────────────────

def fname_suffix(args) -> str:
    return ps.fname_suffix(args)

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
    return ps.load_csvs(metrics_dir, "metrics_")

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

def _model_order(df: pd.DataFrame, ascending=False) -> list:
    """Full model names sorted by mean pass@1. ascending=True puts worst first."""
    return (df.groupby("model")["pass@1"].mean()
              .sort_values(ascending=ascending).index.tolist())

def _problem_type_order(df: pd.DataFrame) -> list:
    """Problem types sorted by mean pass@1 ascending (hardest first)."""
    return (df.groupby("problem type")["pass@1"].mean()
              .sort_values(ascending=True).index.tolist())

# ── Plot 1: k vs pass@k ───────────────────────────────────────────────────────

def plot_k_vs_passk(df: pd.DataFrame, k_values: list, output_dir, suffix=""):
    cols = [f"pass@{k}" for k in k_values if f"pass@{k}" in df.columns]
    agg = df.groupby("model")[cols].mean().reset_index()
    agg = agg.sort_values("pass@1", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))

    for _, row in agg.iterrows():
        model = row["model"]
        ys = [row[c] for c in cols]
        ax.plot(k_values[:len(ys)], ys,
                marker=ps.marker_for(model),
                label=ps.label_for(model),
                color=ps.color_for(model),
                **ps.LINE_KW)

    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("pass@k vs k  (mean across problem types)")
    ax.set_xticks(k_values)
    ax.set_ylim((0, 1))
    ps.style_axis(ax)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    ps.save_both(fig, output_dir, f"k_vs_passk{suffix}")

# ── Plot 2: pass@1 per problem type, grouped by model ────────────────────────

def plot_passk1_by_problem_type(df: pd.DataFrame, output_dir, suffix=""):
    agg = df[["model", "problem type", "pass@1"]].copy()

    # Global orderings shared across all models (full names → styled per model)
    models        = _model_order(df, ascending=True)   # worst → best, left → right
    problem_types = _problem_type_order(df)            # hardest → easiest per group
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
    ax.set_xticklabels([ps.label_for(m) for m in models],
                       rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("pass@1")
    ax.set_ylim(0, 1.05)
    ax.set_title("pass@1 by problem type  (models sorted by avg pass@1; "
                 "bars sorted hardest → easiest)")
    ax.legend(title="problem type", bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=False)
    ps.style_axis(ax)
    fig.tight_layout()
    ps.save_both(fig, output_dir, f"pass1_by_problem_type{suffix}")

# ── Plot 1b: k vs pass@k, one panel per model family ─────────────────────────

def plot_k_vs_passk_by_family(df: pd.DataFrame, k_values: list, output_dir, suffix=""):
    """Small-multiples version of plot 1: few lines per panel stay readable
    even when all models are plotted at once."""
    cols = [f"pass@{k}" for k in k_values if f"pass@{k}" in df.columns]
    agg = df.groupby("model")[cols].mean()

    families: dict[str, list] = {}
    for model in agg.index:
        families.setdefault(ps.family_of(model), []).append(model)
    order = [f for f in ps.FAMILIES if f in families]
    order += sorted(set(families) - set(order))

    ncols = 4
    nrows = -(-len(order) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.7 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for ax, fam in zip(axes.flat, order):
        models = sorted(families[fam], key=lambda m: agg.loc[m, "pass@1"],
                        reverse=True)
        for model in models:
            ys = [agg.loc[model, c] for c in cols]
            ax.plot(k_values[:len(ys)], ys,
                    marker=ps.marker_for(model),
                    label=ps.label_for(model),
                    color=ps.color_for(model),
                    **ps.LINE_KW)
        ax.set_title(fam, fontsize=10)
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1.02)
        ps.style_axis(ax)
        ax.legend(fontsize=7, frameon=False, loc="best")
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("k")
    for ax in axes[:, 0]:
        ax.set_ylabel("pass@k")

    fig.tight_layout()
    ps.save_both(fig, output_dir, f"k_vs_passk_by_family{suffix}")

# ── Plot 2b: pass@1 heatmap (model × problem type) ───────────────────────────

def plot_passk1_heatmap(df: pd.DataFrame, output_dir, suffix=""):
    """Same data as plot 2, as a compact heatmap."""
    models        = _model_order(df, ascending=False)  # best → worst, top → bottom
    problem_types = _problem_type_order(df)             # hardest → easiest, left → right

    grid = (df.pivot_table(index="model", columns="problem type",
                           values="pass@1", aggfunc="mean")
              .reindex(index=models, columns=problem_types))

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(problem_types)),
                                    max(4, 0.5 * len(models))))
    sns.heatmap(grid, ax=ax, cmap="viridis", vmin=0, vmax=1,
                annot=True, fmt=".0%", annot_kws={"fontsize": 7},
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "pass@1", "format": mticker.PercentFormatter(xmax=1)})
    ax.set_yticklabels([ps.label_for(m) for m in grid.index], rotation=0, fontsize=8)
    ax.set_xticklabels(grid.columns, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("pass@1 by model × problem type")
    fig.tight_layout()
    ps.save_both(fig, output_dir, f"pass1_heatmap{suffix}")

# ── Plot 3: speedup@1 and efficiency@1 per model ─────────────────────────────

def plot_speedup_efficiency(df: pd.DataFrame, output_dir, suffix=""):
    agg = df.groupby("model")[["speedup@1", "efficiency@1"]].mean().reset_index()
    agg = agg.sort_values("speedup@1", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.45 * len(agg))), sharey=True)

    for ax, col in zip(axes, ["speedup@1", "efficiency@1"]):
        for _, row in agg.iterrows():
            model = row["model"]
            ax.barh(ps.label_for(model), row[col],
                    color=ps.color_for(model),
                    edgecolor="black", linewidth=0.5,
                    height=0.65, zorder=3)
        ax.set_title(f"{col}  (mean across problem types)")
        ax.axvline(1.0, color=ps.REF_COLOR, linewidth=0.9, linestyle="--",
                   alpha=0.6, zorder=4)
        ps.style_axis(ax, xgrid=True, ygrid=False, pct_axis=None)
        ax.tick_params(axis="y", labelsize=9)

    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle("Speedup and efficiency @ k=1  (mean across problem types)", fontsize=12)
    fig.tight_layout()
    ps.save_both(fig, output_dir, f"speedup_efficiency_1{suffix}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    args.metrics_dir = ps.resolve_metrics_dir(args.metrics_dir)
    if args.output_dir is None:
        args.output_dir = ps.default_output_dir(args.metrics_dir)

    df = load_metrics(args.metrics_dir)
    df = filter_models(df, args)

    if df.empty:
        print("No models left after filtering.")
        return

    models_used = sorted(df["model"].unique())
    print(f"Plotting {len(models_used)} models: {models_used}")

    suffix = fname_suffix(args)
    plot_k_vs_passk(df, args.k, args.output_dir, suffix)
    plot_k_vs_passk_by_family(df, args.k, args.output_dir, suffix)
    plot_passk1_by_problem_type(df, args.output_dir, suffix)
    plot_passk1_heatmap(df, args.output_dir, suffix)
    plot_speedup_efficiency(df, args.output_dir, suffix)


if __name__ == "__main__":
    main()
