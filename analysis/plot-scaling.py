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

import plotstyle as ps

PRESETS = ps.PRESETS

# ── Data loading ──────────────────────────────────────────────────────────────

def load_curves(metrics_dir: str) -> pd.DataFrame:
    return ps.load_csvs(metrics_dir, "curve_")

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
    for model in pivot.columns:
        ax.plot(ns, pivot[model].reindex(ns).values,
                marker=ps.marker_for(model),
                color=ps.color_for(model),
                label=ps.label_for(model),
                **ps.LINE_KW)
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
    ax.plot(ns, ns, color=ps.REF_COLOR, linestyle="--", linewidth=1.0,
            alpha=0.7, label="ideal (linear)")
    ax.set_ylabel("speedup")
    # cap to observed speedups so the linear-ideal reference doesn't squash the data
    ax.set_ylim(0, max(1.0, float(pivot.max().max())) * 1.3)
    ax.set_title("Speedup vs resources  (mean across problem types)")
    ps.style_axis(ax, pct_axis=None)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    ps.save_both(fig, output_dir, f"speedup_vs_n{suffix}")

def plot_efficiency_vs_n(long_df, ns, output_dir, suffix=""):
    agg = long_df.groupby(["model", "n"])["efficiency"].mean().reset_index()
    pivot = agg.pivot(index="n", columns="model", values="efficiency")
    order = pivot.loc[min(ns)].sort_values(ascending=False).index.tolist()
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    _model_lines(ax, pivot, ns)
    ax.axhline(1.0, color=ps.REF_COLOR, linestyle="--", linewidth=1.0,
               alpha=0.7, label="ideal (100%)")
    ax.set_ylabel("efficiency")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Efficiency vs resources  (mean across problem types)")
    ps.style_axis(ax, pct_axis=None)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    ps.save_both(fig, output_dir, f"efficiency_vs_n{suffix}")

# ── Argument parsing ──────────────────────────────────────────────────────────

def fname_suffix(args) -> str:
    return ps.fname_suffix(args)

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

    args.metrics_dir = ps.resolve_metrics_dir(args.metrics_dir)
    if args.output_dir is None:
        args.output_dir = ps.default_output_dir(args.metrics_dir)

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
