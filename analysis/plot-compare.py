""" Compare pass@k for the same models across two prompt sets. """
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

import plotstyle as ps

PRESETS = ps.PRESETS

# ── Data loading & filtering ──────────────────────────────────────────────────

def load_metrics(metrics_dir: str) -> pd.DataFrame:
    return ps.load_csvs(metrics_dir, "metrics_")

def _apply_preset_filter(df: pd.DataFrame, args) -> pd.DataFrame:
    """Apply --filter / --models / --exclude, but NOT --top."""
    all_models = sorted(df["model"].unique())
    if args.models:
        selected = [m for m in args.models if m in all_models]
    elif args.filter in PRESETS:
        selected = PRESETS[args.filter](all_models)
    else:
        raise ValueError(f"Unknown --filter value '{args.filter}'. "
                         f"Valid: {list(PRESETS.keys())}.")
    if args.exclude:
        selected = [m for m in selected if m not in args.exclude]
    return df[df["model"].isin(selected)].copy()

# ── Argument parsing ──────────────────────────────────────────────────────────

def fname_suffix(args) -> str:
    return ps.fname_suffix(args)

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir_a", type=str,
        help="First metrics directory (or run name under outputs/).")
    parser.add_argument("dir_b", type=str,
        help="Second metrics directory (or run name under outputs/).")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
        help="Directory to save plots. Defaults to plots/<dir_a>_vs_<dir_b>.")
    parser.add_argument("--filter", type=str, default="all",
        help=f"Named model subset: {list(PRESETS.keys())}. Default: all.")
    parser.add_argument("--models", nargs="+", type=str, default=None,
        help="Explicit list of model names to include (overrides --filter).")
    parser.add_argument("--exclude", nargs="+", type=str, default=None,
        help="Model names to exclude (applied after --filter / --models).")
    parser.add_argument("--top", type=int, default=None,
        help="Keep only the top-N models by average pass@1 across both sets.")
    parser.add_argument("-k", nargs="+", type=int, default=[1, 5, 10, 20],
        help="K values for the pass@k plot (default: 1 5 10 20).")
    return parser.parse_args()

# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_compare_passk(df_a, label_a, df_b, label_b, k_values, output_dir, suffix=""):
    cols = [f"pass@{k}" for k in k_values
            if f"pass@{k}" in df_a.columns and f"pass@{k}" in df_b.columns]

    agg_a = df_a.groupby("model")[cols].mean()
    agg_b = df_b.groupby("model")[cols].mean()
    model_order = agg_a["pass@1"].sort_values(ascending=False).index.tolist()
    xs = k_values[:len(cols)]

    fig, ax = plt.subplots(figsize=(9, 5))

    for model in model_order:
        color  = ps.color_for(model)
        marker = ps.marker_for(model)
        ax.plot(xs, agg_a.loc[model, cols].tolist(),
                marker=marker, color=color, linestyle="-",
                label=ps.label_for(model), **ps.LINE_KW)
        ax.plot(xs, agg_b.loc[model, cols].tolist(),
                marker=marker, color=color, linestyle="--",
                label="_nolegend_", **ps.LINE_KW)

    # Model colour entries come from the solid lines; add linestyle entries below.
    model_handles, model_labels = ax.get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=1.8, label=label_a),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label=label_b),
    ]
    ax.legend(handles=model_handles + style_handles,
              labels=model_labels + [label_a, label_b],
              bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title(f"pass@k  —  {label_a}  vs  {label_b}  (mean across problem types)")
    ax.set_xticks(xs)
    ax.set_ylim(bottom=0)
    ps.style_axis(ax)
    fig.tight_layout()
    ps.save_both(fig, output_dir, f"compare_passk_{label_a}_vs_{label_b}{suffix}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    args.dir_a = ps.resolve_metrics_dir(args.dir_a)
    args.dir_b = ps.resolve_metrics_dir(args.dir_b)
    label_a = os.path.basename(os.path.normpath(args.dir_a))
    label_b = os.path.basename(os.path.normpath(args.dir_b))

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(__file__), "plots", f"{label_a}_vs_{label_b}")

    df_a = load_metrics(args.dir_a)
    df_b = load_metrics(args.dir_b)

    df_a = _apply_preset_filter(df_a, args)
    df_b = _apply_preset_filter(df_b, args)

    common = set(df_a["model"].unique()) & set(df_b["model"].unique())
    if not common:
        print("No models in common between the two sets after filtering.")
        return
    df_a = df_a[df_a["model"].isin(common)].copy()
    df_b = df_b[df_b["model"].isin(common)].copy()

    if args.top:
        top_models = (
            pd.concat([df_a[["model", "pass@1"]], df_b[["model", "pass@1"]]])
            .groupby("model")["pass@1"].mean()
            .nlargest(args.top)
            .index.tolist()
        )
        df_a = df_a[df_a["model"].isin(top_models)].copy()
        df_b = df_b[df_b["model"].isin(top_models)].copy()

    models_used = sorted(df_a["model"].unique())
    print(f"Comparing {len(models_used)} models: '{label_a}' vs '{label_b}'")
    for m in models_used:
        print(f"  {m}")

    suffix = fname_suffix(args)
    plot_compare_passk(df_a, label_a, df_b, label_b, args.k, args.output_dir, suffix)


if __name__ == "__main__":
    main()
