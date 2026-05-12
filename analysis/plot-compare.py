""" Compare pass@k for the same models across two prompt sets. """
import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
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

def _style(ax, *, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
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

# ── Data loading & filtering ──────────────────────────────────────────────────

def load_metrics(metrics_dir: str) -> pd.DataFrame:
    frames = []
    for fname in sorted(os.listdir(metrics_dir)):
        if not fname.startswith("metrics_") or not fname.endswith(".csv"):
            continue
        frames.append(pd.read_csv(os.path.join(metrics_dir, fname)))
    if not frames:
        raise FileNotFoundError(f"No metrics_*.csv files found in {metrics_dir}")
    return pd.concat(frames, ignore_index=True)

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

    n       = len(model_order)
    palette = sns.color_palette("tab10" if n <= 10 else "tab20", n_colors=n)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", ">", "<", "p"]
    xs      = k_values[:len(cols)]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(model_order):
        color  = palette[i]
        marker = markers[i % len(markers)]
        short  = _short_name(model)
        ax.plot(xs, agg_a.loc[model, cols].tolist(),
                marker=marker, color=color, linestyle="-",
                linewidth=1.8, markersize=6, label=short)
        ax.plot(xs, agg_b.loc[model, cols].tolist(),
                marker=marker, color=color, linestyle="--",
                linewidth=1.8, markersize=6, label="_nolegend_")

    # Model color entries come from the solid lines; add linestyle entries below.
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
    _style(ax)
    fig.tight_layout()
    _save_both(fig, output_dir, f"compare_passk_{label_a}_vs_{label_b}{suffix}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    args.dir_a = resolve_metrics_dir(args.dir_a)
    args.dir_b = resolve_metrics_dir(args.dir_b)
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
