"""Plot per-model failure-mode composition from a classified errors CSV.

Reads outputs/<dir>/errors_<dir>.csv (produced by classify-errors.py, ideally
with --include-success so shares are over all run attempts) and draws a
horizontal stacked bar per model. By default it shows the coarse category_group
split (success / recoverable / validation / runtime / build / other); with
--fine it shows the top individual error categories, folding the long tail into a
grey "remaining" band. Note that "remaining" (fine) is a fold of real but less
common categories, whereas "other" (coarse) is the genuine unclassified bucket.

Usage:
    python plot-failure-modes.py kernel
    python plot-failure-modes.py outputs/kernel --fine --top 6
"""
import argparse
import importlib
import os

import matplotlib.pyplot as plt
import pandas as pd

import plotstyle
# classify-errors.py owns the category_group taxonomy and its share computation.
classify = importlib.import_module("classify-errors")

# Bump every text element on these plots this many points above the plotstyle default.
FONT_BUMP = 2

# Fixed, colour-blind-safe palette for the coarse groups. Order = stacking order.
GROUP_ORDER = ["success", "recoverable", "validation", "runtime", "build", "other"]
GROUP_COLOR = {
    "success":     "#1BB863",   # green: passed validation natively
    "recoverable": "#2273D6",   # blue: failed natively but a relaxation rescued it
    "validation":  "#E67E22",   # orange: ran but result was wrong
    "runtime": "#C0392B",   # red: crash / syntax fault / unfixed error
    "build":       "#7D2FB5",   # purple: rejected (no @task) before running
    "other":       "#9e9e9e",   # grey
}

# Palette for the --fine view error categories. Excludes the hues reserved for
# success (green), validation (orange), and remaining (grey), so those keep a
# single fixed colour. Twelve colours cover the full merged taxonomy.
_FINE_COLORS = ["#C0392B", "#E6A817", "#2273D6", "#12A190", "#7D2FB5",
                "#A81070", "#5D6D7E", "#D81B60", "#00838F", "#8D6E63",
                "#3F51B5", "#558B2F"]


def _errors_csv(metrics_dir: str) -> str:
    name = os.path.basename(os.path.normpath(metrics_dir))
    path = os.path.join(metrics_dir, f"errors_{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run classify-errors.py {name} --include-success first.")
    return path


def _group_shares(df: pd.DataFrame) -> pd.DataFrame:
    """model × category_group share matrix (rows sum to 1), in fixed GROUP_ORDER.

    Missing groups (absent from the run) become zero-share columns; any present
    group outside GROUP_ORDER stays in the shared denominator but is not drawn.
    """
    shares = classify.group_share_matrix(df, "model")
    for g in GROUP_ORDER:
        if g not in shares.columns:
            shares[g] = 0.0
    return shares[GROUP_ORDER]


def _fine_shares(df: pd.DataFrame, top: int) -> tuple[pd.DataFrame, list[str]]:
    """model × top-N error-category share matrix, tail folded into 'remaining'.

    'remaining' collects the less common real categories, so it is not the same
    as the coarse 'other' group (the genuine unclassified bucket).
    """
    counts = (df.groupby(["model", "error_category"]).size()
                .unstack(fill_value=0))
    # success and recoverable are passing outcomes: pin them as leading columns.
    leading = [c for c in ("success", "recoverable") if c in counts.columns]
    order = counts.sum(axis=0).sort_values(ascending=False)
    keep = [c for c in order.index if c not in leading][:top]
    cols = leading + keep
    kept = counts[cols].copy()
    kept["remaining"] = counts.drop(columns=cols, errors="ignore").sum(axis=1)
    shares = kept.div(kept.sum(axis=1), axis=0)
    return shares, list(shares.columns)


def plot(metrics_dir: str, output_dir: str, fine: bool, top: int) -> None:
    df = pd.read_csv(_errors_csv(metrics_dir))
    if "category_group" not in df.columns:
        raise ValueError("errors CSV lacks 'category_group'; regenerate with the "
                         "current classify-errors.py.")

    if fine:
        shares, columns = _fine_shares(df, top)
        # Passing outcomes and 'remaining' keep fixed colours; the rotating palette
        # is only for the error categories.
        fixed = {"success": GROUP_COLOR["success"],
                 "recoverable": GROUP_COLOR["recoverable"],
                 "validation": GROUP_COLOR["validation"],
                 "remaining": GROUP_COLOR["other"]}
        colors, palette_i = {}, 0
        for c in columns:
            if c in fixed:
                colors[c] = fixed[c]
            else:
                colors[c] = _FINE_COLORS[palette_i % len(_FINE_COLORS)]
                palette_i += 1
    else:
        shares = _group_shares(df)
        columns = GROUP_ORDER
        colors = GROUP_COLOR

    # Worst models (least success) at the top of the chart.
    success_col = "success" if "success" in shares.columns else columns[0]
    shares = shares.sort_values(success_col, ascending=True)

    models = list(shares.index)
    labels = [plotstyle.label_for(m) for m in models]
    y = range(len(models))

    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(models) + 1.4))
    left = [0.0] * len(models)
    for col in columns:
        vals = shares[col].values
        ax.barh(list(y), vals, left=left, color=colors[col], label=col,
                edgecolor="white", linewidth=0.5)
        left = [l + v for l, v in zip(left, vals)]

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of run attempts", fontsize=10 + FONT_BUMP)
    ax.invert_yaxis()
    plotstyle.style_axis(ax, xgrid=True, ygrid=False, pct_axis="x", labelsize=9 + FONT_BUMP)
    # Wrap the legend to a few rows so the fine view (a dozen categories) stays readable.
    ax.legend(ncol=min(len(columns), 7), loc="lower center",
              bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=8 + FONT_BUMP)

    base = "failure_modes" + ("_fine" if fine else "")
    plotstyle.save(fig, output_dir, base)


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir",
        help="Run name (e.g. 'kernel') or path to analysis/outputs/<dir>/.")
    parser.add_argument("-o", "--output", default=None,
        help="Output directory for the figure. Defaults to plots/<dir>/.")
    parser.add_argument("--fine", action="store_true",
        help="Show top individual error categories instead of the coarse groups.")
    parser.add_argument("--top", type=int, default=12,
        help="Number of fine categories to show before folding into 'remaining' (default: 12).")
    return parser.parse_args()


def main():
    args = get_args()
    metrics_dir = plotstyle.resolve_metrics_dir(args.dir)
    output_dir = args.output or plotstyle.default_output_dir(metrics_dir)
    plot(metrics_dir, output_dir, args.fine, args.top)


if __name__ == "__main__":
    main()
