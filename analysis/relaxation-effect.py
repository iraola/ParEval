"""Quantify how much source-code relaxations improve correctness.

For each model dataframe in analysis/outputs/<dir>/, compute pass@k twice:
  * baseline:  outputs that only passed after a relaxation are counted as failures
  * relaxed:   those relaxed passes are counted as correct
and report the improvement, per (model, problem_type) and per model overall
(problem_type == "ALL").

The two pass@k values reproduce exactly what metrics.py writes to
outputs/<dir>/ (baseline) and outputs/<dir>-relaxations/ (relaxed); this script
just puts them side by side and takes the difference. Correctness aggregation is
reused from metrics.py so the numbers stay consistent with the rest of the pipeline.

Usage:
    python relaxation-effect.py kernel
    python relaxation-effect.py kernel -k 1 5 10 20
    python relaxation-effect.py kernel --digest -o outputs/kernel-relaxations/relaxation_effect.csv
"""
import argparse
import importlib
import os

import pandas as pd

# metrics.py runs argparse only under __main__, so importing it here is side-effect-free.
metrics = importlib.import_module("metrics")


def _perproblem_passk(correctness: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per-problem pass@k from a per-sample correctness df (get_correctness_df output)."""
    agg = (correctness
           .groupby(["name", "parallelism_model", "problem_type"])
           .agg(total=("is_valid", "count"), valid=("is_valid", "sum"))
           .reset_index())
    agg[f"pass@{k}"] = agg.apply(
        lambda x: metrics._passk(x["total"], x["valid"], k), axis=1)
    return agg


def _passk_by_type_and_overall(df: pd.DataFrame, k: int) -> tuple[pd.Series, float]:
    """Return (pass@k per problem_type, pass@k overall) for one prepared dataframe.

    Overall weights every problem equally (mean of per-problem pass@k); per-type
    is the mean over the problems in that type, matching metrics.py's passk().
    """
    pp = _perproblem_passk(metrics.get_correctness_df(df), k)
    by_type = pp.groupby("problem_type")[f"pass@{k}"].mean()
    overall = pp[f"pass@{k}"].mean()
    return by_type, overall


def _prepare(df: pd.DataFrame, count_relaxed: bool) -> pd.DataFrame:
    """Fill missing run flags and, for the baseline, void relaxed passes.

    Mirrors metrics.py: did_run/is_valid are NaN when a build failed, and without
    --relaxations a relaxed pass is treated as a failure.
    """
    df = df.copy()
    df["did_run"] = df["did_run"].fillna(False)
    df["is_valid"] = df["is_valid"].fillna(False)
    if not count_relaxed and "relaxation_used" in df.columns:
        df.loc[df["relaxation_used"] == True, "is_valid"] = False  # noqa: E712
    return df


def effect_for_model(df: pd.DataFrame, model: str, ks: list[int]) -> list[dict]:
    """Rows of {model, problem_type, k, pass_no_relax, pass_relax, delta_pp, delta_rel_pct}."""
    baseline = _prepare(df, count_relaxed=False)
    relaxed = _prepare(df, count_relaxed=True)

    rows = []
    for k in ks:
        base_type, base_all = _passk_by_type_and_overall(baseline, k)
        relax_type, relax_all = _passk_by_type_and_overall(relaxed, k)

        entries = [(pt, base_type.get(pt, float("nan")), relax_type[pt])
                   for pt in relax_type.index]
        entries.append(("ALL", base_all, relax_all))

        for ptype, no_relax, with_relax in entries:
            delta = with_relax - no_relax
            rel = (100.0 * delta / no_relax) if no_relax and no_relax > 0 else float("nan")
            rows.append({
                "model": model,
                "problem_type": ptype,
                "k": k,
                "pass_no_relax": round(no_relax, 6),
                "pass_relax": round(with_relax, 6),
                "delta_pp": round(delta, 6),
                "delta_rel_pct": round(rel, 2) if rel == rel else float("nan"),
            })
    return rows


def _digest_md(df: pd.DataFrame, run_name: str, k: int) -> str:
    """Markdown table ranking models by overall pass@k gain from relaxations."""
    overall = (df[(df["problem_type"] == "ALL") & (df["k"] == k)]
               .sort_values("delta_pp", ascending=False))
    lines = [f"# Relaxation effect on correctness: {run_name} (pass@{k})", "",
             "| model | pass@{0} baseline | pass@{0} relaxed | gain (pp) |".format(k),
             "|---|---:|---:|---:|"]
    for _, r in overall.iterrows():
        lines.append(
            f"| {r['model']} | {100 * r['pass_no_relax']:.1f}% | "
            f"{100 * r['pass_relax']:.1f}% | +{100 * r['delta_pp']:.1f} |")
    lines.append("")
    return "\n".join(lines)


def resolve_dataframe_dir(value: str) -> str:
    """Accept a real directory or a run name under analysis/outputs/."""
    if os.path.isdir(value):
        return value
    candidate = os.path.join(os.path.dirname(__file__), "outputs", value)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"'{value}' is not a valid directory and was not found under outputs/")


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir",
        help="Run name (e.g. 'kernel') or path to analysis/outputs/<dir>/ with dataframe_*.csv.")
    parser.add_argument("-k", "--k", type=int, nargs="+", default=[1, 5, 10, 20],
        help="k values for pass@k (default: 1 5 10 20).")
    parser.add_argument("-o", "--output", default=None,
        help="Output CSV. Defaults to outputs/<dir>-relaxations/relaxation_effect.csv.")
    parser.add_argument("--digest", action="store_true",
        help="Also write a Markdown digest ranking models by pass@1 gain.")
    return parser.parse_args()


def main():
    args = get_args()
    df_dir = resolve_dataframe_dir(args.dir)
    run_name = os.path.basename(os.path.normpath(df_dir))

    files = sorted(f for f in os.listdir(df_dir)
                   if f.startswith("dataframe_") and f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"No dataframe_*.csv files in {df_dir}")

    all_rows = []
    for fname in files:
        model = fname[len("dataframe_"):-len(".csv")]
        df = pd.read_csv(os.path.join(df_dir, fname))
        if "relaxation_used" not in df.columns:
            print(f"  {model}: no relaxation_used column, skipping")
            continue
        all_rows.extend(effect_for_model(df, model, args.k))
        print(f"  {model}: done")

    result = pd.DataFrame(all_rows)

    out_path = args.output or os.path.join(
        os.path.dirname(__file__), "outputs", f"{run_name}-relaxations",
        "relaxation_effect.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} rows → {out_path}")

    # Quick overview: overall pass@1 gain per model, best first.
    overview = (result[(result["problem_type"] == "ALL") & (result["k"] == 1)]
                .sort_values("delta_pp", ascending=False)
                [["model", "pass_no_relax", "pass_relax", "delta_pp"]])
    print("\n── Overall pass@1 gain from relaxations (best first) ──")
    print(overview.to_string(index=False))

    if args.digest:
        digest_path = os.path.join(os.path.dirname(out_path),
                                   f"relaxation_effect_{run_name}_digest.md")
        with open(digest_path, "w") as f:
            f.write(_digest_md(result, run_name, k=1))
        print(f"\n[Markdown digest] → {digest_path}")


if __name__ == "__main__":
    main()
