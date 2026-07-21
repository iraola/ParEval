""" Compute the metrics over the data.
"""
# std imports
import argparse
import json
from dataclasses import dataclass
from math import comb
from typing import Union

# tpl imports
import numpy as np
import pandas as pd


@dataclass
class ModelSpec:
    """Per-parallelism-model facts the metrics derive from.

    workers: columns whose product is the run's parallel worker count (the scaling
             "n" / the efficiency divisor). Empty means serial, i.e. 1.
    operating_point: the single config the headline speedup/efficiency metrics are
                     reported at, as {column: value}. Empty means no constraint (gpu).
    """
    workers: list
    operating_point: dict


# Single source of truth for per-model resource semantics; add a backend = one entry.
#                         workers                       operating point
MODELS = {
    "serial":   ModelSpec([],                           {}),
    "omp":      ModelSpec(["num_threads"],              {"num_threads": 32}),
    "mpi":      ModelSpec(["num_procs"],                {"num_procs": 512}),
    "mpi+omp":  ModelSpec(["num_procs", "num_threads"], {"num_procs": 4, "num_threads": 64}),
    "kokkos":   ModelSpec(["num_threads"],              {"num_threads": 32}),
    "cuda":     ModelSpec(["problem_size"],             {}),
    "hip":      ModelSpec(["problem_size"],             {}),
    "pycompss": ModelSpec(["num_procs"],                {"num_procs": 64}),
}


def canonical_mask(df: pd.DataFrame) -> pd.Series:
    """Mask selecting each model's operating point; an allowlist.

    Rows whose parallelism_model is not in MODELS are excluded.
    """
    mask = pd.Series(False, index=df.index)
    for model, spec in MODELS.items():
        m = df["parallelism_model"] == model
        for col, val in spec.operating_point.items():
            m &= df[col] == val
        mask |= m
    return mask


def resource_count(df: pd.DataFrame) -> pd.Series:
    """Per-row worker count (the scaling 'n'): product of each model's worker columns.

    Rows of models not in MODELS (and serial, whose worker list is empty) default to 1.
    """
    out = pd.Series(1.0, index=df.index)
    for model, spec in MODELS.items():
        m = df["parallelism_model"] == model
        if not m.any():
            continue
        n = pd.Series(1.0, index=df.index[m])
        for col in spec.workers:
            n *= df.loc[m, col]
        out.loc[m] = n
    return out


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=str, help="Input CSV file containing the test cases.")
    parser.add_argument("-k", "--k", type=int, nargs='+', default=[1,5,10,20], help="K value for pass@k, build@k, and speedup@k.")
    parser.add_argument("-n", "--n", type=int, default=1, help="N value for speedup@k.")
    parser.add_argument("-o", "--output", type=str, help="Output csv file containing the results.")
    parser.add_argument("--problem-sizes", type=str, default='../drivers/problem-sizes.json', help="Json with problem sizes. Used for calculating GPU efficiency.")
    parser.add_argument("--model-name", type=str, help="Add model name column with this value")
    parser.add_argument("--relaxations", action="store_true",
        help="Count outputs that passed only after a relaxation as correct. "
             "Default: treat relaxed passes as failures.")
    parser.add_argument("--baseline", choices=["serial", "n1"], default="serial",
        help="Speedup baseline: 'serial' (purely sequential) or 'n1' (the single-resource "
             "run, factoring out fixed runtime overhead). Default: serial.")
    return parser.parse_args()

def get_correctness_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Group by name, parallelism_model, and output_idx, and set is_valid to true only if all rows in the group have is_valid = true.
        Set it to false otherwise.
    """
    # group all the runs for this LLM output
    df = df.copy()
    agg = df.groupby(["name", "parallelism_model", "output_idx"]).agg({"is_valid": ["count", "sum"]})
    agg.columns = ["count", "sum"]

    # mark as valid only if all runs are valid
    agg["is_valid"] = agg["count"] == agg["sum"]
    agg = agg.reset_index()
    agg = agg.drop(columns=["count", "sum"])
    
    # add problem_type column from df
    agg = agg.merge(df[["name", "problem_type"]].drop_duplicates(), on="name", how="left")

    return agg

def nCr(n: int, r: int) -> int:
    return comb(n, r)

def buildk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Compute the build@k metric """
    agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg({"did_build": ["count", "sum"]})
    agg.columns = ["total_build_attempts", "successful_builds"]
    agg = agg.reset_index()
    agg[f"build@{k}"] = agg.apply(lambda x: _passk(x["total_build_attempts"], x["successful_builds"], k), axis=1)
    return agg.groupby(["parallelism_model", "problem_type"]).agg({f"build@{k}": "mean"})

def _passk(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples < k:
        return float("nan")   # estimator undefined with fewer than k samples
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))

def passk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Compute the pass@k metric """
    agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg({"is_valid": ["count", "sum"]})
    agg.columns = ["total_runs", "valid_count"]
    agg = agg.reset_index()
    agg[f"pass@{k}"] = agg.apply(lambda x: _passk(x["total_runs"], x["valid_count"], k), axis=1)
    return agg.groupby(["parallelism_model", "problem_type"]).agg({f"pass@{k}": "mean"})

def _speedupk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int, col_name: str = 'speedup@{}') -> float:
    """ Compute the speedup@k metric """
    # create a copy of the runtimes
    if isinstance(runtimes, pd.Series):
        runtimes = runtimes.values.copy()
    else:
        runtimes = runtimes.copy()

    # sort the runtimes
    runtimes.sort()

    num_samples = runtimes.shape[0]
    if num_samples < k:
        return pd.Series({col_name.format(k): float("nan")})

    # compute expected value
    sum = 0.0
    for j in range(1, num_samples+1):
        num = nCr(j-1, k-1) * baseline_runtime
        den = nCr(num_samples, k) * max(runtimes[j-1], 1e-8)
        sum += num / den
    return pd.Series({col_name.format(k): sum})

def speedupk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """ Compute the speedup@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # keep each model's canonical operating point (see MODELS)
    df = df[canonical_mask(df)]
    df = df.copy()

    # use min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _speedupk
    df = df.groupby(["name", "parallelism_model", "problem_type"])[["runtime", "best_sequential_runtime"]].apply(
            lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k)
        ).reset_index()

    if df.empty or f"speedup@{k}" not in df.columns:
        idx = pd.MultiIndex.from_tuples([], names=["parallelism_model", "problem_type"])
        return pd.DataFrame(columns=[f"speedup@{k}"], index=idx)

    # compute the mean speedup@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"speedup@{k}": "mean"})

    return df



def speedupk_max(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Compute the speedup_max@k. Same as speedup_n@k, but instead of a fixed n
        we use the n that gives the max speedup
    """
    df = df.copy()
    df.drop(columns=['prompt'], inplace=True)

    # get all the runs where the submission is valid
    df = df[df["is_valid"] == True]

    # choose the min across processor counts
    df["runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["runtime"].transform("min")

    # use the min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # select only run_idx 0
    df["run_idx"] = df["run_idx"].astype(int)
    df = df[df["run_idx"] == 0]

    # group by name, parallelism_model, and output_idx and call _speedupk
    df = df.groupby(["name", "parallelism_model", "problem_type"])[["runtime", "best_sequential_runtime"]].apply(
            lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k, col_name="speedup_max@{}")
        ).reset_index()

    if df.empty or f"speedup_max@{k}" not in df.columns:
        idx = pd.MultiIndex.from_tuples([], names=["parallelism_model", "problem_type"])
        return pd.DataFrame(columns=[f"speedup_max@{k}"], index=idx)

    # compute the mean speedup_max@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"speedup_max@{k}": "mean"})

    return df



def _efficiencyk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int, n_resources: Union[pd.Series, np.ndarray], col_name: str = 'efficiency@{}') -> float:
    """ Compute the efficiency@k metric """
    # create a copy of the runtimes
    if isinstance(runtimes, pd.Series):
        runtimes = runtimes.values.copy()
    else:
        runtimes = runtimes.copy()

    if isinstance(n_resources, pd.Series):
        n_resources = n_resources.values.copy()
    else:
        n_resources = n_resources.copy()

    # sort the runtimes
    runtimes.sort()

    num_samples = runtimes.shape[0]
    if num_samples < k:
        return pd.Series({col_name.format(k): float("nan")})

    # compute expected value
    sum = 0.0
    for j in range(1, num_samples+1):
        num = nCr(j-1, k-1) * baseline_runtime
        den = nCr(num_samples, k) * max(runtimes[j-1], 1e-8) * n_resources[j-1]
        sum += num / den
    return pd.Series({col_name.format(k): sum})

def efficiencyk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """ Compute the efficiency@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # keep each model's canonical operating point (see MODELS)
    df = df[canonical_mask(df)]

    # resource count at that operating point (see MODELS)
    df["n_resources"] = resource_count(df)

    df = df.copy()

    # use min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _efficiencyk
    df = df.groupby(["name", "parallelism_model", "problem_type"])[["runtime", "best_sequential_runtime", "n_resources"]].apply(
            lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"])
        ).reset_index()

    if df.empty or f"efficiency@{k}" not in df.columns:
        idx = pd.MultiIndex.from_tuples([], names=["parallelism_model", "problem_type"])
        return pd.DataFrame(columns=[f"efficiency@{k}"], index=idx)

    # compute the mean efficiency@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency@{k}": "mean"})

    return df


def efficiencyk_max(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Compute the efficiency_max@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # resource count per row (see MODELS)
    df["n_resources"] = resource_count(df)

    # choose the row with min num_resources * runtime
    df = df.groupby(["name", "parallelism_model", "output_idx"]).apply(
            lambda row: row.iloc[np.argmin(row["runtime"] * row["n_resources"])]
        ).reset_index(drop=True)

    # use the min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _efficiencyk
    df = df.groupby(["name", "parallelism_model", "problem_type"])[["runtime", "best_sequential_runtime", "n_resources"]].apply(
            lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"], col_name='efficiency_max@{}')
        ).reset_index()

    if df.empty or f"efficiency_max@{k}" not in df.columns:
        idx = pd.MultiIndex.from_tuples([], names=["parallelism_model", "problem_type"])
        return pd.DataFrame(columns=[f"efficiency_max@{k}"], index=idx)

    # compute the mean efficiency_max@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency_max@{k}": "mean"})

    return df

def apply_n1_baseline(df: pd.DataFrame, void_unmatched: bool = False) -> pd.DataFrame:
    """Replace best_sequential_runtime with the n=1 (single-resource) runtime.

    For runtimes with fixed startup/scheduling overhead (e.g. PyCOMPSs), comparing
    against a purely sequential baseline reflects overhead rather than scaling. Using
    n=1 as the reference isolates how well the code scales with added resources.

    Operates on rows with a populated df["n"] (the resource count); callers set n for
    the models they want rebaselined. Rows without n (e.g. correctness runs that don't
    record it) are left untouched. Not tied to any one parallelism_model.

    void_unmatched controls what happens to an output with no valid n=1 run:
      * True  (correctness, metrics.py): its speedup@k is defined against n=1, so
              with no baseline the number is meaningless. Void the output.
      * False (scaling, metrics-scaling.py): each n is its own point on a curve, so
              n>1 points stay valid on the sequential baseline even if n=1 failed.
    """
    df = df.copy()
    if "n" not in df.columns:
        return df
    mask = df["n"].notna()
    if not mask.any():
        return df

    n1_baselines = (
        df[mask & (df["n"] == 1) & df["is_valid"]]
        .groupby(["name", "output_idx"])["runtime"]
        .min()
        .reset_index()
        .rename(columns={"runtime": "n1_baseline"})
    )

    df = df.merge(n1_baselines, on=["name", "output_idx"], how="left")
    updated = mask & df["n1_baseline"].notna()
    df.loc[updated, "best_sequential_runtime"] = df.loc[updated, "n1_baseline"]
    if void_unmatched:
        # no valid n=1 run: void rather than fall back to the sequential baseline
        df.loc[mask & df["n1_baseline"].isna(), "is_valid"] = False
    return df.drop(columns=["n1_baseline"])


def parse_problem_size(problem_size: str) -> int:
    """ problem size is of format '(1<<n)' """
    if "<<" in problem_size:
        num = problem_size.split("<<")[1][:-1]
        return 2 ** int(num)
    else:
        return int(problem_size)


def prepare_run_flags(df: pd.DataFrame, count_relaxed: bool = False) -> pd.DataFrame:
    """Fill missing run flags and, unless count_relaxed, void relaxed passes.

    did_run/is_valid are NaN when a build failed; treat those as False. Without
    count_relaxed, an output that only passed after a relaxation is counted as a
    failure. Returns a copy; the input is not mutated. Shared by metrics.py,
    metrics-scaling.py, and relaxation-effect.py so "what counts as a pass" lives
    in one place.
    """
    df = df.copy()
    df["did_run"] = df["did_run"].fillna(False)     # nan when it didn't build
    df["is_valid"] = df["is_valid"].fillna(False)   # nan when it didn't build
    if not count_relaxed and "relaxation_used" in df.columns:
        df.loc[df["relaxation_used"] == True, "is_valid"] = False  # noqa: E712
    return df


def main():
    args = get_args()

    # read in input
    df = pd.read_csv(args.input_csv)

    # read in problem sizes
    with open(args.problem_sizes, "r") as f:
        problem_sizes = json.load(f)
        for problem in problem_sizes:
            for parallelism_model, problem_size in problem_sizes[problem].items():
                df.loc[(df["name"] == problem) & (df["parallelism_model"] == parallelism_model), "problem_size"] = parse_problem_size(problem_size)

    # remove rows where parallelism_model is kokkos and num_threads is 64
    df = df[~((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 64))]

    # fill missing run flags; without --relaxations, void relaxed passes
    df = prepare_run_flags(df, count_relaxed=args.relaxations)

    if args.baseline == "n1":
        # set n transiently (pycompss resource count = num_procs) for apply_n1_baseline
        df.loc[df["parallelism_model"] == "pycompss", "n"] = df["num_procs"]
        df = apply_n1_baseline(df, void_unmatched=True)

    # get only valid runs
    valid_runs = get_correctness_df(df)
    
    # get values for each k
    all_results = []
    for k in args.k:
        build_values = buildk(df, k)
        pass_values = passk(valid_runs, k)
        speedup_values = speedupk(df, k, args.n)
        speedup_max_values = speedupk_max(df, k)
        efficiency_values = efficiencyk(df, k, args.n)
        efficiency_max_values = efficiencyk_max(df, k)
        all_results.extend([build_values, pass_values, speedup_values, speedup_max_values, efficiency_values, efficiency_max_values])
    
    # merge all_results; each df has one column and the same index
    # build a new df with all the columns and the same index
    merged_df = pd.concat(all_results, axis=1).reset_index()

    # if there were no successfull builds or runs, then speedup@k will be nan after merging
    # replace NaN speedup@k values with 0.0
    for k in args.k:
        merged_df[f"speedup@{k}"] = merged_df[f"speedup@{k}"].fillna(0.0)
        merged_df[f"speedup_max@{k}"] = merged_df[f"speedup_max@{k}"].fillna(0.0)
        merged_df[f"efficiency@{k}"] = merged_df[f"efficiency@{k}"].fillna(0.0)
        merged_df[f"efficiency_max@{k}"] = merged_df[f"efficiency_max@{k}"].fillna(0.0)

    # add model name column
    if args.model_name:
        merged_df.insert(0, "model_name", args.model_name)

    # clean up column names
    column_name_map = {
        "model_name": "model",
        "parallelism_model": "execution model",
        "problem_type": "problem type",
    }
    merged_df = merged_df.rename(columns=column_name_map)

    # write to csv
    if args.output:
        merged_df.to_csv(args.output, index=False)
    else:
        pd.set_option('display.max_columns', merged_df.shape[1]+1)
        pd.set_option('display.max_rows', merged_df.shape[0]+1)
        print(merged_df)
        


if __name__ == "__main__":
    main()