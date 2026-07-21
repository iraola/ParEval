""" Compute the metrics over the data for various resource counts.
"""
# std imports
import argparse
import json

# tpl imports
import numpy as np
import pandas as pd

import metrics


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=str, help="Input CSV file containing the test cases.")
    parser.add_argument("-k", "--k", type=int, default=1, help="K value for speedup@k and efficiency@k")
    parser.add_argument("-n", "--n", type=int, nargs='+', default=[1,2,4,8,16,32,64,128,256,512], help="Number of resources for speedup@k and efficiency@k")
    parser.add_argument("--execution-model", choices=['mpi', 'mpi+omp', 'omp', 'kokkos', 'pycompss'], default='mpi', help="Execution model to use for speedup@k and efficiency@k")
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

def speedupk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """ Compute the speedup@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # choose processor count; hardcoded right now
    df = df[df["n"] == n]
    df = df.copy()

    col = f"speedup_{n}@{k}"
    if df.empty:
        idx = pd.MultiIndex.from_arrays([[], []], names=["parallelism_model", "problem_type"])
        return pd.DataFrame({col: pd.Series(dtype=float)}, index=idx)

    # use min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _speedupk
    df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
            lambda row: metrics._speedupk(row["runtime"], np.min(row["best_sequential_runtime"]),
                                          k, col_name=f"speedup_{n}@{{}}")
        ).reset_index()

    # compute the mean speedup@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"speedup_{n}@{k}": "mean"})

    return df

def efficiencyk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """ Compute the efficiency@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # choose processor count; hardcoded right now
    df = df[df["n"] == n]
    df = df.copy()

    col = f"efficiency_{n}@{k}"
    if df.empty:
        idx = pd.MultiIndex.from_arrays([[], []], names=["parallelism_model", "problem_type"])
        return pd.DataFrame({col: pd.Series(dtype=float)}, index=idx)

    # use min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _efficiencyk
    df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
            lambda row: metrics._efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]),
                                             k, row["n"], col_name=f"efficiency_{n}@{{}}")
        ).reset_index()

    # compute the mean efficiency@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency_{n}@{k}": "mean"})

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
                df.loc[(df["name"] == problem) & (df["parallelism_model"] == parallelism_model), "problem_size"] = metrics.parse_problem_size(problem_size)

    # remove rows where parallelism_model is kokkos and num_threads is 64
    #df = df[~((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 64))]

    # fill missing run flags; without --relaxations, void relaxed passes
    df = metrics.prepare_run_flags(df, count_relaxed=args.relaxations)

    model = args.execution_model
    if model not in metrics.MODELS:
        raise NotImplementedError(f"Unsupported execution model {model}")
    # select the model and set its resource count as n (see metrics.MODELS)
    df = df[df["parallelism_model"] == model].copy()
    df["n"] = metrics.resource_count(df)

    if args.baseline == "n1":
        # void_unmatched=False: keep n>1 points even when n=1 failed (each n is a point)
        df = metrics.apply_n1_baseline(df, void_unmatched=False)

    # get values for each k
    all_results = []
    for n in args.n:
        speedup_values = speedupk(df, args.k, n)
        efficiency_values = efficiencyk(df, args.k, n)
        all_results.extend([speedup_values, efficiency_values])
    
    # merge all_results; each df has one column and the same index
    # build a new df with all the columns and the same index
    merged_df = pd.concat(all_results, axis=1).reset_index()

    # if there were no successfull builds or runs, then speedup@k will be nan after merging
    # replace NaN speedup@k values with 0.0
    for n in args.n:
        merged_df[f"speedup_{n}@{args.k}"] = merged_df[f"speedup_{n}@{args.k}"].fillna(0.0)
        merged_df[f"efficiency_{n}@{args.k}"] = merged_df[f"efficiency_{n}@{args.k}"].fillna(0.0)

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