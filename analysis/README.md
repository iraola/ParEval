# Analysis

This subdirectory contains scripts for analyzing the LLM outputs and driver
results.

`create-dataframe.py` -- convert results json files into CSV format

`metrics.py` -- compute pass@k, efficiency@k, speedup@k, and build@k for a 
particular results csv file

`metrics-scaling.py` -- compute the metrics at different resource counts; used
to get scaling results

`classify-errors.py` -- classify every run attempt into a failure-mode taxonomy
(one row per attempt) and, with `--summarize`, write per-model / per-problem-type
pivot tables and category-group shares

`relaxation-effect.py` -- measure how much source-code relaxations improve
correctness, as pass@k with vs without relaxed passes counted as correct

`bin-the-stack.py` -- a utility script for analyzing The Stack dataset.

The arguments to each of these scripts can be found with `--help`. In general,
the workflow is to use `create-dataframe.py` to get a CSV results file for the
driver outputs and then feed this into `metrics.py` to get the relevant metrics.
This will in turn output another CSV file with the metrics divided by problem
type and execution model.

`run_analysis_all.sh [dir]` runs the whole correctness + scaling pipeline over
`../drivers/outputs/<dir>/`, including the failure-mode and relaxation steps below.

## Failure modes and relaxations

`classify-errors.py <dir>` reads the driver JSONs in `../drivers/outputs/<dir>/`
and turns each run attempt's stderr into a `(error_category, error_detail)` pair
via matcher registries selected per `(language, parallelism_model)` pair (add,
remove, or reorder entries to extend the taxonomy). Each row also carries a coarse
`category_group`:
- `success`: ran and passed validation natively (no relaxation applied).
- `recoverable`: failed natively but a relaxation actually rescued it (the output
  carries a non-empty `relaxations_applied`). This is the demonstrated fix set,
  not a guess; the reported pass@k does not count these as passes, and
  `relaxation-effect.py` measures the resulting gain.
- `validation`: ran to completion but the result was wrong.
- `build`: for pycompss, rejected before running because the code had no `@task`
  decorator (python is not compiled; the "build" step is a file merge gated by that check).
- `runtime`: any other failed run (crash, syntax fault, or an unfixed
  import/namespace/future error no relaxation cleared).
- `other`: unevaluated, or a failure with no traceable error text.

Run with `--include-success --summarize --digest` for the full outcome composition,
the pivot CSVs, and a Markdown digest.

`plot-failure-modes.py <dir>` renders a per-model stacked bar of the
`category_group` composition (`--fine` shows the top individual categories).

`relaxation-effect.py <dir>` reuses `metrics.py`'s correctness/pass@k helpers to
report, per `(model, problem_type)` and per model overall (`problem_type == "ALL"`),
pass@k with relaxed passes counted as failures vs as correct, and the gain. This is
the same contrast as `metrics.py` with vs without `--relaxations`, put side by side.
Outputs land in `outputs/<dir>-relaxations/relaxation_effect.csv`.

## Plotting

`plot-metrics.py`, `plot-compare.py`, and `plot-scaling.py` render the figures.
All three import `plotstyle.py`, the single source of truth for figure
aesthetics: a fixed colour + marker + short label per model (hues grouped by
family: Qwen red, DeepSeek blue, Llama green, CodeLlama teal, Mistral orange,
OpenAI purple, Gemini charcoal, StarCoder magenta), shared axis styling, and
the save format. A given model looks identical across every figure regardless
of which subset is plotted.

Each plot is written as **PDF** (vector, for LaTeX `\includegraphics`) and a
300-dpi **PNG** (preview/slides), in two variants: `<name>.pdf/png`
(title-less, captions go in the document) and `<name>_with_title.pdf/png`
(for browsing).

To add a new model, add its exact `--model-name` (as it appears in the metrics
`model` column) to the right family in `FAMILIES` in `plotstyle.py`; unknown
models fall back to grey with a warning.