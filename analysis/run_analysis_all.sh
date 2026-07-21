#!/bin/bash
# Create dataframes and compute metrics for all models in a given drivers/outputs/<dir>/ directory.
# Processes two input families found in that directory:
#   output_drivers_*.json (correctness)  → outputs/<dir>/            + plots/<dir>/
#   output_scaling_*.json (scaling)      → outputs/<dir>-scaling/    + plots/<dir>-scaling/
# Each family also gets a -relaxations variant (relaxed passes counted as correct).
# The scaling family additionally produces curve_*.csv (speedup/efficiency vs resource
# count via metrics-scaling.py) and the matching speedup/efficiency scaling-curve plots.
#
# Usage: ./run_analysis_all.sh [dir]
#   dir    Name of the subfolder under drivers/outputs/ to process (default: kernel)
#
# Examples:
#   ./run_analysis_all.sh
#   ./run_analysis_all.sh kernel-20

DIR="${1:-kernel}"

DRIVER_OUTPUTS_DIR="../drivers/outputs/${DIR}"

if [ ! -d "$DRIVER_OUTPUTS_DIR" ]; then
    echo "Error: driver outputs directory not found: $DRIVER_OUTPUTS_DIR"
    exit 1
fi

# Resource counts swept by metrics-scaling.py for the scaling curves.
SCALING_N="1 2 4 8 16 32 64"

# process_set <input-prefix> <outputs-dir> <outputs-relax-dir> <run-scaling-curves>
process_set() {
    local prefix="$1"
    local outputs_dir="$2"
    local outputs_relax_dir="$3"
    local run_scaling_curves="$4"
    local found=0

    for INPUT_FILE in "$DRIVER_OUTPUTS_DIR"/${prefix}*.json; do
        [ -f "$INPUT_FILE" ] || continue
        found=1
        mkdir -p "$outputs_dir" "$outputs_relax_dir"

        BASENAME=$(basename "$INPUT_FILE")          # output_scaling_Qwen3-32B.json
        MODEL_NAME="${BASENAME#$prefix}"            # Qwen3-32B.json
        MODEL_NAME="${MODEL_NAME%.json}"            # Qwen3-32B

        DATAFRAME_FILE="${outputs_dir}/dataframe_${MODEL_NAME}.csv"
        METRICS_FILE="${outputs_dir}/metrics_${MODEL_NAME}.csv"
        METRICS_RELAX_FILE="${outputs_relax_dir}/metrics_${MODEL_NAME}.csv"
        CURVE_FILE="${outputs_dir}/curve_${MODEL_NAME}.csv"

        echo "Processing: $MODEL_NAME  (${prefix%_}*)"
        echo "  Input:          $INPUT_FILE"
        echo "  Dataframe:      $DATAFRAME_FILE"
        echo "  Metrics:        $METRICS_FILE"
        echo "  Metrics (relax):$METRICS_RELAX_FILE"

        if [ -f "$DATAFRAME_FILE" ]; then
            echo "  (dataframe already exists, skipping create-dataframe.py)"
        else
            ../.venv/bin/python create-dataframe.py "$INPUT_FILE" -o "$DATAFRAME_FILE"
        fi

        if [ -f "$METRICS_FILE" ]; then
            echo "  (metrics already exists, skipping metrics.py)"
        else
            ../.venv/bin/python metrics.py "$DATAFRAME_FILE" -o "$METRICS_FILE" --model-name "$MODEL_NAME" --baseline n1
        fi

        if [ -f "$METRICS_RELAX_FILE" ]; then
            echo "  (metrics (relax) already exists, skipping metrics.py --relaxations)"
        else
            ../.venv/bin/python metrics.py "$DATAFRAME_FILE" -o "$METRICS_RELAX_FILE" --model-name "$MODEL_NAME" --relaxations --baseline n1
        fi

        if [ "$run_scaling_curves" = "true" ]; then
            CURVE_RELAX_FILE="${outputs_relax_dir}/curve_${MODEL_NAME}.csv"
            echo "  Scaling curve:  $CURVE_FILE"
            if [ -f "$CURVE_FILE" ]; then
                echo "  (scaling curve already exists, skipping metrics-scaling.py)"
            else
                ../.venv/bin/python metrics-scaling.py "$DATAFRAME_FILE" \
                    --execution-model pycompss -k 1 -n $SCALING_N \
                    --model-name "$MODEL_NAME" --baseline n1 -o "$CURVE_FILE"
            fi
            if [ -f "$CURVE_RELAX_FILE" ]; then
                echo "  (scaling curve (relax) already exists, skipping metrics-scaling.py --relaxations)"
            else
                ../.venv/bin/python metrics-scaling.py "$DATAFRAME_FILE" \
                    --execution-model pycompss -k 1 -n $SCALING_N \
                    --model-name "$MODEL_NAME" --baseline n1 -o "$CURVE_RELAX_FILE" --relaxations
            fi
        fi

        echo "Done: $MODEL_NAME"
        echo "---"
    done

    if [ "$found" -eq 0 ]; then
        echo "No ${prefix}*.json files in $DRIVER_OUTPUTS_DIR, skipping."
        return
    fi

    echo "Plotting metrics for ${outputs_dir} ..."
    ../.venv/bin/python plot-metrics.py "$outputs_dir" --dump-csv
    ../.venv/bin/python plot-metrics.py "$outputs_relax_dir" --dump-csv
    echo "Plotting metrics (top 5) ..."
    ../.venv/bin/python plot-metrics.py "$outputs_dir" --top 5
    ../.venv/bin/python plot-metrics.py "$outputs_relax_dir" --top 5

    if [ "$run_scaling_curves" = "true" ]; then
        echo "Plotting scaling curves for ${outputs_dir} ..."
        ../.venv/bin/python plot-scaling.py "$outputs_dir"
        ../.venv/bin/python plot-scaling.py "$outputs_dir" --top 5
        ../.venv/bin/python plot-scaling.py "$outputs_relax_dir"
        ../.venv/bin/python plot-scaling.py "$outputs_relax_dir" --top 5
    fi
}

# correctness
process_set "output_drivers_" "outputs/${DIR}"          "outputs/${DIR}-relaxations"          "false"

# failure-mode taxonomy + relaxation effect (correctness only)
if ls "$DRIVER_OUTPUTS_DIR"/output_drivers_*.json >/dev/null 2>&1; then
    echo "Classifying errors for ${DIR} ..."
    ../.venv/bin/python classify-errors.py "$DIR" --include-success --summarize --digest
    echo "Computing relaxation effect for ${DIR} ..."
    ../.venv/bin/python relaxation-effect.py "$DIR" --digest
    echo "Plotting failure modes for ${DIR} ..."
    ../.venv/bin/python plot-failure-modes.py "outputs/${DIR}"
    ../.venv/bin/python plot-failure-modes.py "outputs/${DIR}" --fine
fi

# scaling
process_set "output_scaling_" "outputs/${DIR}-scaling"  "outputs/${DIR}-scaling-relaxations"  "true"

echo "All sets processed."
