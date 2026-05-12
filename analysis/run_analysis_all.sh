#!/bin/bash
# Create dataframes and compute metrics for all models in a given drivers/outputs/<dir>/ directory.
# Produces two metrics directories:
#   outputs/<dir>/             — base (relaxed passes count as failures)
#   outputs/<dir>-relaxations/ — relaxations counted as correct
#
# Usage: ./run_analysis_all.sh [dir]
#   dir    Name of the subfolder under drivers/outputs/ to process (default: kernel)
#
# Examples:
#   ./run_analysis_all.sh
#   ./run_analysis_all.sh kernel-20

DIR="${1:-kernel}"

DRIVER_OUTPUTS_DIR="../drivers/outputs/${DIR}"
OUTPUTS_DIR="outputs/${DIR}"
OUTPUTS_RELAX_DIR="outputs/${DIR}-relaxations"

if [ ! -d "$DRIVER_OUTPUTS_DIR" ]; then
    echo "Error: driver outputs directory not found: $DRIVER_OUTPUTS_DIR"
    exit 1
fi

mkdir -p "$OUTPUTS_DIR"
mkdir -p "$OUTPUTS_RELAX_DIR"

for INPUT_FILE in "$DRIVER_OUTPUTS_DIR"/output_drivers_*.json; do
    [ -f "$INPUT_FILE" ] || continue

    BASENAME=$(basename "$INPUT_FILE")               # output_drivers_Qwen3-32B.json
    MODEL_NAME="${BASENAME#output_drivers_}"         # Qwen3-32B.json
    MODEL_NAME="${MODEL_NAME%.json}"                 # Qwen3-32B

    DATAFRAME_FILE="${OUTPUTS_DIR}/dataframe_${MODEL_NAME}.csv"
    METRICS_FILE="${OUTPUTS_DIR}/metrics_${MODEL_NAME}.csv"
    METRICS_RELAX_FILE="${OUTPUTS_RELAX_DIR}/metrics_${MODEL_NAME}.csv"

    echo "Processing: $MODEL_NAME"
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
        ../.venv/bin/python metrics.py "$DATAFRAME_FILE" -o "$METRICS_FILE" --model-name "$MODEL_NAME"
    fi

    if [ -f "$METRICS_RELAX_FILE" ]; then
        echo "  (metrics (relax) already exists, skipping metrics.py --relaxations)"
    else
        ../.venv/bin/python metrics.py "$DATAFRAME_FILE" -o "$METRICS_RELAX_FILE" --model-name "$MODEL_NAME" --relaxations
    fi

    echo "Done: $MODEL_NAME"
    echo "---"
done

echo "All models processed."

echo "Plotting base metrics (all models)..."
../.venv/bin/python plot-metrics.py "$OUTPUTS_DIR"
../.venv/bin/python plot-metrics.py "${OUTPUTS_DIR}-relaxations"

echo "Plotting base metrics (top 5)..."
../.venv/bin/python plot-metrics.py "$OUTPUTS_DIR" --top 5
../.venv/bin/python plot-metrics.py "${OUTPUTS_DIR}-relaxations" --top 5
