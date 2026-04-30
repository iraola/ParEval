#!/bin/bash
# Create dataframes and compute metrics for all models in a given drivers/outputs/<dir>/ directory.
#
# Usage: ./create-all-dataframes.sh [dir]
#   dir    Name of the subfolder under drivers/outputs/ to process (default: kernel)
#
# Examples:
#   ./create-all-dataframes.sh
#   ./create-all-dataframes.sh kernel-20

DIR="${1:-kernel}"

DRIVER_OUTPUTS_DIR="../drivers/outputs/${DIR}"
OUTPUTS_DIR="outputs/${DIR}"

if [ ! -d "$DRIVER_OUTPUTS_DIR" ]; then
    echo "Error: driver outputs directory not found: $DRIVER_OUTPUTS_DIR"
    exit 1
fi

mkdir -p "$OUTPUTS_DIR"

for INPUT_FILE in "$DRIVER_OUTPUTS_DIR"/output_drivers_*.json; do
    [ -f "$INPUT_FILE" ] || continue

    BASENAME=$(basename "$INPUT_FILE")               # output_drivers_Qwen3-32B.json
    MODEL_NAME="${BASENAME#output_drivers_}"         # Qwen3-32B.json
    MODEL_NAME="${MODEL_NAME%.json}"                 # Qwen3-32B

    DATAFRAME_FILE="${OUTPUTS_DIR}/dataframe_${MODEL_NAME}.csv"
    METRICS_FILE="${OUTPUTS_DIR}/metrics_${MODEL_NAME}.csv"

    echo "Processing: $MODEL_NAME"
    echo "  Input:     $INPUT_FILE"
    echo "  Dataframe: $DATAFRAME_FILE"
    echo "  Metrics:   $METRICS_FILE"

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

    echo "Done: $MODEL_NAME"
    echo "---"
done

echo "All models processed."
