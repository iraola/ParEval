#!/bin/bash
# Run the driver evaluation for all models in "generate/outputs" directories.
# Skips models whose output already exists under drivers/outputs/<dir>/.
#
# Usage: ./run-all-models.sh [dir] [--timeout N] [--relaxation MODE]
#   dir              Name of the subfolder under generate/outputs/ to process (default: kernel)
#   --timeout N      Run timeout in seconds (default: 150)
#   --relaxation MODE  Relaxation mode passed to run-all.py (default: all)
#
# Examples:
#   ./run-all-models.sh
#   ./run-all-models.sh kernel
#   ./run-all-models.sh workflow --timeout 300
#   ./run-all-models.sh kernel --relaxation none --timeout 60

DIR="${1:-kernel}"
shift 2>/dev/null

TIMEOUT=150
RELAXATION=all
PROBLEM_SIZE_OVERRIDE=none   # pass a number (e.g. 5) to override all problem sizes

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        --relaxation)
            RELAXATION="$2"; shift 2 ;;
        --problem-size-override)
            PROBLEM_SIZE_OVERRIDE="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done
GENERATE_OUTPUTS_DIR="../generate/outputs/${DIR}"
DRIVER_OUTPUTS_DIR="outputs/${DIR}"
LOGS_DIR="logs/${DIR}"

if [ ! -d "$GENERATE_OUTPUTS_DIR" ]; then
    echo "Error: generate outputs directory not found: $GENERATE_OUTPUTS_DIR"
    exit 1
fi

mkdir -p "$DRIVER_OUTPUTS_DIR"
mkdir -p "$LOGS_DIR"

for INPUT_FILE in "$GENERATE_OUTPUTS_DIR"/output-*.json; do
    [ -f "$INPUT_FILE" ] || continue

    BASENAME=$(basename "$INPUT_FILE")                  # output-Qwen3-32B.json
    MODEL_NAME="${BASENAME#output-}"                    # Qwen3-32B.json
    MODEL_NAME="${MODEL_NAME%.json}"                    # Qwen3-32B

    OUTPUT_FILE="${DRIVER_OUTPUTS_DIR}/output_drivers_${MODEL_NAME}.json"
    LOG_FILE="${LOGS_DIR}/log_drivers_${MODEL_NAME}.txt"
    ARTIFACTS_DIR="artifacts/${DIR}/${MODEL_NAME}"

    mkdir -p "$ARTIFACTS_DIR"

    echo "Running: $MODEL_NAME"
    echo "  Input:     $INPUT_FILE"
    echo "  Output:    $OUTPUT_FILE"
    echo "  Log:       $LOG_FILE"
    echo "  Artifacts: $ARTIFACTS_DIR"

    python3 run-all.py "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        --artifacts-dir "$ARTIFACTS_DIR" \
        --log-build-errors \
        --log-runs \
        --log DEBUG \
        --run-timeout "$TIMEOUT" \
        --relaxation "$RELAXATION" \
        --problem-size-override "$PROBLEM_SIZE_OVERRIDE" \
        --resume \
        --yes-to-all 2>&1 | tee -a "$LOG_FILE"

    echo "Done: $MODEL_NAME"
    echo "---"
done

echo "All models processed."
