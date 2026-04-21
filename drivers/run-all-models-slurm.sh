#!/bin/bash
#SBATCH --job-name=pareval-drivers
#SBATCH --qos=gp_ehpc
#SBATCH --exclusive
#SBATCH --account=ehpc721
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --output=logs-run-all-models-%j.out
#SBATCH --error=logs-run-all-models-%j.err

# Run the driver evaluation for all models in "generate/outputs" directories.
# Skips models whose output already exists under drivers/outputs/<dir>/.
#
# Usage: sbatch run-all-models-slurm.sh [dir] [--timeout N] [--relaxation MODE]
#   dir              Name of the subfolder under generate/outputs/ to process (default: kernel)
#   --timeout N      Run timeout in seconds (default: 150)
#   --relaxation MODE  Relaxation mode passed to run-all.py (default: all)
#
# Examples:
#   sbatch run-all-models-slurm.sh
#   sbatch run-all-models-slurm.sh kernel
#   sbatch run-all-models-slurm.sh workflow --timeout 300
#   sbatch run-all-models-slurm.sh kernel --relaxation none --timeout 60

DIR="${1:-kernel}"
shift 2>/dev/null

TIMEOUT=150
RELAXATION=all

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        --relaxation)
            RELAXATION="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Move to the project directory and activate environment
cd ..
module load intel mkl python/3.12.1
unset PYTHONPATH
source .venv/bin/activate
module load sqlite3
module load COMPSs/3.4.rc1

cd drivers/

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
    SCRATCH_DIR="../scratch/${DIR}/${MODEL_NAME}"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping $MODEL_NAME (output already exists: $OUTPUT_FILE)"
        continue
    fi

    mkdir -p "$ARTIFACTS_DIR"
    mkdir -p "$SCRATCH_DIR"

    echo "Running: $MODEL_NAME"
    echo "  Input:     $INPUT_FILE"
    echo "  Output:    $OUTPUT_FILE"
    echo "  Log:       $LOG_FILE"
    echo "  Artifacts: $ARTIFACTS_DIR"

    python3 run-all.py "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        --artifacts-dir "$ARTIFACTS_DIR" \
        --scratch-dir "$SCRATCH_DIR" \
        --log-build-errors \
        --log-runs \
        --log DEBUG \
        --run-timeout "$TIMEOUT" \
        --relaxation "$RELAXATION" \
        --yes-to-all 2>&1 | tee "$LOG_FILE"

    echo "Done: $MODEL_NAME"
    echo "---"
done

echo "All models processed."
