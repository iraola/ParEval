#!/bin/bash
#SBATCH --job-name=pareval-drivers
#SBATCH --qos=gp_ehpc
#SBATCH --exclusive
#SBATCH --account=ehpc721
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH --array=0-19
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

# Run driver evaluation as a SLURM job array — one task per model.
# Each task picks its model by index from the sorted list of generate output files.
#
# Usage:
#   sbatch run-array-slurm.sh [dir] [--timeout N] [--relaxations MODE]
#
# The --array bound must match the number of models. Use:
#   N=$(ls ../generate/outputs/<dir>/output-*.json | wc -l)
#   sbatch --array=0-$((N-1)) run-array-slurm.sh <dir>
#
# To rerun specific failed tasks (e.g. tasks 3 and 7):
#   sbatch --array=3,7 run-array-slurm.sh <dir>
#
# Examples:
#   sbatch --array=0-19 run-array-slurm.sh kernel-20
#   sbatch --array=0-19 run-array-slurm.sh kernel-20 --timeout 300
#   sbatch --array=3,7  run-array-slurm.sh kernel-20

DIR="${1:-kernel}"
shift 2>/dev/null

TIMEOUT=150
RELAXATIONS=all

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        --relaxations)
            RELAXATIONS="$2"; shift 2 ;;
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

# Build sorted model list and pick this task's model by array task ID
mapfile -t MODELS < <(ls "${GENERATE_OUTPUTS_DIR}"/output-*.json | sort | sed 's|.*/output-||;s|\.json$||')

if [[ $SLURM_ARRAY_TASK_ID -ge ${#MODELS[@]} ]]; then
    echo "Array task ID $SLURM_ARRAY_TASK_ID >= number of models (${#MODELS[@]}); nothing to do."
    exit 0
fi

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

INPUT_FILE="${GENERATE_OUTPUTS_DIR}/output-${MODEL_NAME}.json"
OUTPUT_FILE="${DRIVER_OUTPUTS_DIR}/output_drivers_${MODEL_NAME}.json"
LOG_FILE="${LOGS_DIR}/log_drivers_${MODEL_NAME}.txt"
ARTIFACTS_DIR="artifacts/${DIR}/${MODEL_NAME}"
SCRATCH_DIR="../scratch/${DIR}/${MODEL_NAME}"

mkdir -p "$DRIVER_OUTPUTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$ARTIFACTS_DIR"
mkdir -p "$SCRATCH_DIR"

echo "Array task: ${SLURM_ARRAY_TASK_ID} (job ${SLURM_ARRAY_JOB_ID})"
echo "Model:      $MODEL_NAME"
echo "Input:      $INPUT_FILE"
echo "Output:     $OUTPUT_FILE"
echo "Log:        $LOG_FILE"
echo "Artifacts:  $ARTIFACTS_DIR"

python3 run-all.py "$INPUT_FILE" \
    -o "$OUTPUT_FILE" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --scratch-dir "$SCRATCH_DIR" \
    --log-build-errors \
    --log-runs \
    --log DEBUG \
    --run-timeout "$TIMEOUT" \
    --relaxations "$RELAXATIONS" \
    --resume \
    --yes-to-all 2>&1 | tee -a "$LOG_FILE"

echo "Done: $MODEL_NAME"
