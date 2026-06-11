#!/bin/bash
#SBATCH --job-name=pareval-scaling
#SBATCH --qos=gp_ehpc
#SBATCH --exclusive
#SBATCH --account=ehpc721
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH --array=0-19
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

# Run scaling evaluation for outputs that passed correctness.
#
# Each SLURM array task handles one model.  Within the task all scaling
# configs (num_procs = 1,2,4,8,16,32,64) for each passing output are run
# concurrently in greedy waves that fill the node.
#
# Prerequisites:
#   Correctness run must have already produced:
#     drivers/outputs/<dir>/output_drivers_<model>.json
#
# Usage:
#   sbatch [--array=0-$((N-1))] run-all-models-slurm-scaling.sh <dir> [options]
#
# Options:
#   --timeout N            Run timeout per runcompss call in seconds (default: 300)
#   --relaxations MODE     Relaxation mode (default: all)
#   --problem-sizes FILE   Problem-sizes JSON (default: problem-sizes.json)
#
# Examples:
#   sbatch --array=0-19 run-all-models-slurm-scaling.sh kernel-20
#   sbatch --array=0-19 run-all-models-slurm-scaling.sh kernel-20 --timeout 600

DIR="${1:-kernel}"
shift 2>/dev/null

TIMEOUT=300
RELAXATIONS=all
PROBLEM_SIZES=problem-sizes.json

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        --relaxations)
            RELAXATIONS="$2"; shift 2 ;;
        --problem-sizes)
            PROBLEM_SIZES="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Move to the project directory and activate environment
cd ..
module load hdf5 python/3.12.1
unset PYTHONPATH
source .venv_gpp/bin/activate
module load sqlite3
module load COMPSs/3.4.post2603

cd drivers/

GENERATE_OUTPUTS_DIR="../generate/outputs/${DIR}"
DRIVER_OUTPUTS_DIR="outputs/${DIR}"
LOGS_DIR="logs/${DIR}"

if [ ! -d "$GENERATE_OUTPUTS_DIR" ]; then
    echo "Error: generate outputs directory not found: $GENERATE_OUTPUTS_DIR"
    exit 1
fi

# Build sorted model list and pick this task's model by array task ID
mapfile -t MODELS < <(ls "${GENERATE_OUTPUTS_DIR}"/output-*.json | sort -f | sed 's|.*/output-||;s|\.json$||')

if [[ $SLURM_ARRAY_TASK_ID -ge ${#MODELS[@]} ]]; then
    echo "Array task ID $SLURM_ARRAY_TASK_ID >= number of models (${#MODELS[@]}); nothing to do."
    exit 0
fi

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

CORRECTNESS_FILE="${DRIVER_OUTPUTS_DIR}/output_drivers_${MODEL_NAME}.json"
SCALING_FILE="${DRIVER_OUTPUTS_DIR}/output_scaling_${MODEL_NAME}.json"
LOG_FILE="${LOGS_DIR}/log_scaling_${MODEL_NAME}.txt"
ARTIFACTS_DIR="artifacts/${DIR}/${MODEL_NAME}"
SCRATCH_DIR="../scratch/${DIR}/${MODEL_NAME}"
SLOTS_FILE="${SCRATCH_DIR}/slots_scaling.json"

if [ ! -f "$CORRECTNESS_FILE" ]; then
    echo "Error: correctness output not found: $CORRECTNESS_FILE"
    echo "Run the correctness evaluation first."
    exit 1
fi

mkdir -p "$DRIVER_OUTPUTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$ARTIFACTS_DIR"
mkdir -p "$SCRATCH_DIR"

echo "Array task:   ${SLURM_ARRAY_TASK_ID} (job ${SLURM_ARRAY_JOB_ID})"
echo "Model:        $MODEL_NAME"
echo "Correctness:  $CORRECTNESS_FILE"
echo "Scaling out:  $SCALING_FILE"
echo "Log:          $LOG_FILE"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "CPUs on node: $SLURM_CPUS_ON_NODE"
echo "Timeout:      $TIMEOUT s"

# Filter correctness output: reset passing outputs to strings, keep failed as-is.
# Skip if SCALING_FILE already exists so that a resubmitted job can resume from
# where it left off rather than losing intermediate results.
if [ -f "$SCALING_FILE" ]; then
    echo "Scaling file already exists, skipping filter-correct.py (resuming): $SCALING_FILE"
else
    python3 filter-correct.py "$CORRECTNESS_FILE" "$SCALING_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: filter-correct.py failed."
        exit 1
    fi
fi

# One slot = whole node; wave execution packs configs within it
python3 generate_resource_slots.py \
    --cpus-per-node "$SLURM_CPUS_ON_NODE" \
    --cpus-per-slot "$SLURM_CPUS_ON_NODE" \
    --output "$SLOTS_FILE"
if [ $? -ne 0 ]; then
    echo "Error: failed to generate resource slots."
    exit 1
fi

python3 run-all.py "$SCALING_FILE" \
    -o "$SCALING_FILE" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --scratch-dir "$SCRATCH_DIR" \
    --launch-configs launch-configs-slurm-scaling.json \
    --problem-sizes "$PROBLEM_SIZES" \
    --resource-slots "$SLOTS_FILE" \
    --include-models pycompss \
    --log-build-errors \
    --log-runs \
    --log DEBUG \
    --run-timeout "$TIMEOUT" \
    --relaxations "$RELAXATIONS" \
    --resume \
    --yes-to-all 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo "Done: $MODEL_NAME (exit $EXIT_CODE)"
exit $EXIT_CODE
