#!/bin/bash
#SBATCH --job-name=pareval
#SBATCH --qos=acc_ehpc
#SBATCH --exclusive
#SBATCH --account=ehpc721
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --ntasks=1
#SBATCH --output=logs-generate-all-%j.out
#SBATCH --error=logs-generate-all-%j.err

# Usage: sbatch generate-all.sh [dir] [--prompts FILE] [--samples N] [--tokens N]
#   dir            Subfolder name under generate/outputs/ and prompts convention (default: kernel-guided)
#   --prompts FILE Override prompts JSON (default: ../prompts/generation-prompts-<dir>-pycompss.json)
#   --samples N    Samples per prompt (default: 10)
#   --tokens N     Max new tokens (default: 8192)
#
# Examples:
#   sbatch generate-all.sh
#   sbatch generate-all.sh kernel --prompts ../prompts/generation-prompts-simple.json
#   sbatch generate-all.sh kernel-guided --samples 3

DIR="${1:-kernel}"
shift 2>/dev/null

NUM_SAMPLES_PER_PROMPT="10"
MAX_NEW_TOKENS="8192"
PROMPTS_FILE=""
MODEL_LIST="models.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompts)  PROMPTS_FILE="$2";          shift 2 ;;
        --samples)  NUM_SAMPLES_PER_PROMPT="$2"; shift 2 ;;
        --tokens)   MAX_NEW_TOKENS="$2";         shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$PROMPTS_FILE" ]; then
    PROMPTS_FILE="../prompts/generation-prompts-${DIR}-pycompss.json"
fi

if [ ! -f "$MODEL_LIST" ]; then
    echo "Error: Model list file '$MODEL_LIST' not found."
    exit 1
fi

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: Prompts file not found: $PROMPTS_FILE"
    exit 1
fi

# Move to the project directory
cd ..

# Load modules
module load intel mkl python/3.12.1
unset PYTHONPATH
source .venv/bin/activate
module load sqlite3

cd generate/

# Add fix for Mistral-Small-3.2-24B error (shouldn't hurt GPU performance)
export OMP_NUM_THREADS=1

OUTPUT_DIR="outputs/${DIR}"
mkdir -p "$OUTPUT_DIR"

echo "DIR:     $DIR"
echo "PROMPTS: $PROMPTS_FILE"
echo "OUTPUT:  $OUTPUT_DIR"
echo "SAMPLES: $NUM_SAMPLES_PER_PROMPT"
echo "TOKENS:  $MAX_NEW_TOKENS"

while IFS= read -r model || [ -n "$model" ]; do
    # Skip empty lines and comments
    [[ -z "$model" || "$model" == \#* ]] && continue

    model_name=$(basename "$model")

    echo ""
    echo "============================================"
    echo "Running: $model"
    echo "============================================"

    # Reasoning models need more tokens for their think blocks
    if [[ "$model" == *"R1"* || "$model" == *"Qwen3"* ]]; then
        model_max_tokens=$((MAX_NEW_TOKENS * 2))
    else
        model_max_tokens="$MAX_NEW_TOKENS"
    fi

    rm -f cache.json
    touch cache.json

    output_file="${OUTPUT_DIR}/output-${model_name}.json"
    if [ -f "$output_file" ]; then
        echo "Output file $output_file already exists. Skipping generation for $model_name."
        continue
    fi

    python3 generate-vllm.py \
        --prompts "$PROMPTS_FILE" \
        --model "$model" \
        --output "$output_file" \
        --num_samples_per_prompt "$NUM_SAMPLES_PER_PROMPT" \
        --max_new_tokens "$model_max_tokens" \
        --cache cache.json \
        --enforce_eager

done < "$MODEL_LIST"
