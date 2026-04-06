#!/bin/bash
#SBATCH --job-name=pareval
#SBATCH --qos=acc_debug
#SBATCH --account=bsc19
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --ntasks=1
#SBATCH --output=logs-generate-all-%j.out
#SBATCH --error=logs-generate-all-%j.err

NUM_SAMPLES_PER_PROMPT="3"
MAX_NEW_TOKENS="4096"
MODEL_LIST="models.txt"

if [ ! -f "$MODEL_LIST" ]; then
    echo "Error: Model list file '$MODEL_LIST' not found."
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
        model_max_tokens="8192"
    else
        model_max_tokens="$MAX_NEW_TOKENS"
    fi

    rm -f cache.json
    touch cache.json

    output_file="output-${model_name}.json"
    if [ -f "$output_file" ]; then
        echo "Output file $output_file already exists. Skipping generation for $model_name."
        continue
    fi

    python3 generate-vllm.py \
        --prompts ../prompts/generation-prompts-simple.json \
        --model "$model" \
        --output "$output_file" \
        --num_samples_per_prompt "$NUM_SAMPLES_PER_PROMPT" \
        --max_new_tokens "$model_max_tokens" \
        --cache cache.json \
        --enforce_eager

done < "$MODEL_LIST"