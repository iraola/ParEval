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

while IFS= read -r model || [ -n "$model" ]; do
    # Skip empty lines and comments
    [[ -z "$model" || "$model" == \#* ]] && continue

    model_name=$(basename "$model")

    echo ""
    echo "============================================"
    echo "Running: $model"
    echo "============================================"

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
        --cache cache.json \
        --enforce_eager

done < "$MODEL_LIST"