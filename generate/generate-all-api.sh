#!/bin/bash
# Run all API models (OpenAI, Gemini) over a prompt set.
#
# Usage: ./generate-all-api.sh [dir] [--prompts FILE] [--samples N] [--tokens N]
#   dir            Entry point under outputs/ and prompts convention (default: kernel)
#   --prompts FILE Override prompts JSON (default: ../prompts/generation-prompts-<dir>-pycompss.json)
#   --samples N    Samples per prompt (default: 10)
#   --tokens N     Max new tokens (default: per-language default chosen by each script)
#
# Requires <PROVIDE>_API_KEY environment variables set in `.env`.
#
# Examples:
#   ./generate-all-api.sh
#   ./generate-all-api.sh kernel-guided --samples 3
#   ./generate-all-api.sh kernel --prompts ../prompts/generation-prompts-simple.json

source ../.env

DIR="${1:-kernel}"
shift 2>/dev/null

NUM_SAMPLES_PER_PROMPT="1"
MAX_NEW_TOKENS="32768"
PROMPTS_FILE=""
PYTHON="${PYTHON:-../.venv/bin/python}"

# "provider model" pairs; provider selects the generate-<provider>.py script.
MODELS=(
    "openai gpt-4-turbo-2024-04-09"
    "openai gpt-5.5-2026-04-23"
    "gemini gemini-3.5-flash"
)

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

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: Prompts file not found: $PROMPTS_FILE"
    exit 1
fi

OUTPUT_DIR="outputs/${DIR}"
mkdir -p "$OUTPUT_DIR"

echo "DIR:     $DIR"
echo "PROMPTS: $PROMPTS_FILE"
echo "OUTPUT:  $OUTPUT_DIR"
echo "SAMPLES: $NUM_SAMPLES_PER_PROMPT"
echo "TOKENS:  ${MAX_NEW_TOKENS:-<per-language default>}"

for entry in "${MODELS[@]}"; do
    read -r provider model <<< "$entry"
    script="generate-${provider}.py"
    output_file="${OUTPUT_DIR}/output-${provider}-${model}.json"

    echo ""
    echo "============================================"
    echo "Running: $provider / $model"
    echo "============================================"

    if [ -f "$output_file" ]; then
        echo "Output file $output_file already exists. Skipping $model."
        continue
    fi

    cmd=("$PYTHON" "$script"
        --model "$model"
        --prompts "$PROMPTS_FILE"
        -o "$output_file"
        --num-samples-per-prompt "$NUM_SAMPLES_PER_PROMPT")
    [ -n "$MAX_NEW_TOKENS" ] && cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")

    "${cmd[@]}"
done
