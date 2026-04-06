#!/bin/bash
# Download a model from Hugging Face to $PROJECTS/models 
# with retries on interruption.
#
# Usage: ./download_model.sh <model_id> [extra_hf_args...]
# Example: ./download_model.sh meta-llama/Llama-3.3-70B-Instruct

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_id> [extra_hf_args...]"
    echo "Example: $0 meta-llama/Llama-3.3-70B-Instruct"
    exit 1
fi

MODEL_ID="$1"
shift
EXTRA_ARGS=("$@")

DOWNLOAD_DIR="$PROJECTS/models/${MODEL_ID}"
RETRY_DELAY=1
MAX_RETRIES=100
ATTEMPT=0

echo "============================================"
echo "Downloading: ${MODEL_ID}"
echo "Local dir:   ${DOWNLOAD_DIR}"
echo "Extra args:  ${EXTRA_ARGS[*]:-none}"
echo "============================================"

mkdir -p "${DOWNLOAD_DIR}"

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempt ${ATTEMPT}/${MAX_RETRIES}"

    if hf download "${MODEL_ID}" \
        --local-dir "${DOWNLOAD_DIR}" \
        --exclude 'original/*' \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
        echo ""
        echo "============================================"
        echo "Download complete: ${MODEL_ID}"
        echo "Location: ${DOWNLOAD_DIR}"
        echo "Attempts: ${ATTEMPT}"
        echo "============================================"
        exit 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download interrupted. Retrying in ${RETRY_DELAY}s..."
    sleep "${RETRY_DELAY}"
done

echo "ERROR: Failed after ${MAX_RETRIES} attempts."
exit 1