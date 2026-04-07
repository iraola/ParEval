"""
Re-apply cleaning logic to already-generated output JSON files without
re-running inference. Useful when utils.py cleaning logic changes.

Usage:
    python clean-outputs.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
        --input output-DeepSeek-R1-Distill-Llama-70B.json \
        --output output-DeepSeek-R1-Distill-Llama-70B.json

Pass the same path to --input and --output to update in-place.
"""
import argparse
import json
import os

from utils import check_output_integrity, get_inference_config


def main():
    parser = argparse.ArgumentParser(description='Re-clean model outputs using utils.py InferenceConfig')
    parser.add_argument('--model', required=True, help='Model name/path (same value passed to generate-vllm.py)')
    parser.add_argument('--input', required=True, help='Input JSON file (output of generate-vllm.py)')
    parser.add_argument('--output', required=True, help='Output JSON file (use same path as --input to update in-place)')
    parser.add_argument('--prompted', action='store_true', help='Use prompted generation mode (must match how the file was generated)')
    args = parser.parse_args()

    # Resolve model path the same way generate-vllm.py does
    model_name = args.model
    local_model_path = os.path.join('..', 'models', model_name)
    if os.path.isdir(local_model_path):
        get_inference_model_path = '/'.join(local_model_path.split('/')[-2:])
    else:
        get_inference_model_path = model_name

    inference_config = get_inference_config(get_inference_model_path, prompted=args.prompted)

    with open(args.input, 'r') as f:
        responses = json.load(f)

    changed = 0
    for r in responses:
        prompt = r['prompt']
        new_outputs = []
        for raw in r['raw_outputs']:
            new_outputs.append(inference_config.clean_output(raw, prompt))

        if new_outputs != r['outputs']:
            changed += 1
        r['outputs'] = new_outputs

    with open(args.output, 'w') as f:
        json.dump(responses, f, indent=4)

    print(f"Processed {len(responses)} entries ({changed} changed). Written to {args.output}")
    check_output_integrity(responses)


if __name__ == '__main__':
    main()
