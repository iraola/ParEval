""" Get the model outputs from OpenAI's API.
    author: Daniel Nichols
    date: January 2024
"""
# std imports
from argparse import ArgumentParser
import json
import os
import time
from typing import Optional

# tpl imports
from tqdm import tqdm
from openai import OpenAI

# local imports
from utils import check_output_integrity
from api_common import (
    ModelSpec, DEFAULT_MAX_NEW_TOKENS, get_env_var, get_system_template,
    build_prompt_text, postprocess, resolve_max_new_tokens,
)

# Models supported. Rate limits as of January 2024 for the legacy models
MODELS = {
    "gpt-3.5-turbo-1106": ModelSpec(max_output_tokens=4096,  tokens_per_minute=60_000,  requests_per_minute=3_500, requests_per_day=10_000),
    "gpt-4-1106-preview": ModelSpec(max_output_tokens=4096,  tokens_per_minute=150_000, requests_per_minute=500,   requests_per_day=10_000),  # No longer reachable, deprecated
    "gpt-4-turbo-2024-04-09": ModelSpec(max_output_tokens=4096),
    "gpt-5.5-2026-04-23": ModelSpec(reasoning=True, max_output_tokens=128_000),
}

def is_reasoning_model(model: str) -> bool:
    return MODELS[model].reasoning


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", choices=list(MODELS), required=True, help="The model to use.")
    parser.add_argument("-p", "--prompts", type=str, required=True, help="Path to prompts json")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output json")
    parser.add_argument("--api-key", type=str, help="OpenAI API key. " +
        "If not provided, then uses environment variable OPENAI_API_KEY.")
    parser.add_argument("--openai-organization", type=str, help="OpenAI organization. " +
        "If not provided, then uses environment variable OPENAI_ORGANIZATION.")
    parser.add_argument("--max-requests", type=int, help="If provided, then only makes this many requests.")
    parser.add_argument("--max-tokens-per-second", help="Limit the rate of token generation.")
    parser.add_argument("--max-requests-per-second", help="Limit the rate of request generation.")
    parser.add_argument("--dry", action="store_true", help="If provided, then don't make any requests.")
    parser.add_argument("--overwrite", action="store_true", help="If provided, then overwrite outputs already in file.")
    parser.add_argument("--temperature", type=float, default=0.2, help="The temperature to use for sampling.")
    parser.add_argument("--top-p", type=float, default=0.95, help="The top p to use for sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
        help="The maximum number of tokens to generate. If unset, defaults per language: "
             f"{DEFAULT_MAX_NEW_TOKENS['cpp']} for C++, {DEFAULT_MAX_NEW_TOKENS['python']} for Python.")
    parser.add_argument("--num-samples-per-prompt", type=int, default=20, help="The number of samples to generate per prompt.")
    return parser.parse_args()


def get_max_tokens_per_second(model: str) -> Optional[float]:
    tpm = MODELS[model].tokens_per_minute
    return tpm / 60 if tpm is not None else None

def get_max_requests_per_second(model: str) -> Optional[float]:
    rpm = MODELS[model].requests_per_minute
    return rpm / 60 if rpm is not None else None

def get_max_requests_per_day(model: str) -> Optional[int]:
    return MODELS[model].requests_per_day

def main():
    args = get_args()

    # get the prompts
    with open(args.prompts, 'r') as prompts_json:
        prompts = json.load(prompts_json)

    # read in outputs
    if not args.overwrite and os.path.exists(args.output):
        with open(args.output, 'r') as output_json:
            outputs = json.load(output_json)

        # copy existing outputs into prompts
        copy_count = 0
        for prompt in prompts:
            for o in outputs:
                if o["prompt"] == prompt["prompt"] and \
                   o["name"] == prompt["name"] and \
                   o["parallelism_model"] == prompt["parallelism_model"] and \
                   "outputs" in o and \
                   len(o["outputs"]) == args.num_samples_per_prompt and \
                   o["temperature"] == args.temperature and \
                   o["top_p"] == args.top_p:
                    for col in ["temperature", "top_p", "do_sample", "max_new_tokens", "outputs"]:
                        prompt[col] = o[col]
                    copy_count += 1
                    break
        print(f"Copied {copy_count} existing outputs.")

    # get the keys
    api_key = args.api_key or get_env_var("OPENAI_API_KEY")
    organization = args.openai_organization or os.environ.get("OPENAI_ORGANIZATION")  # Not mandatory

    # create the client
    client = OpenAI(api_key=api_key, organization=organization)

    # generation metadata
    MAX_TOKENS_PER_SECOND = args.max_tokens_per_second or get_max_tokens_per_second(args.model)
    MAX_REQUESTS_PER_SECOND = args.max_requests_per_second or get_max_requests_per_second(args.model)
    MAX_REQUESTS = args.max_requests or get_max_requests_per_day(args.model)

    # generate outputs
    rate_state = {
        "request_counter": 0,
        "request_rate_counter": 0,
        "token_counter": 0,
        "token_rate_counter": 0,
        "token_timer": time.time(),
        "request_timer": time.time(),
    }

    def record_request(num_tokens: int) -> bool:
        """ Update counters, throttle if needed. Returns True if MAX_REQUESTS has been hit. """
        rate_state["request_counter"] += 1
        rate_state["request_rate_counter"] += 1
        rate_state["token_counter"] += num_tokens
        rate_state["token_rate_counter"] += num_tokens

        tokens_per_second = rate_state["token_rate_counter"] / (time.time() - rate_state["token_timer"])
        if MAX_TOKENS_PER_SECOND is not None and tokens_per_second > (MAX_TOKENS_PER_SECOND*0.9):
            sleep_time = 30
            print(f"Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
            rate_state["token_timer"] = time.time()
            rate_state["token_rate_counter"] = 0

        requests_per_second = rate_state["request_rate_counter"] / (time.time() - rate_state["request_timer"])
        if MAX_REQUESTS_PER_SECOND is not None and requests_per_second > (MAX_REQUESTS_PER_SECOND*0.95):
            sleep_time = 60
            print(f"Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
            rate_state["request_timer"] = time.time()
            rate_state["request_rate_counter"] = 0

        return MAX_REQUESTS is not None and rate_state["request_counter"] >= MAX_REQUESTS

    stop = False
    for prompt in tqdm(prompts, desc="Generating outputs"):
        # see if we can skip this
        if not args.overwrite and "outputs" in prompt:
            continue

        # get the prompt
        original_prompt = prompt["prompt"]
        system_template = get_system_template(prompt)
        prompt_text = build_prompt_text(prompt)

        # generate the outputs
        if args.dry:
            print("system", system_template)
            print("prompt", prompt_text)
            continue

        # resolve the token budget for this prompt (per-language default when unset)
        max_new_tokens = resolve_max_new_tokens(args.max_new_tokens, prompt, MODELS[args.model])

        # set metadata
        prompt["temperature"] = args.temperature
        prompt["top_p"] = args.top_p
        prompt["do_sample"] = True
        prompt["max_new_tokens"] = max_new_tokens

        messages = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": prompt_text}
        ]

        # generate the outputs
        if is_reasoning_model(args.model):
            # reasoning models fix temperature/top_p/n=1, so sample one completion at a time
            raw_outputs = []
            for _ in range(args.num_samples_per_prompt):
                # Use newer client.responses.create for better performance in reasoning models
                response = client.responses.create(
                    model=args.model,
                    input=messages,
                    max_output_tokens=max_new_tokens,
                    reasoning={"effort": "high"},
                    store=False
                )
                raw_outputs.append(response.output_text)
                stop = record_request(response.usage.total_tokens)
                if stop:
                    break
        else:
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=False,
                n=args.num_samples_per_prompt
            )
            raw_outputs = [c.message.content for c in completion.choices]
            stop = record_request(completion.usage.total_tokens)

        prompt["raw_outputs"] = raw_outputs
        prompt["outputs"] = [postprocess(original_prompt, o) for o in raw_outputs]

        # write intermediate outputs
        with open(args.output, 'w') as output_json:
            json.dump(prompts, output_json, indent=2)

        # check if we should stop
        if stop:
            print(f"Stopping after {rate_state['request_counter']} requests.")
            break

    # summary stats
    print(f"Submitted {rate_state['request_counter']} requests.")
    print(f"Used {rate_state['token_counter']} tokens.")

    # write outputs
    with open(args.output, 'w') as output_json:
        json.dump(prompts, output_json, indent=2)

    check_output_integrity(prompts)


if __name__ == "__main__":
    main()