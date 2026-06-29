""" Get the model outputs from Google's Gemini API.
    author: Daniel Nichols
    date: February 2024
"""
# std imports
from argparse import ArgumentParser
import json
import os
import time
from typing import Optional

# tpl imports
from tqdm import tqdm
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# local imports
from utils import check_output_integrity
from api_common import (
    ModelSpec, DEFAULT_MAX_NEW_TOKENS, get_env_var, get_system_template,
    build_prompt_text, postprocess, resolve_max_new_tokens,
)

# Models supported
MODELS = {
    "gemini-2.5-pro": ModelSpec(reasoning=True, max_output_tokens=65_536),
    "gemini-3.5-flash": ModelSpec(reasoning=True, max_output_tokens=65_536),
}

# Disable safety filtering: these are benign HPC coding prompts and we don't want refusals.
SAFETY_SETTINGS = [
    types.SafetySetting(category=category, threshold="BLOCK_NONE")
    for category in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]

# Cap retries per sample so a persistently-blocked/failing prompt can't loop forever.
MAX_RETRIES_PER_SAMPLE = 5

# Retries for transient API errors retrying
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_API_RETRIES = 10000
RETRY_BASE_DELAY = 5   # seconds; doubled each attempt
RETRY_MAX_DELAY = 1   # seconds; backoff cap


def is_retryable_api_error(error: genai_errors.APIError) -> bool:
    return getattr(error, "code", None) in RETRYABLE_STATUS_CODES


def generate_with_retries(client, model: str, contents, config):
    """Call generate_content, retrying transient API errors with exponential backoff.
    Non-retryable errors propagate immediately; a persistent transient error raises after
    MAX_API_RETRIES so the run stops loudly rather than recording bogus empty outputs."""
    for attempt in range(MAX_API_RETRIES):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except genai_errors.APIError as error:
            if not is_retryable_api_error(error) or attempt == MAX_API_RETRIES - 1:
                raise
            delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
            print(f"[retry] transient API error {error.code} {error.status}; "
                  f"attempt {attempt + 1}/{MAX_API_RETRIES}, sleeping {delay}s.")
            time.sleep(delay)


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", choices=list(MODELS), required=True, help="The model to use.")
    parser.add_argument("-p", "--prompts", type=str, required=True, help="Path to prompts json")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output json")
    parser.add_argument("--api-key", type=str, help="Google AI API key. " +
        "If not provided, then uses environment variable GOOGLE_API_KEY.")
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
    parser.add_argument("--num-samples-per-prompt", type=int, default=20, help="The number of samples to generate " +
        "per prompt.")
    return parser.parse_args()


def get_max_tokens_per_second(model: str) -> Optional[float]:
    tpm = MODELS[model].tokens_per_minute
    return tpm / 60 if tpm is not None else None

def get_max_requests_per_second(model: str) -> Optional[float]:
    rpm = MODELS[model].requests_per_minute
    return rpm / 60 if rpm is not None else None

def get_max_requests_per_day(model: str) -> Optional[int]:
    return MODELS[model].requests_per_day


def extract_text(response) -> Optional[str]:
    """Return the response text, or None if the generation was not usable (blocked, empty, or
    truncated before any output). STOP and MAX_TOKENS are accepted; anything else is treated as
    a transient failure worth retrying."""
    candidates = response.candidates or []
    if not candidates:
        return None
    finish_reason = candidates[0].finish_reason
    if finish_reason not in (types.FinishReason.STOP, types.FinishReason.MAX_TOKENS):
        return None
    return response.text or ""


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

    # create the client
    api_key = args.api_key or get_env_var("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

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
            sleep_time = 5
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

        if args.model == "gemini-3.5-flash":
            # Google recommends not setting temperature and top_k
            config = types.GenerateContentConfig(
                system_instruction=system_template,
                max_output_tokens=max_new_tokens,
                candidate_count=1,
                safety_settings=SAFETY_SETTINGS,
                thinking_config=types.ThinkingConfig(thinking_level="high")
            )
        else:
            config = types.GenerateContentConfig(
                system_instruction=system_template,
                temperature=args.temperature,
                top_p=args.top_p,
                max_output_tokens=max_new_tokens,
                candidate_count=1,
                safety_settings=SAFETY_SETTINGS
            )

        # Gemini returns one candidate per call, so sample one completion at a time.
        raw_outputs = []
        for _ in range(args.num_samples_per_prompt):
            text = None
            for attempt in range(MAX_RETRIES_PER_SAMPLE):
                response = generate_with_retries(client, args.model, prompt_text, config)
                text = extract_text(response)
                usage = getattr(response, "usage_metadata", None)
                stop = record_request(getattr(usage, "total_token_count", 0) or 0)
                if text is not None:
                    break
                print(f"Unusable completion (attempt {attempt + 1}/{MAX_RETRIES_PER_SAMPLE}); retrying.")
                time.sleep(5)
                if stop:
                    break
            raw_outputs.append(text or "")
            if stop:
                break

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
