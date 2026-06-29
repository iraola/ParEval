""" Shared helpers for the API-based generation scripts (generate-openai.py, generate-gemini.py).

Holds the provider-agnostic pieces: prompt templates, language-based prompt dispatch, the
per-model metadata dataclass, the token-budget resolver, and output post-processing. Each
provider script keeps its own MODELS registry and its own request/response code, since those
are provider-specific.
"""
# std imports
from dataclasses import dataclass
import os
from typing import Optional

# local imports
from utils import clean_instruct_output, get_function_name

# Prompt templates: one pair per language
CPP_SYSTEM_TEMPLATE = """You are a helpful coding assistant.
You are helping a programmer write a C++ function. Write the body of the function and put it in a markdown code block.
Do not write any other code or explanations.
"""

CPP_PROMPT_TEMPLATE = """Complete the C++ function {function_name}. Only write the body of the function {function_name}.

```cpp
{prompt}
```
"""

PYTHON_SYSTEM_TEMPLATE = """You are a helpful coding assistant.
You are helping a programmer write a Python module. Write the complete module and put it in a markdown code block.
Do not write any other code or explanations.
"""

PYTHON_PROMPT_TEMPLATE = """Implement the Python module described by the imports and specification below.

```python
{prompt}
```
"""

# Per-language token budget used when --max-new-tokens is not given
DEFAULT_MAX_NEW_TOKENS = {"cpp": 1024, "python": 32768}


@dataclass(frozen=True)
class ModelSpec:
    """
    Per-model API metadata.
    - `reasoning` models may fix sampling params (e.g. OpenAI reasoning
    models fix temperature/top_p/n=1) and spend hidden reasoning tokens.
    - `max_output_tokens` is the hard cap the API enforces on output length;
    resolved token budgets are clamped to it.
    - Rate limits are None when unknown (no throttling applied).
    """
    reasoning: bool = False
    max_output_tokens: int = 4096
    tokens_per_minute: Optional[int] = None
    requests_per_minute: Optional[int] = None
    requests_per_day: Optional[int] = None


def get_env_var(name: str) -> str:
    """ Get an environment variable. """
    if name not in os.environ:
        raise ValueError(f"Environment variable {name} not set.")
    return os.environ[name]


def is_python(prompt_entry: dict) -> bool:
    return prompt_entry["language"] == "python"


def get_system_template(prompt_entry: dict) -> str:
    return PYTHON_SYSTEM_TEMPLATE if is_python(prompt_entry) else CPP_SYSTEM_TEMPLATE


def build_prompt_text(prompt_entry: dict) -> str:
    original_prompt = prompt_entry["prompt"]
    if is_python(prompt_entry):
        return PYTHON_PROMPT_TEMPLATE.format(prompt=original_prompt)
    function_name = get_function_name(original_prompt, prompt_entry["parallelism_model"])
    return CPP_PROMPT_TEMPLATE.format(prompt=original_prompt, function_name=function_name)


def postprocess(prompt: str, output: str) -> str:
    """ Post-process the output, reusing the same code-extraction logic as the local inference pipeline. """
    return clean_instruct_output(output, prompt, response_tag="")


def resolve_max_new_tokens(requested: Optional[int], prompt_entry: dict, spec: ModelSpec) -> int:
    """Resolve the token budget for one prompt. `requested` is the --max-new-tokens flag value.
    The result is passed to the model's hard output cap."""
    if requested is None:
        requested = DEFAULT_MAX_NEW_TOKENS["python"] if is_python(prompt_entry) else DEFAULT_MAX_NEW_TOKENS["cpp"]
    if requested > spec.max_output_tokens:
        print(f"[warn] requested {requested} tokens exceeds model cap of {spec.max_output_tokens}; clamping...")
        return spec.max_output_tokens
    return requested
