# std imports
from abc import ABC, abstractmethod
import re
from typing import List

# tpl imports
import torch
from torch.utils.data import Dataset
from transformers import StoppingCriteria

def extract_pycompss_solution(code: str) -> str:
    """
    Extracts Python code starting from imports/defs and ending before any
    non-indented statement that is not an import, decorator, or function
    definition (e.g. `if __name__ == '__main__'` blocks).
    """
    lines = code.splitlines()

    # Build the set of line indices that fall inside triple-quoted strings so
    # that "from x import y" inside a docstring is not mistaken for a real import.
    triple_lines: set = set()
    in_triple = False
    triple_seq = None
    for i, line in enumerate(lines):
        if in_triple:
            triple_lines.add(i)
            if triple_seq in line:
                in_triple = False
                triple_seq = None
        else:
            for tq in ('"""', "'''"):
                if tq in line:
                    if line.count(tq) % 2 == 1:  # opens but does not close on this line
                        in_triple = True
                        triple_seq = tq
                        triple_lines.add(i)
                    break

    # Find start (first import or function definition) outside docstrings
    start_index = 0
    start_pattern = re.compile(r'^\s*(import|from|def)\s+|^\s*@task')

    for i, line in enumerate(lines):
        if i in triple_lines:
            continue
        if start_pattern.match(line):
            start_index = i
            break

    # Walk lines and include imports, decorators and functions
    top_level_ok = re.compile(r'^\s*(import|from|@|def)\s*')
    last_valid_index = start_index

    for i in range(start_index, len(lines)):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            continue

        # Lines inside triple-quotes are kept but not used as a stopping criterion
        if i in triple_lines:
            last_valid_index = i
            continue

        current_indent = len(line) - len(line.lstrip())
        if current_indent > 0:
            # Inside a function body: always include
            last_valid_index = i
            continue

        # Top-level line
        if stripped.startswith('#') or top_level_ok.match(line):
            last_valid_index = i
            continue

        # Module-level constant/variable assignment (e.g. NUM_BINS = 10).
        # Bare function calls like init() don't match (no '=')
        if re.match(r'^[A-Za-z_]\w*\s*=[^=]', stripped):
            last_valid_index = i
            continue

        # Non-indented, non-function code (e.g. if __name__ == '__main__'): stop
        break

    return "\n".join(lines[start_index:last_valid_index + 1]).strip()


def clean_output(output: str, prompt: str) -> str:
    """ 
    Removes `prompt` from the beginning of `output`.
    For C++/CUDA: Truncates at the matching closing brace.
    For PyCOMPSs: Truncates after the 'main' function ends.
    """
    prompt_loc = output.find(prompt)
    if prompt_loc == -1:
        raw_output = output
    else:
        raw_output = output[prompt_loc + len(prompt):].strip()

    if "pycompss" in prompt.lower():
        return extract_pycompss_solution(raw_output)

    # Prepend '{' to simulate a complete function body and reuse brace-matching logic
    cpp_output = '{' + raw_output

    stack = []
    index = 0
    while index < len(cpp_output):
        token = cpp_output[index]
        if token == '{':
            stack.append(token)
        elif token == '}':
            stack.pop()
            if len(stack) == 0:
                break
        index += 1

    # Strip the artificial opening brace before returning
    return cpp_output[1:index+1]

GPU_FUNCTION_NAME_PATTERN = re.compile(r"__global__ void ([a-zA-Z0-9_]+)\(")
CPU_FUNCTION_NAME_PATTERN = re.compile(r"\s*[a-zA-Z_]+ ([a-zA-Z0-9_]+)\(")
def get_function_name(prompt: str, execution_model: str) -> str:
    if execution_model in ['cuda', 'hip']:
        match = GPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
    else:
        match = CPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
    if match is None:
        raise ValueError(f"Could not find function name in prompt: {prompt}")
    return match.group(1)


def find_matching_brace_index(code: str, open_brace_index: int) -> int:
    """Finds the index of the closing brace that matches the opening brace at the given index."""

    brace_count = 1
    for i in range(open_brace_index + 1, len(code)):
        if code[i] == "{":
            brace_count += 1
        elif code[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return i

    raise ValueError("Unmatched opening brace")


def clean_instruct_output(output: str, prompt: str, response_tag: str) -> str:
    """ Clean LLM output to find code solution. """

    # Strip thinking tokens emitted by reasoning models (e.g. DeepSeek-R1, Qwen3)
    think_end = output.rfind('</think>')
    if think_end != -1:
        output = output[think_end + len('</think>'):].strip()
    elif '<think>' in output:
        # Thinking block was not closed (generation truncated mid-thought) — no usable output
        return ''

    prompt_loc = output.find(response_tag)
    if prompt_loc != -1:
        output = output[prompt_loc + len(response_tag):].strip()

    # Extract code blocks (```python, ```c++, or plain ```)
    code_blocks = re.findall(r"```(?:\w*)\n(.*?)\n```", output, flags=re.DOTALL)

    if len(code_blocks) > 0:
        raw_code = code_blocks[0]
    else:
        raw_code = output.strip()
        if raw_code.startswith("```"): raw_code = raw_code[3:]
        if raw_code.endswith("```"): raw_code = raw_code[:-3]

    if "pycompss" in prompt.lower():
        # Prefer blocks that contain Python code markers
        python_blocks = [b for b in code_blocks if re.search(r'(@task|^\s*import\s|^\s*from\s|^\s*def\s)', b, flags=re.MULTILINE)]
        if python_blocks:
            return extract_pycompss_solution(python_blocks[0])
        # Fallback: look for an unclosed ```python fence (truncated generation)
        unclosed = re.search(r"```(?:python)?\n(.*)", output, flags=re.DOTALL)
        if unclosed:
            return extract_pycompss_solution(unclosed.group(1))
        return extract_pycompss_solution(raw_code)

    try:
        sub_prompt = prompt.rstrip().removesuffix(response_tag).rstrip()
        if "```" in sub_prompt:
             sub_prompt = sub_prompt.split("```")[-1]

        function_name = get_function_name(sub_prompt, "cuda" if "__global__" in sub_prompt else "serial")

        selected_block = raw_code # Default
        if len(code_blocks) > 0:
            prioritized_blocks = [block for block in code_blocks if function_name in block]
            if prioritized_blocks:
                selected_block = prioritized_blocks[0]
            else:
                selected_block = code_blocks[0]

        if function_name in selected_block:
            function_start_index = selected_block.index(function_name)
            open_brace_index = selected_block.find("{", function_start_index)

            if open_brace_index != -1:
                try:
                    close_brace_index = find_matching_brace_index(selected_block, open_brace_index)
                    return (selected_block[open_brace_index + 1 : close_brace_index] + "}").strip()
                except ValueError:
                    pass

        return selected_block

    except ValueError:
        return raw_code


class InferenceConfig(ABC):

    def __init__(self, prompted : bool = False):
        self.prompted = prompted
    
    @abstractmethod
    def get_dtype(self):
        pass
    
    @abstractmethod
    def init_padding(self, tokenizer):
        pass

    @abstractmethod
    def get_pad_token_id(self, tokenizer) -> int:
        pass

    @abstractmethod
    def get_eos_token_id(self, tokenizer) -> int:
        pass

    @abstractmethod
    def trust_remote_code(self) -> bool:
        pass

    @abstractmethod
    def format_prompt(self, prompt : str) -> str:
        pass

    @abstractmethod
    def clean_output(self, output: str, prompt: str) -> str:
        pass


class StarCoderConfig(InferenceConfig):

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return None

    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"<filename>solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)

class CodeLlamaConfig(InferenceConfig):

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models
        pass

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)

class Llama3InstructConfig(InferenceConfig):
    """ Configuration for Llama 3 and 3.1 Instruct models """

    PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if "pycompss" in prompt.lower():
            return self.PROMPT_TEMPLATE.format(instruction=prompt.strip())

        function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
        instruct_prompt = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
        return self.PROMPT_TEMPLATE.format(instruction=instruct_prompt)

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_instruct_output(output, prompt, "<|start_header_id|>assistant<|end_header_id|>\n\n")


class PolyCoderConfig(InferenceConfig):

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()
    
    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)


class PhindConfig(InferenceConfig):

    def __init__(self, prompted: bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)


class ReplitConfig(InferenceConfig):

    def __init__(self, prompted: bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        pass
        #tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        #tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return None
        #return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return None
        #return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return True

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)


class MagicoderConfig(InferenceConfig):

    PROMPT_TEMPLATE = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models
        pass

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
            prompt = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
            return self.PROMPT_TEMPLATE.format(instruction=prompt)
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_instruct_output(output, prompt, "@@ Response")


class DeepSeekBaseConfig(InferenceConfig):

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)


class InstructConfig(InferenceConfig):
    def __init__(self, prompted : bool = False, instruction_tag : str = "### Instruction", response_tag : str = "### Response"):
        super().__init__(prompted=prompted)
        self.instruction_tag = instruction_tag
        self.response_tag = response_tag

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt: str) -> str:
        if "pycompss" in prompt.lower():
            formatted = f"{self.instruction_tag}\n{prompt.strip()}\n{self.response_tag}\n"
            return formatted

        function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
        prompt = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
        prompt = f"{self.instruction_tag}\n{prompt}\n{self.response_tag}\n"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_instruct_output(output, prompt, self.response_tag)

class QwenConfig(InferenceConfig):
    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return None
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)

class ChatMLConfig(InferenceConfig):
    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt: str) -> str:
        if "pycompss" in prompt.lower():
            instruction = prompt.strip()
        else:
            function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
            instruction = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
        return f"<|im_start|>system\nYou are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_instruct_output(output, prompt,"<|im_start|>assistant\n")

class MistralInstructConfig(InferenceConfig):
    """Configuration for Mistral instruct models (Codestral, Mistral Small, etc.)"""

    def __init__(self, prompted: bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt: str) -> str:
        if "pycompss" in prompt.lower():
            instruction = prompt.strip()
        else:
            function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
            instruction = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
        return f"<s>[INST] {instruction} [/INST]"

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_instruct_output(output, prompt, "[/INST]")


class DeepSeekR1Config(InferenceConfig):
    """Configuration for DeepSeek-R1 distilled models (Llama and Qwen variants).
    These models use DeepSeek's own User/Assistant tokens regardless of backbone architecture,
    and always emit a <think>...</think> block before the answer."""

    def __init__(self, prompted: bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt: str) -> str:
        if "pycompss" in prompt.lower():
            instruction = prompt.strip()
        else:
            function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
            instruction = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
        return f"<｜User｜>{instruction}<｜Assistant｜>"

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_instruct_output(output, prompt, "<｜Assistant｜>")



class HarmonyConfig(InferenceConfig):
    """Configuration for OpenAI open-weight models using the Harmony chat format (e.g. gpt-oss-20b, gpt-oss-120b)."""

    def __init__(self, prompted: bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt: str) -> str:
        if "pycompss" in prompt.lower():
            instruction = prompt.strip()
        else:
            function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
            instruction = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
        return f"<|start|>user<|message|>{instruction}<|end|>\n<|start|>assistant<|message|>"

    def clean_output(self, output: str, prompt: str) -> str:
        # gpt-oss models interleave internal reasoning between the assistant tag and the final answer.
        # The actual response starts after the literal token 'assistantfinal'.
        assistant_tag = "<|start|>assistant<|message|>"
        assistant_idx = output.find(assistant_tag)
        if assistant_idx != -1:
            assistant_content = output[assistant_idx + len(assistant_tag):]
            final_idx = assistant_content.find('assistantfinal')
            if final_idx != -1:
                output = assistant_tag + assistant_content[final_idx + len('assistantfinal'):]
        return clean_instruct_output(output, prompt, assistant_tag)


class GLM4Config(InferenceConfig):
    """Configuration for ZhipuAI GLM-4 models (zai-org/GLM-*)."""

    def __init__(self, prompted: bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt: str) -> str:
        if "pycompss" in prompt.lower():
            instruction = prompt.strip()
        else:
            function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
            instruction = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
        return f"[gMASK]<sop><|user|>\n{instruction}\n<|assistant|>\n"

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_instruct_output(output, prompt, "<|assistant|>\n")


SHORT_OUTPUT_THRESHOLD = 30  # chars; cleaned outputs shorter than this are flagged


def check_output_integrity(responses: List[dict], extra_counters: dict = None) -> dict:
    """Check generated outputs for integrity issues and print a summary report.

    Checks performed on raw_outputs / outputs:
    - truncated_think   : <think> opened but never closed (generation ran out of tokens)
    - unclosed_fence    : odd number of ``` markers (code block not closed)
    - empty_output      : cleaned output is empty string
    - short_output      : cleaned output shorter than SHORT_OUTPUT_THRESHOLD chars
    - no_code_structure : pycompss output lacks @task/def; C++ output lacks {
    - identical_samples : all N samples for a prompt are the same (degenerate generation)

    extra_counters: optional dict of additional pre-computed counters to include
    in the report (e.g. {'truncated_by_length': 12} from vLLM finish_reason).
    """
    stats = {
        'truncated_think': 0,
        'unclosed_fence': 0,
        'empty_output': 0,
        'short_output': 0,
        'no_code_structure': 0,
        'identical_samples': 0,
    }

    for r in responses:
        outputs = r.get('outputs', [])
        raw_outputs = r.get('raw_outputs', [])
        is_pycompss = 'pycompss' in r.get('prompt', '').lower()

        for raw, out in zip(raw_outputs, outputs):
            if '<think>' in raw and '</think>' not in raw:
                stats['truncated_think'] += 1
            if raw.count('```') % 2 != 0:
                stats['unclosed_fence'] += 1
            stripped = out.strip()
            if not stripped:
                stats['empty_output'] += 1
            elif len(stripped) < SHORT_OUTPUT_THRESHOLD:
                stats['short_output'] += 1
            if stripped:
                if is_pycompss and '@task' not in out and 'def ' not in out:
                    stats['no_code_structure'] += 1
                elif not is_pycompss and '{' not in out:
                    stats['no_code_structure'] += 1

        if len(outputs) > 1 and len(set(outputs)) == 1:
            stats['identical_samples'] += 1

    total_entries = len(responses)
    total_samples = sum(len(r.get('outputs', [])) for r in responses)

    print(f"\n{'='*40}")
    print(f"Output Integrity Report")
    print(f"  Entries: {total_entries} | Samples: {total_samples}")
    if extra_counters:
        for key, val in extra_counters.items():
            label = key.replace('_', ' ').capitalize()
            flag = ' !!!' if val > 0 else ''
            print(f"  {label}: {val}{flag}")
    issues = {k: v for k, v in stats.items() if v > 0}
    if not issues:
        print("  No issues detected.")
    else:
        for key, val in issues.items():
            label = key.replace('_', ' ').capitalize()
            print(f"  {label}: {val} !!!")
    print(f"{'='*40}\n")

    return {**stats, **(extra_counters or {})}


def get_inference_config(model_name : str, **kwargs) -> InferenceConfig:
    if model_name == "bigcode/starcoderbase":
        return StarCoderConfig(**kwargs)
    elif model_name in ["bigcode/starcoder2-3b", "bigcode/starcoder2-7b", "bigcode/starcoder2-15b"]:
        return StarCoderConfig(**kwargs)
    elif model_name.startswith("codellama/CodeLlama-") and 'Instruct' not in model_name:
        return CodeLlamaConfig(**kwargs)
    elif model_name == "NinedayWang/PolyCoder-2.7B":
        return PolyCoderConfig(**kwargs)
    elif model_name == 'Phind/Phind-CodeLlama-34B-v2':
        return PhindConfig(**kwargs)
    elif model_name == 'replit/replit-code-v1_5-3b':
        return ReplitConfig(**kwargs)
    elif model_name.startswith('ise-uiuc/Magicoder'):
        return MagicoderConfig(**kwargs)
    elif model_name.startswith('deepseek-ai/') and 'Instruct' not in model_name and 'R1' not in model_name:
        return DeepSeekBaseConfig(**kwargs)
    elif model_name.startswith('hpcgroup/hpc-coder-v2'):
        return InstructConfig(instruction_tag='Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:', response_tag='### Response:', **kwargs)
    elif model_name.startswith('hpcgroup/rlpf'):
        return InstructConfig(instruction_tag='### Instruction', response_tag='### Response', **kwargs)
    elif model_name.startswith('Qwen/Qwen2.5') and 'Instruct' in model_name:
        return ChatMLConfig(**kwargs)
    elif model_name.startswith('Qwen/Qwen3'):
        return ChatMLConfig(**kwargs)
    elif model_name.startswith('Qwen/Qwen2.5'):
        return QwenConfig(**kwargs)
    elif model_name == 'deepseek-ai/deepseek-coder-6.7b-instruct':
        return InstructConfig(instruction_tag='### Instruction:', response_tag='### Response:', **kwargs)
    elif ('Llama-3.1' in model_name or 'Llama-3.3' in model_name) and 'Instruct' in model_name:
        return Llama3InstructConfig(**kwargs)
    elif 'DeepSeek-R1' in model_name:
        # Covers DeepSeek-R1, DeepSeek-R1-0528, and all DeepSeek-R1-Distill-* variants
        return DeepSeekR1Config(**kwargs)
    elif model_name == 'deepseek-ai/DeepSeek-Coder-V2-Instruct':
        return InstructConfig(instruction_tag='User:', response_tag='Assistant:', **kwargs)
    elif model_name.startswith('mistralai/Codestral') \
            or ('Mistral-Small' in model_name and 'Instruct' in model_name) \
            or ('Mixtral' in model_name and 'Instruct' in model_name) \
            or 'Magistral' in model_name:
        return MistralInstructConfig(**kwargs)
    elif model_name.startswith('openai/gpt-oss'):
        return HarmonyConfig(**kwargs)
    # Qwen3 loaded from a 3-level local path (org prefix stripped by path parser)
    elif model_name.startswith('Qwen3/'):
        return ChatMLConfig(**kwargs)
    # Qwen2.5 loaded from a 3-level local path (org prefix stripped by path parser)
    elif model_name.startswith('Qwen2.5/') and 'Instruct' in model_name:
        return ChatMLConfig(**kwargs)
    elif model_name.startswith('Qwen2.5/'):
        return QwenConfig(**kwargs)
    elif model_name.startswith('microsoft/bitnet'):
        # Base completion model — no chat template
        return StarCoderConfig(**kwargs)
    elif model_name.startswith('zai-org/GLM'):
        return GLM4Config(**kwargs)
    elif model_name.startswith('moonshotai/Kimi'):
        return ChatMLConfig(**kwargs)
    elif model_name.startswith('ByteDance-Seed/Seed-OSS'):
        # NOTE: chat template not officially documented; ChatML assumed — verify if wrong
        return ChatMLConfig(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class PromptDataset(Dataset):
    ''' PyTorch dataset that simply wraps a list of strings. They do not have to have the same length.
    '''

    def __init__(self, prompts):
        super().__init__()
        self.prompts_ = prompts
    
    def __len__(self):
        return len(self.prompts_)
    
    def __getitem__(self, idx):
        return self.prompts_[idx]

    def __iter__(self):
        return iter(self.prompts_)


def has_balanced_brackets(text : str, left_bracket : str = '{', right_bracket : str = '}') -> bool:
    ''' Check if string has balanced brackets.
        modified from: https://stackoverflow.com/a/38834249/3769237

        Arguments:
            text: string to check for balanced brackets in.
            left_bracket: left bracket to balance
            right_bracket: right bracket to balance

        Returns:
            true if left_bracket and right_bracket are balanced
    '''
    stack = []
    balanced = True
    index = 0
    while index < len(text) and balanced:
        token = text[index]
        if token == left_bracket:
            stack.append(token)
        elif token == right_bracket:
            if len(stack) == 0:
                balanced = False
            else:
                stack.pop()

        index += 1

    return balanced and len(stack) == 0


class BalancedBracketsCriteria(StoppingCriteria):
    ''' extension of transformers' text-generation stopping criteria.
        Stops either when function is complete (i.e. { and } are balanced) or when max_length is surpassed, whichever
        happens first. 

        _Note:_ This is a slow stopping criteria, but it's much faster than continually running model inference when 
        it does not need to be run anymore.
    '''

    def __init__(self, max_length : int, tokenizer, left_bracket : str = '{', right_bracket : str = '}'):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.left_bracket = left_bracket
        self.right_bracket = right_bracket
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] > self.max_length:
            # already too long, early stop
            return True

        # return true if {} are balanced i.e. the function is complete
        return all(
            has_balanced_brackets(
                self.tokenizer.decode(t), 
                left_bracket=self.left_bracket, 
                right_bracket=self.right_bracket
            ) for t in input_ids)