# std imports
from abc import ABC, abstractmethod
import re

# tpl imports
import torch
from torch.utils.data import Dataset
from transformers import StoppingCriteria

def extract_pycompss_solution(code: str) -> str:
    """
    Extracts Python code starting from imports/defs and ending strictly 
    after the 'main' function returns.
    """
    lines = code.splitlines()
    
    # Find start (first import or function definition)
    start_index = 0
    start_pattern = re.compile(r'^\s*(import|from|def)\s+|^\s*@task')

    for i, line in enumerate(lines):
        if start_pattern.match(line):
            start_index = i
            break
            
    # Find 'def main'
    main_start_index = -1
    main_pattern = re.compile(r'^(\s*)def\s+main\s*\(')
    for i in range(start_index, len(lines)):
        match = main_pattern.match(lines[i])
        if match:
            main_start_index = i
            base_indent = len(match.group(1))
            break
            
    # Fallback: if no main found, return everything from start
    if main_start_index == -1:
        return "\n".join(lines[start_index:]).strip()

    # Find end (walk lines until indentation breaks)
    last_valid_index = main_start_index
    for i in range(main_start_index + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            continue
        current_indent = len(line) - len(line.lstrip())
        if stripped.startswith('#'):
            # Only include comments that are indented inside main
            if current_indent > base_indent:
                last_valid_index = i
            continue
        if current_indent <= base_indent:
            break
        last_valid_index = i
    end_index = last_valid_index + 1

    return "\n".join(lines[start_index:end_index]).strip()


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

    def format_prompt(self, prompt : str) -> str:
        function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
        prompt = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
        prompt = f"<|im_start|>system\nYou are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_instruct_output(output, prompt,"<|im_start|>assistant\n")

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
    elif model_name in ['deepseek-ai/deepseek-coder-6.7b-base', 'deepseek-ai/deepseek-coder-7b-base-v1.5']:
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
    elif 'Llama-3.1' in model_name and 'Instruct' in model_name:
        return Llama3InstructConfig(**kwargs)
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