"""
This module preprocesses the application files to clean the content. It's designed to get the descriptions of the readme
file and also get the parallel and sequential code as plain text. Codes are cleaned for better readability. The main
two tasks of the module are:
    - Extract the description from the README file by using regular expressions.
    - From the parallel code, obtain a clean version of both parallel and sequential code by:
        1. Transforming the code to tokens (Python tokenize).
        2. Filtering the unwanted parallel statements tokens: imports, API calls, decorators, and literals.
        3. Reassembling the tokens back to a clean code string.
        4. Passing both codes to black (https://black.readthedocs.io/) to format them.

"""
from typing import Generator, Iterator, Callable, Iterable

from utils import compose, Graph, read_file, log_message
import io
import tokenize
import black
import re
import logging
from collections import namedtuple
import pathlib

ENCODING = 'utf-8'

# Imports to be removed from parallel code
PYCOMPSS_PACKAGES = ["pycompss"]

# API calls to be removed from parallel code. An API call is a function call that is used to interact with the PyCOMPSs
# runtime, with the return being redundant.
PYCOMPSS_API_CALLS = ["compss_barrier"]

# Decorators to be removed from parallel code. These decorators are used to mark functions for parallel execution.
PYCOMPSS_DECORATORS = ["task", "constraint", "binary", "mpi"]

# Literals to be removed from parallel code. They will be removed as they are to get the sequential code.
# Example: x = compss_wait_on(x) will be transformed to x = (x)
PYCOMPSS_LITERALS = ["compss_wait_on"]

logger = logging.getLogger(__name__)

Token = namedtuple('Token', ['type', 'string'])

@log_message(logger.info, "Removed imports from the code.")
def remove_imports(tokens: Iterator[Token], packages: list[str]) -> Generator[Token, None, None]:
    """
    Remove each import line that contains any of the packages in the list.
    This generator removes the corresponding import lines by not yielding them.
    Args:
        tokens (iterable): An iterable of Token objects representing the code.
        packages (list): A list of package names to be removed from import lines.
    Yields:
        Token: Tokens that are not part of the import lines containing the specified packages.
    """

    # Flag to be used when an import/from statement is found, this will be set to True
    potential_removing_now = False

    # List to be used to store the tokens of the import line (even if it is a false positive)
    import_line = []

    # Flag to be set to True when an indent is found, it helps to determine if a pass statement should be added
    previous_indent = False

    # Flag to be set to True when performed a removal of an import line, it helps to determine if a pass statement
    # should be added
    just_removed = False

    for token in tokens:
        if potential_removing_now:
            if token.type == tokenize.NEWLINE: # End of the import line
                if any(map(lambda x: x.string in packages, import_line)):
                    # Some of the packages names are in the import line, discard the import line by not yielding it
                    just_removed = True
                else:
                    # None of the packages names are in the import line (false positive)
                    # Yield the tokens of the import line
                    for t in import_line:
                        yield t
                    yield token # Yield the newline token
                    just_removed, previous_indent = False, False
                potential_removing_now, import_line = False, []
            else: # Accumulate tokens in the import line
                import_line.append(token)
        else:
            if token.string in ("from", "import"): # Start of an import line
                import_line.append(token)
                potential_removing_now = True
            elif token.type == tokenize.INDENT: # Detected an indent
                yield token
                previous_indent, just_removed = True, False
            elif token.type == tokenize.DEDENT:
                if previous_indent and just_removed:
                    # The previous line was removed resulting in an empty indent, we need to add a pass statement
                    # to keep the code structure valid
                    # This is an edge case, for example:
                    # try:
                    #     from pycompss.api import compss_wait_on
                    # except ImportError:
                    #     print("PyCOMPSs is not available")
                    yield Token(tokenize.NAME, "pass")
                    yield Token(tokenize.NEWLINE, '\n')
                yield token
                previous_indent, just_removed = False, False
            else:
                yield token
                previous_indent, just_removed = False, False

@log_message(logger.info, "Removed API calls from the code.")
def remove_api_calls(tokens: Iterator[Token], api_calls: list[str]) -> Generator[Token, None, None]:
    """
    Remove each API call line included in api_calls list.
    This generator removes the corresponding API call lines by not yielding them.
    Args:
        tokens (iterable): An iterable of Token objects representing the code.
        api_calls (list): A list of package names to be removed from import lines.
    Yields:
        Token: Tokens that are not part of the specified API calls.
    """
    # Flag to be used when an API call is found, this will be set to True
    removing_now: bool = False

    # Counter to keep track of the number of open parentheses, used to determine when to stop removing
    open_parenthesis: int = 0

    # Flag to be set to True when an indent is found, it helps to determine if a pass statement should be added
    previous_indent: bool = False

    # Flag to be set to True when performed a removal of an API call, it helps to determine if a pass statement should
    # be added
    just_removed: bool = False

    for token in tokens:
        if removing_now:
            if token.string == '(': # New open parenthesis found while parsing the API call
                open_parenthesis += 1
            elif token.string == ')': # Closing parenthesis found while parsing the API call
                if open_parenthesis == 1:
                    # The API call is finished, it gets discarded as it has not been yielded
                    removing_now, just_removed = False, True
                open_parenthesis -= 1
        else:
            if token.string in api_calls: # An API call is found
                removing_now = True
            elif token.type == tokenize.INDENT: # Detected an indent
                yield token
                previous_indent, just_removed = True, False
            elif token.type == tokenize.DEDENT:
                if previous_indent and just_removed:
                    # The previous line was removed resulting in an empty indent, we need to add a pass statement
                    # to keep the code structure valid
                    # This is an edge case, for example:
                    # try:
                    #     compss_barrier()
                    # except ImportError:
                    #     print("compss_barrier() is not available")
                    yield Token(tokenize.NAME, "pass")
                    yield Token(tokenize.NEWLINE, '\n')
                yield token
                previous_indent, just_removed = False, False
            else:
                yield token
                previous_indent, just_removed = False, False

@log_message(logger.info, "Removed decorators from the code.")
def remove_decorators(tokens: Iterator[Token], decorators: list[str]) -> Generator[Token, None, None]:
    """
    Remove each decorator included in decorators list.
    This generator removes the corresponding decorators by not yielding them.
    Args:
        tokens (iterable): An iterable of Token objects representing the code.
        decorators (list): A list of package names to be removed from import lines.
    Yields:
        Token: Tokens that are not part of the specified decorators.
    """

    # Flag to be used when a decorator is found, this will be set to True
    removing_now: bool = False

    # Counter to keep track of the number of open parentheses, used to determine when to stop removing
    open_parenthesis: int = 0

    # Variable to keep track of the previous '@' token, used to yield it if the next token is not a decorator
    # from the decorators list. Note that '@' is a token different from the decorator name, so we need to keep track of
    # it to yield it when needed (false positive).
    previous_at: Token | None = None
    for token in tokens:
        if removing_now:
            if token.string == '(': # New open parenthesis found while parsing a decorator
                open_parenthesis += 1
            elif token.string == ')': # Closing parenthesis found while parsing the API call
                if open_parenthesis == 1:
                    # The decorator is finished, it gets discarded as it has not been yielded
                    removing_now = False
                    previous_at = None
                open_parenthesis -= 1
        else:
            if token.string == '@':
                previous_at = token
            else:
                if previous_at is None: # Common case, the token is not a decorator
                    yield token
                else:
                    if token.string in decorators: # A decorator is found, and it is in the decorators list
                        removing_now = True
                    else:
                        # False positive, the decorator is not in the decorators list, yield the previous '@' token
                        # and the current token
                        yield previous_at
                        yield token
                        previous_at = None


def get_merged_code(paths: list[str]) -> str:
    """
    Merge the code from all files in the given paths, removing import statements that reference other files.
    This function creates a directed graph to represent dependencies between files, where an edge from file A to file B
    indicates that file B imports file A (so file A should go first). The function then performs a topological sort on
    the graph to determine the order in which files should be processed, ensuring that all dependencies are resolved
    before unifying into a single file.
    Args:
        paths (list[str]): List of paths to the application files.
    Returns:
        str: Merged code from all files, with relative import statements removed.
    """
    graph = Graph(len(paths))
    contents = list(map(read_file, paths))
    names = list(map(lambda x: pathlib.Path(x).stem, paths))
    for i in range(len(contents)):
        for j in range(len(names)):
            if i != j:
                pattern = rf'\s*(from|import)\s+{re.escape(names[j])}.*'
                match = re.search(pattern, contents[i])
                if match:
                    contents[i] = contents[i][:match.start()] + contents[i][match.end():]
                    graph.add_edge(j, i)

    topographical_order = graph.topological_sort()
    merged_code = "\n\n".join(contents[i] for i in topographical_order)

    logger.info("Successfully merged the code from all files.")
    return merged_code

def parse_description(readme_path: str | None) -> str:
    """
    Parse the README file to extract the description of the application.
    The description is expected to be in a specific format:

    == Description ==
    The description is extracted from the README file using regular expressions.

    == Next Section ==

    Args:
        readme_path (str): Path to the README file.
    Returns:
        str: The extracted description from the README file.
    """
    if readme_path is None:
        return ""

    with open(readme_path, 'r') as f:
        content = f.read()

    # Assuming the description is in a specific format, e.g., starting with "Description:"
    match = re.search(r'==\s+Description\s+==\s*(.*?)(?=\n==\s*\w|\Z)', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError(f"Description not found in {readme_path}. Please ensure the README is formatted correctly.")

def clean_application(paths: list[str]) -> tuple[str, str]:
    """
    Clean the application files to get the parallel and sequential code as plain text.
    Args:
        paths (list[str]): List of paths to the application files.
    Returns:
        tuple: A tuple containing the parallel and sequential clean code.
    """

    # Phase 1: Get the code from all files, convert to tokens, and remove comments. Both parallel and sequential
    # code pipelines undergo through this phase.
    phase1_func: Callable[[list[str]], list[Token]] = compose(
        list, # Converts the filter object to a list
        lambda x: filter(lambda y: y.type != tokenize.COMMENT, x),  # Remove comments
        lambda x: map(lambda y: Token(y.type, y.string), x),  # Convert to namedtuple Token
        lambda x: tokenize.tokenize(io.BytesIO(x.encode(ENCODING)).readline),  # Get tokenize Tokens
        get_merged_code, # Merge the code from all files
    )

    # Sequentialization phase: This phase is exclusive to the sequential code pipeline, it removes the decorators, API
    # calls, imports, and literals from the code.
    get_sequential_code_func: Callable[[Iterable[Token]], Iterable[Token]] = compose(
        lambda x: filter(lambda y: y.string not in PYCOMPSS_LITERALS, x),  # Remove literals from the code
        lambda x: remove_decorators(x, PYCOMPSS_DECORATORS),  # Remove decorators from the code
        lambda x: remove_api_calls(x, PYCOMPSS_API_CALLS),  # Remove API calls from the code
        lambda x: remove_imports(x, PYCOMPSS_PACKAGES),  # Remove imports from the code
    )

    # Phase 2: Convert the tokens to code string, and format it with black.
    # This phase is applied to both parallel and sequential code pipelines.
    phase2_func: Callable[[Iterable[Token]], str] = compose(
        lambda x: black.format_str(x, mode=black.FileMode()), # Format the code with black
        lambda x: x.decode(ENCODING), # Convert bytes to string
        lambda x: tokenize.untokenize(x), # Convert tokens to bytes
    )


    phase1_code: list[Token] = phase1_func(paths)
    sequential_code: str = phase2_func(get_sequential_code_func(phase1_code))
    parallel_code: str = phase2_func(phase1_code)

    return parallel_code, sequential_code