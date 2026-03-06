"""
Utility functions for various tasks including:
    - file searching
    - function composition
    - directory generation
    - graph operations
    - logging decorator
"""
import functools
import inspect
import pathlib
from functools import reduce
import logging
import random
import time
from collections import defaultdict
from typing import TypeVar, Callable, Any, Generator, cast, Iterable

ADJECTIVES = [
    "brave", "calm", "eager", "fancy", "gentle", "happy", "jolly", "kind",
    "lucky", "mighty", "nice", "proud", "quick", "silly", "witty", "zany"
]

ANIMALS = [
    "lion", "tiger", "bear", "eagle", "panda", "shark", "fox", "dragon",
    "wolf", "rhino", "zebra", "whale", "otter", "falcon", "leopard", "koala"
]

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

def find_files(root: str, ignore_paths: list[str] | None = None) -> dict[str, tuple[str, list[str]]]:
    """
    Function to gather all Python applications in a given directory structure. Check i_find_files for more details.
    Args:
        root (str): The root directory to start the search from.
        ignore_paths (list[str] | None): A list of paths to ignore during the search
    Returns:
        dict[str, tuple[str, list[str]]]: The returned value of i_find_files
    """
    root = pathlib.Path(root)
    ignore_paths = list(map(pathlib.Path, ignore_paths or []))  # Convert ignore paths to Path objects
    return i_find_files(root, ignore_paths, prefix="")  # Start the recursive search with an empty prefix


def i_find_files(root: pathlib.Path, ignore_paths: list[pathlib.Path] | None = None, prefix: str = "") -> dict[str, tuple[str, list[str]]]:
    """
    Immersive function for recursively finding Python files in a given directory structure.

    This class scans and gathers python applications that follow a specific structure:
    - The root directory contains a 'src' directory and a 'README'.
    - The 'src' directory contains Python files (.py).

    It returns a dictionary of the form:
    {
        "application_name": (readme_file_path, [python_file1_path, python_file2_path, ...])
    }

    The application name is derived from the directory structure by concatenating directory names with a hyphen.

    Args:
        root (pathlib.Path): The root directory to start the search from.
        ignore_paths (list[pathlib.Path] | None): A list of paths to ignore during the search.
        prefix (str): A prefix to prepend to the application name, used for recursive calls.
    Returns:
        dict[str, tuple[str, list[str]]]: A dictionary where keys are application names and
        values are tuples containing the path to the README file and a list of paths to Python files
        in the 'src' directory.
    Raises:
        ValueError: If the directory structure is invalid, i.e., if a directory contains a README file
        without a corresponding 'src' directory or vice versa.
    """
    if root in ignore_paths or not root.is_dir():
        # Base case: if the root is in the ignore paths or is not a directory, finish the search
        return {}

    paths_in_root = list(root.iterdir())
    py_script_present = any(map(lambda p: p.is_file() and p.suffix == ".py", paths_in_root))
    readme_present = any(map(lambda p: p.is_file() and p.name == "README", paths_in_root))

    if py_script_present:
        python_files = sorted(map(str, filter(lambda x: x.suffix == '.py', root.iterdir())))
        if readme_present:
            readme_file = str(root / "README")
        else:
            readme_file = None
        return {prefix: (readme_file, python_files)}
    else:
        # Recursive case: continue searching in subdirectories
        results = map(lambda p: i_find_files(p, ignore_paths, prefix + "-" + p.name if prefix else p.name), paths_in_root)
        return reduce(lambda x, y: x | y, results, {})


def compose2(f, g):
    """
    Compose two functions f and g, returning a new function h: x -> f(g(x)).
    Args:
        f: The outer function to apply.
        g: The inner function to apply.
    Returns:
        A single function that represents the composition of f and g.
    Example:
        >>> f = compose2(lambda x: x + 1, lambda x: x * 2)  # h: x -> (x * 2) + 1
        >>> f(3)  # Output will be (3 * 2) + 1 = 7
    """
    return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):
    """
    Compose multiple functions f1, f2, ..., fn; returning a new function g: x -> f1(f2(...(fn(x)))).
    Args:
        *fs: A variable number of functions to compose.
    Returns:
        A single function that represents the composition of all input functions.
    Example:
        >>> def add_one(x): return x + 1
        >>> def double(x): return x * 2
        >>> f = compose(add_one, double) # f: x -> add_one(double(x))
        >>> f(3)  # Output will be (3 * 2) + 1 = 7
    """
    return reduce(compose2, fs)

def generate_unique_dir(base_dir: str) -> pathlib.Path:
    """
    Generate a unique directory name based on the current timestamp and random adjectives and animals.
    The directory will be created under the specified base_dir.
    Args:
        base_dir (str): The base directory where the unique directory will be created.
    Returns:
        pathlib.Path: The path to the newly created unique directory.
    Raises:
        RuntimeError: If a unique directory cannot be created after 100 attempts.
    """
    base_path = pathlib.Path(base_dir)
    base_path.mkdir(exist_ok=True)

    for _ in range(100):  # Avoid infinite loop just in case
        name = f"{random.choice(ADJECTIVES)}_{random.choice(ANIMALS)}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir = base_path / name
        if not run_dir.exists():
            run_dir.mkdir()
            return run_dir
    raise RuntimeError("Could not create a unique output directory after 100 attempts")

class Graph:
    """
    A class representing a directed graph which a method for topological sorting.
    """
    def __init__(self, num_nodes: int):
        """
        Initializes a directed graph with a specified number of nodes.
        The edges are stored in a dictionary where keys are node indices and values are lists of adjacent nodes.
        Args:
            num_nodes (int): The number of nodes in the graph.
        """
        self.edges = defaultdict(list)
        self.num_nodes = num_nodes

    def add_edge(self, u: int, v: int) -> None:
        """
        Adds a directed edge from node u to node v in the graph.
        Args:
            u (int): The source node index.
            v (int): The destination node index.
        """
        self.edges[u].append(v)

    def topological_sort(self):
        """
        Performs a topological sort on the directed graph.
        Returns:
            list[int]: A list of nodes sorted in topological order.
        Raises:
            ValueError: If the graph is not a Directed Acyclic Graph (DAG), meaning it contains cycles.
        Example:
            >>> g = Graph(4)
            >>> g.add_edge(0, 1)
            >>> g.add_edge(0, 2)
            >>> g.add_edge(1, 3)
            >>> g.add_edge(2, 3)
            >>> order = g.topological_sort()
            >>> print(order)  # Output will be a valid topological order, e.g., [0, 1, 2, 3]
        """
        sorted_nodes = []
        # in-degree stands for incoming edges for a node
        # Initialize in-degrees for all nodes
        in_degrees = {i: 0 for i in range(self.num_nodes)}
        for node_edges in self.edges.values():
            for v in node_edges:
                in_degrees[v] += 1

        while len(sorted_nodes) < self.num_nodes:
            # Get the node with in-degree 0, and update the graph after removing it

            min_node = min(in_degrees, key=lambda i: in_degrees[i])
            degree = in_degrees.pop(min_node)

            if degree > 0:
                raise ValueError("Graph is not a DAG (Directed Acyclic Graph)")

            sorted_nodes.append(min_node)
            for v in self.edges[min_node]:
                in_degrees[v] -= 1

        return sorted_nodes

    def __str__(self) -> str:
        """
        Returns a string representation of the graph, showing each node and its adjacent nodes.
        Returns:
            str: The string representation
        Example:
            >>> g = Graph(3)
            >>> g.add_edge(0, 1)
            >>> g.add_edge(0, 2)
            >>> print(g) # Output will be: V = {0, 1, ..., 2}, E = {(0, 1), (0, 2)}
        """
        nodes = "V = {0, 1, ..., %d}" % (self.num_nodes - 1)
        edges = "E = {%s}" % sorted((u, v) for u, vs in self.edges.items() for v in vs)
        edges = edges.replace("[", "").replace("]", "")
        return nodes + ", " + edges


def read_file(path: str) -> str:
    """
    Reads the content of a file at the specified path.
    Args:
        path (str): The path to the file to read.
    Returns:
        str: The content of the file as a string.
    """
    with open(path, 'r') as f:
        content = f.read()
    return content


def log_message(log_func: Callable[[str], Any], info: str) -> Callable[[F], F]:
    """A decorator that logs a message after the wrapped function finishes.

    This decorator supports both normal functions and generator functions:
      - For normal functions, the message is printed immediately after the function returns.
      - For generator functions, the message is printed after the generator is fully
        exhausted (or when iteration stops, e.g., via break).

    Args:
        log_func (callable): A logging function that takes a string argument to log the message.
        info (str): The message to be printed after the function or generator finishes.

    Returns:
        Callable[[F], F]: The decorated function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            if inspect.isgenerator(result):
                def generator_wrapper() -> Generator[Any, None, None]:
                    try:
                        yield from cast(Iterable[Any], result)
                    finally:
                        log_func(info)
                return generator_wrapper()
            else:
                log_func(info)
                return result
        return cast(F, wrapper)
    return decorator
