"""Main script to import PyCOMPSs examples into ParEval JSON format.

Scans a PyCOMPSs examples repository and produces a JSON file compatible
with the ParEval evaluation pipeline. Optionally runs the sequentialized
code for validation.

Supported modes:
1. `generate`: Generates the dataset from the PyCOMPSs examples repository.
2. `reconstruct`: Reconstructs the Python source code from the dataset.
"""

from reconstruct import reconstruct_code
from preprocessor import clean_application
from runner import Runner
from utils import generate_unique_dir, find_files
import argparse
import logging
import pathlib
import sys
import json
import re


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parses command line arguments for the script.
    Args:
        argv (list[str], optional): List of command line arguments. If None, uses sys.argv.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser("Import PyCOMPSs examples into ParEval JSON format.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Generate the dataset mode
    generate = subparsers.add_parser("generate", help="Generate the dataset from the PyCOMPSs examples repository")
    generate.add_argument("--repo", type=str, required=True, help="Path to the PyCOMPSs examples repository")
    generate.add_argument("--ignore", type=str, nargs='*', default=[],
                          help="List of directories to ignore")
    generate.add_argument("--run-sequential", type=str, nargs=2, metavar=("exec_command", "args_file"),
                          help="Run each sequential code using the provided command and arguments file."
                               " Example: --run-sequential \"env/bin/python -c\" arguments.json. More info in the README.")
    generate.add_argument("--output", type=str, default=None,
                          help="Path for the output JSON file. Accepts absolute or relative paths."
                               " Defaults to <save_dir>/dataset/generate_output.json.")

    # Reconstruct mode
    reconstruct = subparsers.add_parser("reconstruct", help="Reconstruct the Python source code from the dataset")
    reconstruct.add_argument("--dataset-file", type=str, required=True, help="Path to the dataset json file")
    reconstruct.add_argument("--dataset-type", type=str, choices=["seq_to_par", "descr_to_par"],
                             required=True, help="Type of dataset to reconstruct")

    arguments = parser.parse_args(argv)
    logger.info(f"Arguments received: {arguments}")
    return arguments


def generate_dataset_pipeline(arguments: argparse.Namespace, save_dir: pathlib.Path | None) -> None:
    """
    Generates a ParEval-compatible dataset from the PyCOMPSs examples repository.
    Args:
        arguments (argparse.Namespace): Parsed command line arguments.
        save_dir (pathlib.Path | None): Directory where auxiliary output files will be saved.
            Not required when --output is provided and --run-sequential is not used.
    """
    repo_path = arguments.repo
    logger.info(f"Processing repository: {repo_path}")

    runner: Runner | None = None
    execution_results_dir: pathlib.Path | None = None
    if arguments.run_sequential is not None:
        execution_results_dir = save_dir / "execution_results"
        execution_results_dir.mkdir(parents=True, exist_ok=True)
        runner = Runner(*arguments.run_sequential, base_dir=repo_path)

    files = find_files(repo_path, arguments.ignore)
    logger.info(f"Found {len(files)} applications in the repository.")
    logger.debug(f"Applications found: {files}")

    templates_path = pathlib.Path(__file__).parent / "templates.json"
    with open(templates_path) as f:
        template = json.load(f)

    entries = []
    for i, (name, (readme_file, python_files)) in enumerate(files.items(), start=1):
        logger.info(f"Processing application {i}/{len(files)}: {name}")
        logger.info(f"Python files: {python_files}")

        if runner is not None:
            _, sequential_code = clean_application(python_files)
            runner.add_task(sequential_code, name)

        parts = name.split('-')
        problem_type = parts[0]
        docstring = extract_docstring_from_file(python_files[0]) if python_files else ""
        code = "\n".join(open(f).read() for f in python_files)

        entry = dict(template)
        entry["problem_type"] = problem_type
        entry["language"] = "python"
        entry["name"] = name.replace("-", "_")
        entry["parallelism_model"] = "pycompss"
        entry["prompt"] = docstring
        entry["outputs"] = [code]
        entry["raw_outputs"] = []
        entries.append(entry)

    if arguments.output is not None:
        output_path = str(pathlib.Path(arguments.output).resolve())
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        dataset_dir = save_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(dataset_dir / "generate_output.json")
    with open(output_path, 'w') as f:
        json.dump(entries, f, indent=2)
    logger.info(f"Output written to {output_path}")

    if runner is not None:
        runner.run_tasks(str(execution_results_dir))

    logger.info(f"Finished processing repository: {repo_path}")


def extract_docstring_from_file(pyfile):
    with open(pyfile, 'r') as f:
        content = f.read()
    match = re.match(r'([\s\n]*)([\'\"]{3})(.*?)([\'\"]{3})', content, re.DOTALL)
    if match:
        return match.group(3).strip()
    return ""


def setup_logger():
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler("logging.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("Logger is set up.")


if __name__ == '__main__':
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Starting the script")

    args = parse_args()

    needs_save_dir = (args.mode == "reconstruct" or
                      getattr(args, 'output', None) is None or
                      getattr(args, 'run_sequential', None) is not None)

    if needs_save_dir:
        output_dir = pathlib.Path(__file__).parent.parent / ".script_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_dir = generate_unique_dir(str(output_dir))
    else:
        save_dir = None

    if args.mode == "generate":
        generate_dataset_pipeline(args, save_dir)
    elif args.mode == "reconstruct":
        reconstruct_code(args.dataset_file, args.dataset_type, save_dir)
