"""
This script scrapes a dataset of with question-answer fields and generates Python source code files with the corresponding codes.
Check the method `Evaluator.scrape_dataset()` to see how it works.
"""

import pathlib
import json
import re
from typing import AnyStr


def match_or_raise(pattern: str, text: str) -> str:
    """
    Match a regex pattern in the given text and return the first capturing group.
    If the pattern is not found, raise a ValueError with a descriptive message.
    Args:
        pattern (str): The regex pattern to match.
        text (str): The text to search within.
    Returns:
        str: The first capturing group from the match.
    Raises:
        ValueError: If the pattern is not found in the text.
    """
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        result = match.group(1)
    else:
        raise ValueError(f"Pattern {pattern} not found in {text}")
    return result

def scrape_template(json_file: pathlib.Path, output_file_template: str, question_pattern: AnyStr, answer_pattern: AnyStr):
    content = json.loads(json_file.read_bytes())
    for i in range(len(content)):
        output_file = pathlib.Path(output_file_template % i)
        question_match = match_or_raise(question_pattern, content[i]["question"])
        answer_match = match_or_raise(answer_pattern, content[i]["answer"])
        output = f'"""\n{question_match}\n"""\n{answer_match}'
        output_file.write_text(output)

def reconstruct_code(dataset_file: str, dataset_type: str, save_dir: pathlib.Path) -> None:
    """
    Reconstructs the code from the dataset file and saves the generated Python source code files in the specified
    output directory.
    Args:
        dataset_file (str): The path to the dataset file containing the question-answer pairs.
        dataset_type (str): The type of dataset to reconstruct (one of "seq_to_par" or "descr_to_par").
        save_dir (pathlib.Path): Directory where the output files will be saved.
    Raises:
        ValueError: If the dataset file does not exist or is not a valid JSON file.
        ValueError: If the dataset type is not one of "seq_to_par" or "descr_to_par".
        ValueError: If the templates file does not exist or is not a file.
    """
    dataset_file = pathlib.Path(dataset_file)
    if not dataset_file.exists() or not dataset_file.is_file() or not dataset_file.suffix == '.json':
        raise ValueError(f"Dataset file {dataset_file} does not exist or is not a valid JSON file.")

    output_dir = save_dir / "dataset_codes"
    output_dir.mkdir(parents=True, exist_ok=True)

    templates_path = pathlib.Path(__file__).parent / "templates.json"
    if not templates_path.exists() or not templates_path.is_file():
        raise ValueError(f"Templates file {templates_path} does not exist or is not a file.")
    content = json.loads(templates_path.read_bytes())

    filename_template = str(output_dir / ("_" + dataset_file.stem + "_code%d.py"))

    if dataset_type == "descr_to_par":
        question_pattern = re.escape(content["question_descr_to_par"]).replace(r"%s", r"(.*)")
        answer_pattern = re.escape(content["answer_descr_to_par"]).replace(r"%s", r"(.*)")
        scrape_template(dataset_file, filename_template, question_pattern, answer_pattern)
    elif dataset_type == "seq_to_par":
        question_pattern = re.escape(content["question_seq_to_par"]).replace(r"%s", r"(.*)")
        answer_pattern = re.escape(content["answer_seq_to_par"]).replace(r"%s", r"(.*)")
        scrape_template(dataset_file, filename_template, question_pattern, answer_pattern)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Must be one of 'seq_to_par' or 'descr_to_par'.")
