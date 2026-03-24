"""
Tests for the PyCOMPSs-relevant cleaning functions in generate/utils.py.

Functions under test:
  - extract_pycompss_solution
  - clean_output          (pycompss path only)
  - clean_instruct_output (pycompss path only)

Run with:
  python -m unittest generate/tests/test_cleaning.py
"""

import unittest

from generate.utils import extract_pycompss_solution, clean_output, clean_instruct_output


PYCOMPSS_PROMPT = "Write a PyCOMPSs solution"
PYCOMPSS_RESPONSE_TAG = "@@ Response"


# ===========================================================================
# extract_pycompss_solution
# ===========================================================================

class TestExtractStartDetection(unittest.TestCase):
    """Step 1 – finding the first code line (import / from / @task / def)."""

    def test_starts_with_import(self):
        code = "import numpy as np\n\ndef main():\n    pass"
        self.assertIn("import numpy", extract_pycompss_solution(code))

    def test_starts_with_from(self):
        code = "from pycompss.api.task import task\n\ndef main():\n    pass"
        result = extract_pycompss_solution(code)
        self.assertTrue(result.startswith("from pycompss"))

    def test_starts_with_at_task(self):
        code = "@task(returns=1)\ndef foo():\n    pass\n\ndef main():\n    pass"
        result = extract_pycompss_solution(code)
        self.assertTrue(result.startswith("@task"))

    def test_starts_with_def(self):
        code = "def helper():\n    pass\n\ndef main():\n    pass"
        result = extract_pycompss_solution(code)
        self.assertTrue(result.startswith("def helper"))

    def test_preamble_text_before_code_is_discarded(self):
        code = (
            "Here is my solution:\n"
            "As you can see below:\n"
            "import numpy as np\n"
            "\n"
            "def main():\n"
            "    pass"
        )
        result = extract_pycompss_solution(code)
        self.assertTrue(result.startswith("import numpy"))
        self.assertNotIn("Here is my solution", result)

    def test_no_matching_start_uses_whole_input_from_line_zero(self):
        # No import/from/@task/def — start_index stays 0
        code = "x = 1\ny = 2"
        result = extract_pycompss_solution(code)
        self.assertIn("x = 1", result)

    def test_indented_import_is_matched(self):
        # The pattern allows leading whitespace (\s*)
        code = "  import os\n\ndef main():\n    pass"
        self.assertIn("import os", extract_pycompss_solution(code))

    def test_only_first_match_sets_start(self):
        # Second import should not move start_index backwards
        code = "import os\nimport sys\n\ndef main():\n    pass"
        result = extract_pycompss_solution(code)
        self.assertTrue(result.startswith("import os"))


class TestExtractFunctionInclusion(unittest.TestCase):
    """Step 2 – any function (decorated or not) and its body is included."""

    def test_single_function_included(self):
        code = "import os\n\ndef foo():\n    x = 1"
        result = extract_pycompss_solution(code)
        self.assertIn("def foo", result)
        self.assertIn("x = 1", result)

    def test_def_main_included_as_any_other_function(self):
        code = "import os\n\ndef main():\n    x = 1"
        self.assertIn("def main", extract_pycompss_solution(code))

    def test_multiple_functions_all_included(self):
        code = (
            "import os\n\n"
            "def foo():\n"
            "    return 1\n\n"
            "def bar():\n"
            "    return 2\n\n"
            "def main():\n"
            "    print(foo(), bar())"
        )
        result = extract_pycompss_solution(code)
        self.assertIn("def foo", result)
        self.assertIn("def bar", result)
        self.assertIn("def main", result)

    def test_decorated_function_included(self):
        code = (
            "from pycompss.api.task import task\n\n"
            "@task(returns=1)\n"
            "def compute(x):\n"
            "    return x * 2\n\n"
            "def main():\n"
            "    pass"
        )
        result = extract_pycompss_solution(code)
        self.assertIn("@task(returns=1)", result)
        self.assertIn("def compute", result)

    def test_function_with_arguments_included(self):
        code = "import os\n\ndef process(data, n):\n    return data[:n]"
        self.assertIn("def process(data, n)", extract_pycompss_solution(code))

    def test_function_named_main_helper_included(self):
        # Previously this was a fallback edge case; now it's just a normal function
        code = "import os\n\ndef main_helper():\n    pass"
        self.assertIn("def main_helper", extract_pycompss_solution(code))


class TestExtractStoppingCondition(unittest.TestCase):
    """Step 3 – stop at non-indented code that is not import/decorator/def."""

    def test_if_name_main_block_excluded(self):
        code = (
            "import os\n\n"
            "def main():\n"
            "    x = 1\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )
        result = extract_pycompss_solution(code)
        self.assertNotIn("if __name__", result)
        self.assertIn("main()", result)

    def test_top_level_assignment_excluded(self):
        code = (
            "import os\n\n"
            "def main():\n"
            "    x = 1\n\n"
            "result = main()"
        )
        result = extract_pycompss_solution(code)
        self.assertNotIn("result = main()", result)

    def test_function_body_lines_included(self):
        code = "import os\n\ndef main():\n    x = 1\n    y = 2"
        result = extract_pycompss_solution(code)
        self.assertIn("x = 1", result)
        self.assertIn("y = 2", result)

    def test_multiple_functions_separated_by_blank_lines_all_included(self):
        code = (
            "import os\n\n"
            "def foo():\n"
            "    return 1\n\n\n"
            "def main():\n"
            "    x = 1"
        )
        result = extract_pycompss_solution(code)
        self.assertIn("def foo", result)
        self.assertIn("def main", result)

    def test_blank_lines_inside_function_do_not_break_early(self):
        code = (
            "import os\n\n"
            "def main():\n"
            "    x = 1\n"
            "\n"
            "    y = 2\n"
            "\n"
            "    z = 3"
        )
        result = extract_pycompss_solution(code)
        self.assertIn("x = 1", result)
        self.assertIn("y = 2", result)
        self.assertIn("z = 3", result)

    def test_comment_inside_function_included(self):
        code = (
            "import os\n\n"
            "def main():\n"
            "    # step 1\n"
            "    x = 1"
        )
        self.assertIn("# step 1", extract_pycompss_solution(code))

    def test_top_level_comment_between_functions_included(self):
        # Top-level comments are valid code annotations; they are included
        code = (
            "import os\n\n"
            "# helper function\n"
            "def foo():\n"
            "    return 1\n\n"
            "def main():\n"
            "    pass"
        )
        result = extract_pycompss_solution(code)
        self.assertIn("# helper function", result)

    def test_trailing_top_level_comment_included(self):
        # A top-level comment at the end (before if __name__) is included
        code = (
            "import os\n\n"
            "def main():\n"
            "    x = 1\n"
            "# end of solution"
        )
        result = extract_pycompss_solution(code)
        self.assertIn("# end of solution", result)

    def test_deeply_nested_code_included(self):
        code = (
            "import os\n\n"
            "def main():\n"
            "    for i in range(10):\n"
            "        if i > 5:\n"
            "            print(i)"
        )
        result = extract_pycompss_solution(code)
        self.assertIn("for i in range(10)", result)
        self.assertIn("if i > 5", result)
        self.assertIn("print(i)", result)

    def test_blank_lines_before_if_name_not_included_in_output(self):
        # Trailing blank lines before the stopping line are not included
        # because blank lines never update last_valid_index
        code = (
            "import os\n\n"
            "def main():\n"
            "    x = 1\n"
            "\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )
        result = extract_pycompss_solution(code)
        self.assertFalse(result.endswith("\n\n"))


class TestExtractOutputStripping(unittest.TestCase):
    """Step 4 – the returned string is stripped of leading/trailing whitespace."""

    def test_result_has_no_leading_trailing_whitespace(self):
        code = "\n\n\nimport os\n\ndef main():\n    x = 1\n\n\n"
        result = extract_pycompss_solution(code)
        self.assertEqual(result, result.strip())

    def test_preamble_excluded_from_result(self):
        code = "# preamble\nimport os\n\ndef main():\n    pass"
        # '# preamble' does not match start_pattern, so start_index -> import
        result = extract_pycompss_solution(code)
        self.assertTrue(result.startswith("import os"))


# ===========================================================================
# clean_output  (pycompss path)
# ===========================================================================

class TestCleanOutputPromptStripping(unittest.TestCase):
    """Step 1 – stripping the prompt prefix from the raw output string."""

    def test_prompt_found_is_removed(self):
        prompt = PYCOMPSS_PROMPT
        output = prompt + "\nimport os\n\ndef main():\n    x = 1"
        result = clean_output(output, prompt)
        self.assertNotIn(PYCOMPSS_PROMPT, result)
        self.assertIn("import os", result)

    def test_prompt_not_found_uses_full_output(self):
        prompt = PYCOMPSS_PROMPT
        output = "import os\n\ndef main():\n    x = 1"
        result = clean_output(output, prompt)
        self.assertIn("import os", result)

    def test_prompt_found_mid_output_everything_before_is_dropped(self):
        prompt = PYCOMPSS_PROMPT
        output = "random garbage\n" + prompt + "\nimport os\n\ndef main():\n    x = 1"
        result = clean_output(output, prompt)
        self.assertNotIn("random garbage", result)
        self.assertIn("import os", result)

    def test_raw_output_after_prompt_is_whitespace_stripped_before_extraction(self):
        prompt = PYCOMPSS_PROMPT
        # Several blank lines between prompt and code
        output = prompt + "\n\n\n   import os\n\ndef main():\n    pass"
        result = clean_output(output, prompt)
        self.assertIsInstance(result, str)
        self.assertIn("import os", result)


class TestCleanOutputPycompssDetection(unittest.TestCase):
    """Step 2 – 'pycompss' in prompt triggers extraction; detection is case-insensitive."""

    def test_lowercase_pycompss_triggers_extraction(self):
        prompt = "write a pycompss solution"
        output = prompt + "\nimport os\n\ndef main():\n    x = 1"
        self.assertIn("import os", clean_output(output, prompt))

    def test_uppercase_pycompss_triggers_extraction(self):
        prompt = "Write a PYCOMPSS Solution"
        output = prompt + "\nimport os\n\ndef main():\n    x = 1"
        self.assertIn("import os", clean_output(output, prompt))

    def test_mixed_case_pycompss_triggers_extraction(self):
        prompt = "Use PyCOMPSs to solve this"
        output = prompt + "\nimport os\n\ndef main():\n    x = 1"
        self.assertIn("import os", clean_output(output, prompt))

    def test_extraction_stops_at_if_name_main(self):
        # Proves that clean_output delegates to extract_pycompss_solution
        prompt = PYCOMPSS_PROMPT
        output = (
            prompt + "\n"
            "import os\n\n"
            "def main():\n"
            "    x = 1\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )
        result = clean_output(output, prompt)
        self.assertIn("x = 1", result)
        self.assertNotIn("if __name__", result)


# ===========================================================================
# clean_instruct_output  (pycompss path)
# ===========================================================================

class TestCleanInstructOutputResponseTagStripping(unittest.TestCase):
    """Step 1 – stripping output up to and including the response tag."""

    def test_response_tag_found_preamble_is_dropped(self):
        tag = PYCOMPSS_RESPONSE_TAG
        output = f"some preamble\n{tag}\nimport os\n\ndef main():\n    x = 1\n"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertNotIn("some preamble", result)

    def test_response_tag_found_code_after_tag_is_kept(self):
        tag = PYCOMPSS_RESPONSE_TAG
        output = f"{tag}\nimport os\n\ndef main():\n    x = 1"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertIn("import os", result)

    def test_response_tag_not_found_uses_full_output(self):
        tag = PYCOMPSS_RESPONSE_TAG
        output = "import os\n\ndef main():\n    x = 1"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertIn("import os", result)

    def test_content_after_tag_is_whitespace_stripped(self):
        tag = "### Response"
        output = f"{tag}\n\n   import os\n\ndef main():\n    pass"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertIn("import os", result)


class TestCleanInstructOutputCodeBlockExtraction(unittest.TestCase):
    """Step 2 – extracting fenced code blocks from the stripped output."""

    def test_python_fenced_block_is_extracted(self):
        tag = PYCOMPSS_RESPONSE_TAG
        code = "import os\n\ndef main():\n    x = 1"
        output = f"{tag}\nHere is the answer:\n```python\n{code}\n```\n"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertIn("import os", result)

    def test_plain_fenced_block_is_extracted(self):
        tag = PYCOMPSS_RESPONSE_TAG
        code = "import os\n\ndef main():\n    x = 1"
        output = f"{tag}\n```\n{code}\n```\n"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertIn("import os", result)

    def test_language_tag_not_included_in_extracted_code(self):
        tag = PYCOMPSS_RESPONSE_TAG
        code = "import os\n\ndef main():\n    x = 1"
        output = f"{tag}\n```python\n{code}\n```\n"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        # 'python' from the fence should not appear as a standalone word in code
        self.assertNotIn("python\n", result)

    def test_multiple_fenced_blocks_first_is_used(self):
        tag = PYCOMPSS_RESPONSE_TAG
        code1 = "import os\n\ndef main():\n    x = 1"
        code2 = "import sys\n\ndef main():\n    y = 2"
        output = f"{tag}\n```python\n{code1}\n```\nalso:\n```python\n{code2}\n```\n"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertIn("import os", result)
        self.assertNotIn("import sys", result)

    def test_no_fenced_block_uses_raw_stripped_output(self):
        tag = PYCOMPSS_RESPONSE_TAG
        code = "import os\n\ndef main():\n    x = 1"
        output = f"{tag}\n{code}"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertIn("import os", result)

    def test_no_block_trailing_backticks_stripped(self):
        tag = PYCOMPSS_RESPONSE_TAG
        output = f"{tag}\nimport os\n\ndef main():\n    x = 1\n```"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertNotIn("```", result)

    def test_no_block_leading_backticks_stripped(self):
        tag = PYCOMPSS_RESPONSE_TAG
        # Opens with ``` but no proper closing fence line → falls into no-block path
        output = f"{tag}\n```import os\n\ndef main():\n    x = 1"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        # The 3 backtick chars are stripped from the front of raw_code
        self.assertFalse(result.startswith("```"))


class TestCleanInstructOutputPycompssDetection(unittest.TestCase):
    """Step 3 – 'pycompss' in prompt routes raw_code through extract_pycompss_solution."""

    def test_if_name_main_block_is_trimmed(self):
        tag = PYCOMPSS_RESPONSE_TAG
        code = (
            "import os\n\n"
            "def main():\n"
            "    x = 1\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )
        output = f"{tag}\n```python\n{code}\n```\n"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertIn("x = 1", result)
        self.assertNotIn("if __name__", result)

    def test_pycompss_detection_is_case_insensitive(self):
        tag = PYCOMPSS_RESPONSE_TAG
        prompt = "PYCOMPSS TASK"
        code = "import os\n\ndef main():\n    x = 1"
        output = f"{tag}\n```python\n{code}\n```\n"
        result = clean_instruct_output(output, prompt, tag)
        self.assertIn("import os", result)

    def test_if_name_main_trimmed_without_fenced_block(self):
        tag = PYCOMPSS_RESPONSE_TAG
        code = (
            "import os\n\n"
            "def main():\n"
            "    x = 1\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )
        output = f"{tag}\n{code}"
        result = clean_instruct_output(output, PYCOMPSS_PROMPT, tag)
        self.assertIn("x = 1", result)
        self.assertNotIn("if __name__", result)

    def test_full_realistic_llama3_pycompss_output(self):
        """End-to-end: Llama-3 response tag + python fenced block + full pycompss structure."""
        tag = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt = "Write a PyCOMPSs parallel solution for the following problem"
        code = (
            "from pycompss.api.task import task\n"
            "from pycompss.api.api import compss_wait_on\n"
            "\n"
            "@task(returns=1)\n"
            "def compute(x):\n"
            "    return x * 2\n"
            "\n"
            "def main():\n"
            "    results = [compute(i) for i in range(10)]\n"
            "    results = compss_wait_on(results)\n"
            "    print(results)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )
        output = f"Some explanation.\n{tag}```python\n{code}\n```\n"
        result = clean_instruct_output(output, prompt, tag)
        self.assertIn("from pycompss", result)
        self.assertIn("@task", result)
        self.assertIn("def main", result)
        self.assertIn("print(results)", result)
        # if __name__ block is outside main — must be excluded
        self.assertNotIn("if __name__", result)

    def test_full_realistic_magicoder_pycompss_output(self):
        """End-to-end: Magicoder @@ Response tag + no fenced block."""
        tag = "@@ Response"
        prompt = "Use PyCOMPSs to implement the following"
        code = (
            "from pycompss.api.task import task\n"
            "\n"
            "@task(returns=1)\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "def main():\n"
            "    result = add(1, 2)\n"
            "    print(result)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        output = f"Explanation here.\n{tag}\n{code}"
        result = clean_instruct_output(output, prompt, tag)
        self.assertIn("from pycompss", result)
        self.assertIn("def main", result)
        self.assertIn("print(result)", result)
        self.assertNotIn("if __name__", result)


if __name__ == "__main__":
    unittest.main()
