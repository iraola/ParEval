""" Source-code relaxations for Python/PyCOMPSs generated outputs.

A relaxation is a named, targeted transformation applied to `generated_code.py`
when an initial run fails, to work around common LLM code issues and retry.
Each relaxation returns the modified source string, or None if it cannot apply.
"""
import ast
import re
from typing import Optional


class Relaxation:
    name: str

    def apply(self, source: str) -> Optional[str]:
        """Return modified source, or None if this relaxation does not apply."""
        raise NotImplementedError


class RenameMainRelaxation(Relaxation):
    """Rename the sole non-decorated top-level function to 'main'.

    Applies when the generated code has exactly one non-decorated top-level
    function and it is not already called 'main'. Uses the AST to identify
    the target name, then regex to rename the definition and call sites while
    preserving formatting, comments, and docstrings.
    """
    name = "rename_main"

    def apply(self, source: str) -> Optional[str]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        top_level_undecorated = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.decorator_list
        ]

        if len(top_level_undecorated) != 1:
            return None
        old_name = top_level_undecorated[0].name
        if old_name == "main":
            return None

        # Rename `def <old_name>(` and bare call sites `<old_name>(` using
        # word-boundary regex to avoid partial matches.
        modified = re.sub(
            r'\bdef\s+' + re.escape(old_name) + r'\s*\(',
            'def main(',
            source,
        )
        modified = re.sub(
            r'\b' + re.escape(old_name) + r'\s*\(',
            'main(',
            modified,
        )
        return modified


class FixPyCompssImportsRelaxation(Relaxation):
    """Replace PyCOMPSs imports with broad wildcard defaults.

    When the generated code uses specific PyCOMPSs imports that may be
    incomplete or misspelled, this relaxation replaces ALL pycompss import
    lines with a safe set of wildcard imports that cover the full public API.
    Applies only when at least one pycompss import is present.
    """
    name = "fix_pycompss_imports"

    DEFAULT_IMPORTS = [
        "from pycompss.api.task import *",
        "from pycompss.api.parameter import *",
        "from pycompss.api.api import *",
    ]

    def apply(self, source: str) -> Optional[str]:
        lines = source.split("\n")
        new_lines = []
        found_pycompss = False
        defaults_written = False

        for line in lines:
            if re.match(r"\s*(from\s+pycompss|import\s+pycompss)", line):
                found_pycompss = True
                if not defaults_written:
                    new_lines.extend(self.DEFAULT_IMPORTS)
                    defaults_written = True
                # Drop the original import line
            else:
                new_lines.append(line)

        if not found_pycompss:
            return None

        return "\n".join(new_lines)


# Ordered list of all available relaxations.
# The order determines the retry priority: earlier relaxations are tried first.
ALL_RELAXATIONS: list[Relaxation] = [
    RenameMainRelaxation(),
    FixPyCompssImportsRelaxation(),
]

# Name → instance mapping, used by the CLI to resolve user-specified names.
RELAXATION_MAP: dict[str, Relaxation] = {r.name: r for r in ALL_RELAXATIONS}
