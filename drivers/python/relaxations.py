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
    # Parallelism models this relaxation is relevant for
    applies_to: set[str] = {"pycompss"}

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
    applies_to = {"pycompss", "serial"}

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
    Also drops 'import pycompss.interactive' (causes NoneType crash in batch
    context) and any top-level 'compss.*' imports (package does not exist).
    Applies only when at least one pycompss/compss import is present.
    """
    name = "fix_pycompss_imports"

    DEFAULT_IMPORTS = [
        "from pycompss.api.task import *",
        "from pycompss.api.parameter import *",
        "from pycompss.api.api import *",
        "from pycompss.api.constraint import constraint",
        "from pycompss.api.binary import binary",
    ]

    def apply(self, source: str) -> Optional[str]:
        lines = source.split("\n")
        new_lines = []
        found = False
        defaults_written = False

        for line in lines:
            is_pycompss = re.match(r"\s*(from\s+pycompss|import\s+pycompss)", line)
            is_bad_compss = re.match(r"\s*(from\s+compss[.\s]|import\s+compss\b)", line)

            if is_pycompss or is_bad_compss:
                found = True
                if is_pycompss and not defaults_written:
                    new_lines.extend(self.DEFAULT_IMPORTS)
                    defaults_written = True
                # Drop the original line in all cases
            else:
                new_lines.append(line)

        if not found:
            return None

        return "\n".join(new_lines)


class RemoveFutureImportsRelaxation(Relaxation):
    """Remove 'from __future__ import ...' lines.

    When generated code is merged with driver_config.py (which contains
    Python assignment statements), any 'from __future__ import' becomes a
    SyntaxError because it no longer appears at the very start of the merged
    file.  These imports are cosmetic in this context and are simply dropped.
    """
    name = "remove_future_imports"
    applies_to = {"pycompss", "serial"}

    def apply(self, source: str) -> Optional[str]:
        lines = source.split("\n")
        filtered = [l for l in lines if not re.match(r"\s*from\s+__future__\s+import\b", l)]
        if len(filtered) == len(lines):
            return None
        return "\n".join(filtered)


class FixWrongTaskDecoratorRelaxation(Relaxation):
    """Replace incorrectly namespaced @task decorators with the bare @task form.

    Models frequently use @pycompss.task, @compss.task, @pycompss.api.task,
    or @compss.api.task.  The correct form is the bare @task imported from
    pycompss.api.task.  This relaxation performs the substitution and adds
    the correct import if it is not already present.

    Patterns observed in outputs (by frequency):
        @compss.task(...)          566 lines
        @pycompss.api.task(...)    ~520 lines
        @pycompss.task(...)        ~440 lines
        @compss.api.task(...)       67 lines
    """
    name = "fix_task_decorator"

    _WRONG_DEC = re.compile(r"^(\s*)@(pycompss|compss)(\.api)?\.task\b", re.MULTILINE)
    _HAS_TASK_IMPORT = re.compile(r"from\s+pycompss\.api\.task\s+import\s+(\*|[^#\n]*\btask\b)")
    _CORRECT_IMPORT = "from pycompss.api.task import task"

    def apply(self, source: str) -> Optional[str]:
        if not self._WRONG_DEC.search(source):
            return None

        modified = self._WRONG_DEC.sub(r"\1@task", source)

        if not self._HAS_TASK_IMPORT.search(modified):
            lines = modified.split("\n")
            insert_at = 0
            for i, line in enumerate(lines):
                if re.match(r"\s*(from|import)\s+", line):
                    insert_at = i + 1
            lines.insert(insert_at, self._CORRECT_IMPORT)
            modified = "\n".join(lines)

        return modified


class FixFutureResolutionRelaxation(Relaxation):
    """Replace .result() / .get() Future resolution with compss_wait_on().

    PyCOMPSs Future objects have no .result() (asyncio/Java pattern) or
    .get() (concurrent.futures pattern) methods.  The only correct way to
    synchronise is compss_wait_on().

    Handles list-comprehension form (most common in outputs):
        [f.result() for f in futures]  ->  compss_wait_on(futures)
        [f.get()    for f in futures]  ->  compss_wait_on(futures)

    And bare single-value calls:
        future.result()  ->  compss_wait_on(future)
        future.get()     ->  compss_wait_on(future)

    Also adds the compss_wait_on import if not already present.
    """
    name = "fix_future_resolution"

    _LC_RESULT = re.compile(r"\[\s*\w+\.result\(\)\s+for\s+\w+\s+in\s+(\w+)\s*\]")
    _LC_GET    = re.compile(r"\[\s*\w+\.get\(\)\s+for\s+\w+\s+in\s+(\w+)\s*\]")
    _SIMPLE    = re.compile(r"(\w+)\.(result|get)\(\s*\)")

    _HAS_WAIT  = re.compile(r"from\s+pycompss\.api\.api\s+import\s+(\*|[^#\n]*\bcompss_wait_on\b)")
    _WAIT_IMPORT = "from pycompss.api.api import compss_wait_on"

    def apply(self, source: str) -> Optional[str]:
        if not (self._LC_RESULT.search(source) or
                self._LC_GET.search(source) or
                self._SIMPLE.search(source)):
            return None

        modified = self._LC_RESULT.sub(lambda m: f"compss_wait_on({m.group(1)})", source)
        modified = self._LC_GET.sub(   lambda m: f"compss_wait_on({m.group(1)})", modified)
        modified = self._SIMPLE.sub(   lambda m: f"compss_wait_on({m.group(1)})", modified)

        if not self._HAS_WAIT.search(modified):
            lines = modified.split("\n")
            insert_at = 0
            for i, line in enumerate(lines):
                if re.match(r"\s*(from|import)\s+", line):
                    insert_at = i + 1
            lines.insert(insert_at, self._WAIT_IMPORT)
            modified = "\n".join(lines)

        return modified


class RemoveInteractiveImportRelaxation(Relaxation):
    """Remove unused 'import pycompss.interactive' lines.

    Importing the interactive module in a batch context initialises a global
    PyCOMPSs state object and leaves it as None when start() is never called.
    Any subsequent runtime call then fails with NoneType errors.  When the
    imported alias is not actually called anywhere in the code the import is
    simply dropped.

    Observed in outputs (confirmed counts):
        starcoder2-15b:           383 outputs
        DeepSeek-Coder-V2-Lite-Base: 87 outputs
    """
    name = "remove_interactive_import"

    _IMPORT_RE = re.compile(r"import\s+pycompss\.interactive(?:\s+as\s+(\w+))?$")
    _FROM_RE   = re.compile(r"from\s+pycompss\.interactive\b")

    def apply(self, source: str) -> Optional[str]:
        lines = source.split("\n")
        drop_indices: set[int] = set()
        aliases: set[str] = set()

        for i, line in enumerate(lines):
            s = line.strip()
            m = self._IMPORT_RE.match(s)
            if m:
                drop_indices.add(i)
                if m.group(1):
                    aliases.add(m.group(1))
                continue
            if self._FROM_RE.match(s):
                drop_indices.add(i)

        if not drop_indices:
            return None

        remaining = "\n".join(l for i, l in enumerate(lines) if i not in drop_indices)

        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\s*\.", remaining):
                return None  # alias is actually used, leave it

        return remaining


class FixCollectParameterRelaxation(Relaxation):
    """Replace the hallucinated COLLECT parameter constant with COLLECTION.

    The PyCOMPSs parameter module exposes COLLECTION (and COLLECTION_IN,
    COLLECTION_OUT, etc.) but not COLLECT.  Word-boundary matching ensures
    the substitution does not fire inside 'COLLECTION' itself.
    """
    name = "fix_collect_parameter"

    _RE = re.compile(r"\bCOLLECT\b")

    def apply(self, source: str) -> Optional[str]:
        if not self._RE.search(source):
            return None
        return self._RE.sub("COLLECTION", source)


class RemoveHallucinatedImportsRelaxation(Relaxation):
    """Strip import lines that reference non-existent PyCOMPSs symbols or modules.

    Three categories of hallucinated imports are handled:

    1.  Top-level 'compss.*' package does not exist; pycompss is the real
        package.  All imports from 'compss.*' are dropped unconditionally.

        Most frequent (from kernel-20 outputs):
          from compss.management import TaskGroup        36 066 lines
          from compss.onsched import on_sched             2 966 lines
          from compss.api import compss_get_*             2 922 lines
          from compss.utilities.parameter import InOut    1 474 lines
          from compss.ons.ons import ONs                  1 472 lines
          from compss.constants import {DISK,MEMORY}_SCOPE  644 lines each

    2.  Non-existent pycompss submodule paths: lines are dropped entirely.
        Includes: pycompss.api.parallel, pycompss.api.decos, pycompss.api.data,
        pycompss.api.shared, pycompss.api.constants, pycompss.api.main,
        pycompss.api.operators, pycompss.api.COMPSs, pycompss.decorators,
        pycompss.api.compss, pycompss.task, pycompss.interactive, exaqute,
        pycomps (typo).

    3.  Known hallucinated symbol names imported from real module paths: the
        module path is valid but the symbol does not exist.  Lines importing
        only bad names are dropped; lines mixing valid and hallucinated names
        are rewritten to keep only the valid names.

        Hallucinated pycompss.api.api names (verified against pycompss 3.4):
          compss_type, compss_close, compss_close_file, compss_close_object,
          compss_barrier_object, compss_wait_on_object, compss_wait_on_objects,
          compss_wait_on_group, compss_open_object, compss_open_file,
          compss_delete_objects, compss_delete_files, compss_delete_object_file,
          compss_get_file, compss_barrier_file, compss_wait_on_files,
          compss_objects_barrier, compss_function, compss_reduce, compss_map,
          compss_filter, compss_wait, compss_submit, compss_result,
          compss_future, compss_code, compss_get_task_execution_time,
          compss_barrier_group_objects, compss_wait_on_directory_object,
          wait, code

        Hallucinated pycompss.api / pycompss.api direct imports:
          get_current_task_group_* (many variants), get_current_task_graph_*,
          parallel, comp, compss (re-import of package), task (wrong path),
          compss_wait_on (wrong path, correct is from pycompss.api.api)

        Hallucinated pycompss.api.parameter names:
          Input, Range, DIRECTORY_FILE_STREAM_INOUT, auto, data
    """
    name = "remove_hallucinated_imports"

    _BAD_MODULE_PATHS = frozenset({
        "pycomps",                   # typo: missing final 's'
        "pycompss.interactive",
        "pycompss.task",
        "pycompss.api.parallel",
        "pycompss.api.decos",
        "pycompss.api.data",
        "pycompss.api.shared",
        "pycompss.api.constants",
        "pycompss.api.main",
        "pycompss.api.operators",
        "pycompss.api.COMPSs",
        "pycompss.decorators",
        "pycompss.api.compss",
        "exaqute",
    })

    # Hallucinated names that appear in 'from pycompss.api.api import ...'
    _BAD_API_NAMES = frozenset({
        "compss_type", "compss_close", "compss_close_file", "compss_close_object",
        "compss_barrier_object", "compss_wait_on_object", "compss_wait_on_objects",
        "compss_wait_on_group", "compss_open_object", "compss_open_file",
        "compss_delete_objects", "compss_delete_files", "compss_delete_object_file",
        "compss_get_file", "compss_barrier_file", "compss_wait_on_files",
        "compss_objects_barrier", "compss_function", "compss_reduce", "compss_map",
        "compss_filter", "compss_wait", "compss_submit", "compss_result",
        "compss_future", "compss_code", "compss_get_task_execution_time",
        "compss_barrier_group_objects", "compss_wait_on_directory_object",
        "wait", "code",
    })

    # Hallucinated names imported directly from 'pycompss.api' (the package
    # __init__ is empty; nothing is exported from there directly).
    _BAD_API_DIRECT_NAMES = frozenset({
        # long hallucinated get_current_task_* / get_current_task_graph_* names
        "get_current_task_group_tasks_status_ids_names",
        "get_current_task_group_tasks_status_names_ids",
        "get_current_task_graph_nodes_ids",
        "get_current_task_group_tasks_return_values_types_names_",
        "get_current_task_graph_dependencies_versions",
        "get_current_task_group_tasks_resumed_times",
        "get_current_task_group_tasks_execution_worker_exception",
        "get_current_task_group_tasks_returns_values_types_value",
        "get_current_task_group_tasks_return_values_values_value",
        "get_current_task_group_tasks_returns_values_types",
        "get_current_task_group_tasks_execution_energy_breakdown",
        "get_current_task_group_ranks_list_list_list_list_list_l",
        "get_current_task_graph_node_dependencies_inputs",
        "get_current_task_graph_node_dependencies_outputs",
        "get_current_task_graph_node_input_data_type_size",
        "get_current_task_graph_node_output_data_type_size",
        # wrong paths / hallucinated shorts
        "parallel", "comp", "compss",
        # these exist but at the wrong path
        "task",          # correct: from pycompss.api.task import task
        "compss_wait_on",  # correct: from pycompss.api.api import compss_wait_on
        "compss_constants",
    })

    # Hallucinated names imported from pycompss.api.parameter or pycompss.api.task
    _BAD_PARAM_NAMES = frozenset({
        "Input", "Range", "DIRECTORY_FILE_STREAM_INOUT",
        "auto",   # from pycompss.api.task import auto
        "data",   # from pycompss.api.task import data
    })

    @staticmethod
    def _parse_names(import_line: str) -> set:
        """Return set of original imported names (before any 'as' alias)."""
        m = re.match(r"\s*from\s+\S+\s+import\s+(.+)", import_line)
        if not m:
            return set()
        spec = re.sub(r"#.*", "", m.group(1)).strip()
        if spec == "*":
            return {"*"}
        names = set()
        for part in spec.split(","):
            orig = part.strip().split(" as ")[0].strip()
            if orig:
                names.add(orig)
        return names

    def _process_line(self, line: str):
        """Return (keep: bool, replacement: str | None)."""
        s = line.strip()
        if not re.match(r"(from|import)\s+", s):
            return True, None

        indent = len(line) - len(line.lstrip())
        pad = " " * indent

        # ── 1. Top-level 'compss' package (not pycompss) ────────────────────
        if re.match(r"(from\s+compss[.\s]|import\s+compss(\s|$))", s):
            return False, None

        # ── 2. Known bad pycompss submodule paths ────────────────────────────
        for mod in self._BAD_MODULE_PATHS:
            if re.match(
                r"(from\s+" + re.escape(mod) + r"(\s|\.|$)"
                r"|import\s+" + re.escape(mod) + r"(\s|$))", s
            ):
                return False, None

        # ── 3a. from pycompss.api.api import <names> ────────────────────────
        if re.match(r"from\s+pycompss\.api\.api\s+import\b", s):
            names = self._parse_names(s)
            if "*" in names:
                return True, None
            bad = names & self._BAD_API_NAMES
            if bad:
                valid = names - self._BAD_API_NAMES
                if not valid:
                    return False, None
                return True, pad + f"from pycompss.api.api import {', '.join(sorted(valid))}"

        # ── 3b. from pycompss.api import <names>  (direct, not submodule) ───
        if re.match(r"from\s+pycompss\.api\s+import\b", s) and \
                not re.match(r"from\s+pycompss\.api\.", s):
            names = self._parse_names(s)
            if "*" in names:
                return True, None
            # everything in _BAD_API_DIRECT_NAMES is wrong here; also catch any
            # name that looks like a get_current_task_* hallucination not listed
            bad = {n for n in names
                   if n in self._BAD_API_DIRECT_NAMES
                   or n.startswith("get_current_task")}
            if bad:
                valid = names - bad
                if not valid:
                    return False, None
                return True, pad + f"from pycompss.api import {', '.join(sorted(valid))}"

        # ── 3c. from pycompss.api.parameter import <names> ──────────────────
        if re.match(r"from\s+pycompss\.api\.parameter\s+import\b", s):
            names = self._parse_names(s)
            if "*" in names:
                return True, None
            bad = names & self._BAD_PARAM_NAMES
            if bad:
                valid = names - self._BAD_PARAM_NAMES
                if not valid:
                    return False, None
                return True, pad + f"from pycompss.api.parameter import {', '.join(sorted(valid))}"

        # ── 3d. from pycompss.api.task import <names> ───────────────────────
        if re.match(r"from\s+pycompss\.api\.task\s+import\b", s):
            names = self._parse_names(s)
            if "*" in names:
                return True, None
            bad = names & self._BAD_PARAM_NAMES
            if bad:
                valid = names - self._BAD_PARAM_NAMES
                if not valid:
                    return False, None
                return True, pad + f"from pycompss.api.task import {', '.join(sorted(valid))}"

        return True, None

    def apply(self, source: str) -> Optional[str]:
        lines = source.split("\n")
        new_lines = []
        changed = False

        for line in lines:
            keep, replacement = self._process_line(line)
            if not keep:
                changed = True
            elif replacement is not None:
                new_lines.append(replacement)
                changed = True
            else:
                new_lines.append(line)

        return "\n".join(new_lines) if changed else None


# ── Registry ────────────────────────────────────────────────────────────────

# Ordered list of all available relaxations.
# The order determines retry priority: earlier relaxations are tried first.
ALL_RELAXATIONS: list[Relaxation] = [
    RenameMainRelaxation(),
    RemoveFutureImportsRelaxation(),
    FixWrongTaskDecoratorRelaxation(),
    RemoveHallucinatedImportsRelaxation(),
    FixFutureResolutionRelaxation(),
    RemoveInteractiveImportRelaxation(),
    FixCollectParameterRelaxation(),
    FixPyCompssImportsRelaxation(),
]

# Name → instance mapping, used by the CLI to resolve user-specified names.
RELAXATION_MAP: dict[str, Relaxation] = {r.name: r for r in ALL_RELAXATIONS}
