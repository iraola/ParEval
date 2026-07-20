"""Unit tests for drivers/python/relaxations.py.

Run from the repo root:
    cd drivers && python -m unittest python.test_relaxations -v
Or from drivers/python/:
    python -m unittest test_relaxations -v
"""
import unittest

from relaxations import (
    ALL_RELAXATIONS,
    RELAXATION_MAP,
    RenameMainRelaxation,
    FixPyCompssImportsRelaxation,
    RemoveFutureImportsRelaxation,
    FixWrongTaskDecoratorRelaxation,
    FixFutureResolutionRelaxation,
    RemoveInteractiveImportRelaxation,
    FixCollectParameterRelaxation,
    RemoveHallucinatedImportsRelaxation,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def lines(s):
    """Split source into stripped non-empty lines for easy assertion."""
    return [l for l in s.splitlines() if l.strip()]


# ── RenameMainRelaxation ─────────────────────────────────────────────────────

class TestRenameMain(unittest.TestCase):
    def setUp(self):
        self.r = RenameMainRelaxation()

    def test_renames_sole_undecorated_function(self):
        src = (
            "from pycompss.api.task import task\n"
            "\n"
            "@task(returns=1)\n"
            "def compute(x):\n"
            "    return x\n"
            "\n"
            "def solve(A, b):\n"
            "    result = compute(A)\n"
            "    return result\n"
        )
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertIn("def main(", out)
        self.assertNotIn("def solve(", out)

    def test_renames_call_sites(self):
        src = "def solve(x):\n    return x\n\nresult = solve(data)\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertIn("main(data)", out)
        self.assertNotIn("solve(data)", out)

    def test_no_op_when_already_named_main(self):
        src = "def main(x):\n    return x\n"
        self.assertIsNone(self.r.apply(src))

    def test_no_op_when_multiple_undecorated_functions(self):
        src = "def foo(x):\n    pass\n\ndef bar(x):\n    pass\n"
        self.assertIsNone(self.r.apply(src))

    def test_no_op_when_only_decorated_functions(self):
        src = "@task(returns=1)\ndef compute(x):\n    return x\n"
        self.assertIsNone(self.r.apply(src))

    def test_no_op_on_syntax_error(self):
        self.assertIsNone(self.r.apply("def (broken:"))

    def test_decorated_functions_not_counted(self):
        # one @task function + one bare function → should rename the bare one
        src = (
            "@task(returns=1)\n"
            "def compute(x):\n"
            "    return x\n"
            "\n"
            "def run(A):\n"
            "    return compute(A)\n"
        )
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertIn("def main(", out)


# ── FixPyCompssImportsRelaxation ─────────────────────────────────────────────

class TestFixPyCompssImports(unittest.TestCase):
    def setUp(self):
        self.r = FixPyCompssImportsRelaxation()
        self.defaults = set(FixPyCompssImportsRelaxation.DEFAULT_IMPORTS)

    def test_replaces_specific_imports_with_wildcards(self):
        src = "from pycompss.api.task import task\nimport numpy as np\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        for imp in self.defaults:
            self.assertIn(imp, out)
        self.assertIn("import numpy as np", out)
        self.assertNotIn("from pycompss.api.task import task", out)

    def test_defaults_written_only_once(self):
        src = (
            "from pycompss.api.task import task\n"
            "from pycompss.api.api import compss_wait_on\n"
        )
        out = self.r.apply(src)
        self.assertEqual(out.count("from pycompss.api.task import *"), 1)

    def test_no_op_when_no_pycompss(self):
        self.assertIsNone(self.r.apply("import numpy as np\n"))

    def test_drops_interactive_import(self):
        src = "import pycompss.interactive as ipycompss\nfrom pycompss.api.task import task\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertNotIn("pycompss.interactive", out)

    def test_drops_top_level_compss_imports(self):
        src = (
            "from compss.management import TaskGroup\n"
            "from pycompss.api.task import task\n"
        )
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertNotIn("compss.management", out)

    def test_preserves_non_pycompss_lines(self):
        src = "import numpy as np\nfrom pycompss.api.task import task\nx = 1\n"
        out = self.r.apply(src)
        self.assertIn("import numpy as np", out)
        self.assertIn("x = 1", out)


# ── RemoveFutureImportsRelaxation ─────────────────────────────────────────────

class TestRemoveFutureImports(unittest.TestCase):
    def setUp(self):
        self.r = RemoveFutureImportsRelaxation()

    def test_removes_future_import(self):
        src = "from __future__ import annotations\nimport numpy as np\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertNotIn("__future__", out)
        self.assertIn("import numpy as np", out)

    def test_removes_multiple_future_imports(self):
        src = "from __future__ import annotations\nfrom __future__ import print_function\nx = 1\n"
        out = self.r.apply(src)
        self.assertEqual(out.count("__future__"), 0)
        self.assertIn("x = 1", out)

    def test_no_op_when_absent(self):
        self.assertIsNone(self.r.apply("import numpy as np\n"))

    def test_no_op_on_empty(self):
        self.assertIsNone(self.r.apply(""))


# ── FixWrongTaskDecoratorRelaxation ───────────────────────────────────────────

class TestFixWrongTaskDecorator(unittest.TestCase):
    def setUp(self):
        self.r = FixWrongTaskDecoratorRelaxation()

    def _has_task_import(self, src):
        return "from pycompss.api.task import task" in src

    def test_fixes_compss_task(self):
        src = "@compss.task(returns=1)\ndef foo(x):\n    return x\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertIn("@task(returns=1)", out)
        self.assertNotIn("@compss.task", out)

    def test_fixes_pycompss_task(self):
        src = "@pycompss.task(returns=1)\ndef foo(x):\n    return x\n"
        out = self.r.apply(src)
        self.assertIn("@task(returns=1)", out)

    def test_fixes_pycompss_api_task(self):
        src = "@pycompss.api.task(returns=np.ndarray)\ndef foo(x):\n    return x\n"
        out = self.r.apply(src)
        self.assertIn("@task(returns=np.ndarray)", out)

    def test_fixes_compss_api_task(self):
        src = "@compss.api.task(returns=int)\ndef foo(x):\n    return x\n"
        out = self.r.apply(src)
        self.assertIn("@task(returns=int)", out)

    def test_fixes_bare_decorator_no_args(self):
        src = "@compss.task\ndef foo(x):\n    return x\n"
        out = self.r.apply(src)
        self.assertIn("@task\n", out)

    def test_adds_import_when_missing(self):
        src = "@compss.task(returns=1)\ndef foo(x):\n    return x\n"
        out = self.r.apply(src)
        self.assertTrue(self._has_task_import(out))

    def test_no_duplicate_import(self):
        src = (
            "from pycompss.api.task import task\n"
            "@compss.task(returns=1)\n"
            "def foo(x):\n    return x\n"
        )
        out = self.r.apply(src)
        self.assertEqual(out.count("from pycompss.api.task import task"), 1)

    def test_import_also_added_with_no_existing_imports(self):
        src = "@compss.task(returns=1)\ndef foo(x):\n    return x\n"
        out = self.r.apply(src)
        self.assertTrue(self._has_task_import(out))

    def test_no_op_when_already_correct(self):
        src = (
            "from pycompss.api.task import task\n"
            "@task(returns=1)\n"
            "def foo(x):\n    return x\n"
        )
        self.assertIsNone(self.r.apply(src))

    def test_no_op_when_bare_task_with_wildcard_import(self):
        src = (
            "from pycompss.api.task import *\n"
            "@task(returns=1)\n"
            "def foo(x):\n    return x\n"
        )
        self.assertIsNone(self.r.apply(src))

    def test_wildcard_import_satisfies_check(self):
        src = (
            "from pycompss.api.task import *\n"
            "@compss.task(returns=1)\n"
            "def foo(x):\n    return x\n"
        )
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        # should NOT add a redundant import when wildcard already covers it
        self.assertNotIn("from pycompss.api.task import task", out)


# ── FixFutureResolutionRelaxation ─────────────────────────────────────────────

class TestFixFutureResolution(unittest.TestCase):
    def setUp(self):
        self.r = FixFutureResolutionRelaxation()

    def test_fixes_listcomp_result(self):
        src = "results = [f.result() for f in futures]\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertIn("compss_wait_on(futures)", out)
        self.assertNotIn(".result()", out)

    def test_fixes_listcomp_get(self):
        src = "results = [f.get() for f in tasks]\n"
        out = self.r.apply(src)
        self.assertIn("compss_wait_on(tasks)", out)
        self.assertNotIn(".get()", out)

    def test_fixes_bare_result(self):
        src = "val = future.result()\n"
        out = self.r.apply(src)
        self.assertIn("compss_wait_on(future)", out)

    def test_fixes_bare_get(self):
        src = "val = task.get()\n"
        out = self.r.apply(src)
        self.assertIn("compss_wait_on(task)", out)

    def test_adds_wait_import_when_missing(self):
        src = "results = [f.result() for f in futures]\n"
        out = self.r.apply(src)
        self.assertIn("from pycompss.api.api import compss_wait_on", out)

    def test_no_duplicate_import(self):
        src = (
            "from pycompss.api.api import compss_wait_on\n"
            "val = future.result()\n"
        )
        out = self.r.apply(src)
        self.assertEqual(out.count("from pycompss.api.api import compss_wait_on"), 1)

    def test_wildcard_import_satisfies_check(self):
        src = (
            "from pycompss.api.api import *\n"
            "val = future.result()\n"
        )
        out = self.r.apply(src)
        self.assertNotIn("from pycompss.api.api import compss_wait_on", out)

    def test_no_op_when_absent(self):
        src = "result = compute(x)\nreturn compss_wait_on(result)\n"
        self.assertIsNone(self.r.apply(src))

    def test_no_op_on_empty(self):
        self.assertIsNone(self.r.apply(""))


# ── RemoveInteractiveImportRelaxation ─────────────────────────────────────────

class TestRemoveInteractiveImport(unittest.TestCase):
    def setUp(self):
        self.r = RemoveInteractiveImportRelaxation()

    def test_removes_unused_interactive_import(self):
        src = (
            "import pycompss.interactive as ipycompss\n"
            "from pycompss.api.task import task\n"
            "\n"
            "@task(returns=1)\n"
            "def foo(x):\n    return x\n"
        )
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertNotIn("pycompss.interactive", out)
        self.assertIn("from pycompss.api.task import task", out)

    def test_no_op_when_alias_is_used(self):
        src = (
            "import pycompss.interactive as ipycompss\n"
            "ipycompss.start()\n"
        )
        self.assertIsNone(self.r.apply(src))

    def test_no_op_when_absent(self):
        src = "from pycompss.api.task import task\n"
        self.assertIsNone(self.r.apply(src))

    def test_removes_from_interactive_import(self):
        src = "from pycompss.interactive import compss_start\nimport numpy\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertNotIn("pycompss.interactive", out)
        self.assertIn("import numpy", out)

    def test_no_op_on_empty(self):
        self.assertIsNone(self.r.apply(""))


# ── FixCollectParameterRelaxation ─────────────────────────────────────────────

class TestFixCollectParameter(unittest.TestCase):
    def setUp(self):
        self.r = FixCollectParameterRelaxation()

    def test_replaces_collect_in_import(self):
        src = "from pycompss.api.parameter import INOUT, COLLECT, IN\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertIn("COLLECTION", out)
        self.assertNotIn("import COLLECT,", out)
        self.assertNotIn(", COLLECT,", out)

    def test_replaces_collect_in_decorator(self):
        src = "@task(returns=COLLECT)\ndef foo(x):\n    return x\n"
        out = self.r.apply(src)
        self.assertIn("returns=COLLECTION", out)

    def test_does_not_touch_collection(self):
        src = "from pycompss.api.parameter import COLLECTION\n"
        self.assertIsNone(self.r.apply(src))

    def test_does_not_corrupt_collection_in(self):
        src = "from pycompss.api.parameter import COLLECTION_IN, COLLECT\n"
        out = self.r.apply(src)
        self.assertIn("COLLECTION_IN", out)
        # COLLECT should be replaced but COLLECTION_IN must remain intact
        self.assertNotIn("COLLECTIONION_IN", out)

    def test_no_op_when_absent(self):
        self.assertIsNone(self.r.apply("from pycompss.api.parameter import IN, INOUT\n"))


# ── RemoveHallucinatedImportsRelaxation ───────────────────────────────────────

class TestRemoveHallucinatedImports(unittest.TestCase):
    def setUp(self):
        self.r = RemoveHallucinatedImportsRelaxation()

    # ── category 1: top-level compss.* ──────────────────────────────────────

    def test_removes_compss_management(self):
        src = "from compss.management import TaskGroup\nimport numpy\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertNotIn("compss.management", out)
        self.assertIn("import numpy", out)

    def test_removes_compss_onsched(self):
        out = self.r.apply("from compss.onsched import on_sched\n")
        self.assertIsNotNone(out)
        self.assertNotIn("compss.onsched", out)

    def test_removes_bare_import_compss(self):
        out = self.r.apply("import compss\nfrom pycompss.api.task import task\n")
        self.assertIsNotNone(out)
        self.assertNotIn("import compss\n", out)
        self.assertIn("from pycompss.api.task import task", out)

    # ── category 2: bad pycompss submodule paths ─────────────────────────────

    def test_removes_pycomps_typo(self):
        out = self.r.apply("from pycomps.api.task import task\n")
        self.assertIsNotNone(out)
        self.assertNotIn("pycomps", out)

    def test_removes_pycompss_api_parallel(self):
        out = self.r.apply("from pycompss.api.parallel import parallelize\n")
        self.assertIsNotNone(out)
        self.assertNotIn("pycompss.api.parallel", out)

    def test_removes_pycompss_api_decos(self):
        out = self.r.apply("from pycompss.api.decos import task\n")
        self.assertIsNotNone(out)
        self.assertNotIn("pycompss.api.decos", out)

    def test_removes_exaqute(self):
        out = self.r.apply("import exaqute\n")
        self.assertIsNotNone(out)
        self.assertNotIn("exaqute", out)

    # ── category 3a: hallucinated names in pycompss.api.api ─────────────────

    def test_removes_compss_type(self):
        out = self.r.apply("from pycompss.api.api import compss_type\n")
        self.assertIsNotNone(out)
        self.assertNotIn("compss_type", out)

    def test_removes_compss_close(self):
        out = self.r.apply("from pycompss.api.api import compss_close\n")
        self.assertIsNotNone(out)

    def test_keeps_valid_names_strips_bad_ones(self):
        src = "from pycompss.api.api import compss_wait_on, compss_type, compss_barrier\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertIn("compss_wait_on", out)
        self.assertIn("compss_barrier", out)
        self.assertNotIn("compss_type", out)

    def test_drops_line_when_all_names_bad(self):
        src = "from pycompss.api.api import compss_type, compss_close\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertNotIn("compss_type", out)
        self.assertNotIn("compss_close", out)

    def test_wildcard_api_import_left_alone(self):
        src = "from pycompss.api.api import *\n"
        self.assertIsNone(self.r.apply(src))

    # ── category 3b: hallucinated names from pycompss.api directly ──────────

    def test_removes_get_current_task_hallucination(self):
        src = "from pycompss.api import get_current_task_group_tasks_status_ids_names\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertNotIn("get_current_task", out)

    def test_removes_parallel_from_api(self):
        out = self.r.apply("from pycompss.api import parallel\n")
        self.assertIsNotNone(out)

    def test_removes_comp_from_api(self):
        out = self.r.apply("from pycompss.api import comp\n")
        self.assertIsNotNone(out)

    # ── category 3c: hallucinated parameter names ────────────────────────────

    def test_removes_Input_parameter(self):
        src = "from pycompss.api.parameter import IN, Input, INOUT\n"
        out = self.r.apply(src)
        self.assertIsNotNone(out)
        self.assertIn("IN", out)
        self.assertIn("INOUT", out)
        self.assertNotIn("Input", out)

    def test_removes_Range_parameter(self):
        src = "from pycompss.api.parameter import IN, Range\n"
        out = self.r.apply(src)
        self.assertIn("IN", out)
        self.assertNotIn("Range", out)

    def test_removes_auto_from_task(self):
        src = "from pycompss.api.task import task, auto\n"
        out = self.r.apply(src)
        self.assertIn("task", out)
        self.assertNotIn("auto", out)

    # ── no-op cases ──────────────────────────────────────────────────────────

    def test_no_op_on_all_valid_imports(self):
        src = (
            "from pycompss.api.task import task\n"
            "from pycompss.api.api import compss_wait_on, compss_barrier\n"
            "from pycompss.api.parameter import IN, INOUT, COLLECTION\n"
        )
        self.assertIsNone(self.r.apply(src))

    def test_no_op_on_empty(self):
        self.assertIsNone(self.r.apply(""))

    def test_no_op_on_no_imports(self):
        src = "def main(x):\n    return x\n"
        self.assertIsNone(self.r.apply(src))


# ── Registry sanity checks ───────────────────────────────────────────────────

class TestRegistry(unittest.TestCase):

    def test_all_relaxations_non_empty(self):
        self.assertGreater(len(ALL_RELAXATIONS), 0)

    def test_relaxation_map_consistent_with_list(self):
        for r in ALL_RELAXATIONS:
            self.assertIn(r.name, RELAXATION_MAP)
            self.assertIs(RELAXATION_MAP[r.name], r)

    def test_no_duplicate_names(self):
        names = [r.name for r in ALL_RELAXATIONS]
        self.assertEqual(len(names), len(set(names)))

    def test_expected_relaxations_present(self):
        names = set(RELAXATION_MAP.keys())
        for expected in [
            "rename_main",
            "fix_pycompss_imports",
            "remove_future_imports",
            "fix_task_decorator",
            "fix_future_resolution",
            "remove_interactive_import",
            "fix_collect_parameter",
            "remove_hallucinated_imports",
        ]:
            self.assertIn(expected, names)

    def test_none_return_on_unchanged_correct_code(self):
        """Every relaxation must return None when the source needs no change."""
        correct = (
            "from pycompss.api.task import task\n"
            "from pycompss.api.api import compss_wait_on\n"
            "from pycompss.api.parameter import IN, INOUT, COLLECTION\n"
            "\n"
            "@task(returns=1)\n"
            "def compute(x):\n"
            "    return x * 2\n"
            "\n"
            "def main(A):\n"
            "    result = compute(A)\n"
            "    return compss_wait_on(result)\n"
        )
        for r in ALL_RELAXATIONS:
            with self.subTest(relaxation=r.name):
                out = r.apply(correct)
                # fix_pycompss_imports replaces all pycompss imports by design
                if r.name == "fix_pycompss_imports":
                    continue
                self.assertIsNone(
                    out,
                    msg=f"{r.name} should return None on correct code but returned:\n{out}",
                )


if __name__ == "__main__":
    unittest.main()
