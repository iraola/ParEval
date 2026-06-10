"""Unit tests for driver_wrapper.py.

Coverage:
  - RunOutput._parse_output: partial output, out-of-order lines, missing fields, noise
  - _find_problem_size: exact match, hyphen/underscore normalization, substring fallback, default
  - GeneratedTextResult: did_build, did_any_run, did_all_run, are_any_valid, are_all_valid,
    best_sequential_runtime
"""
import pytest
from driver_wrapper import BuildOutput, GeneratedTextResult, RunOutput, _find_problem_size, _run_executed
from python.parallel_validation import PyCOMPSSValidator

PASS_OUT = "Time: 0.100\nBestSequential: 0.200\nValidation: PASS\n"
FAIL_OUT = "Time: 0.050\nBestSequential: 0.100\nValidation: FAIL\n"


def _run(exit_code=0, output=PASS_OUT):
    return RunOutput(exit_code, output, "")


def _build(did_build=True):
    return BuildOutput(0 if did_build else 1, "", "")


# ---------------------------------------------------------------------------
# RunOutput._parse_output
# ---------------------------------------------------------------------------

class TestRunOutputParsing:
    def test_full_pass(self):
        r = _run(output=PASS_OUT)
        assert r.is_valid is True
        assert abs(r.runtime - 0.100) < 1e-9
        assert abs(r.best_sequential_runtime - 0.200) < 1e-9

    def test_full_fail(self):
        r = _run(output=FAIL_OUT)
        assert r.is_valid is False
        assert abs(r.runtime - 0.050) < 1e-9

    def test_out_of_order_lines(self):
        out = "Validation: PASS\nBestSequential: 0.300\nTime: 0.150\n"
        r = _run(output=out)
        assert r.is_valid is True
        assert abs(r.runtime - 0.150) < 1e-9
        assert abs(r.best_sequential_runtime - 0.300) < 1e-9

    def test_missing_validation_leaves_is_valid_none(self):
        r = _run(output="Time: 0.100\nBestSequential: 0.200\n")
        assert r.is_valid is None
        assert abs(r.runtime - 0.100) < 1e-9

    def test_missing_time_leaves_runtime_none(self):
        r = _run(output="BestSequential: 0.200\nValidation: PASS\n")
        assert r.runtime is None
        assert r.is_valid is True

    def test_empty_output_all_none(self):
        r = _run(output="")
        assert r.is_valid is None
        assert r.runtime is None
        assert r.best_sequential_runtime is None

    def test_noise_lines_ignored(self):
        out = "Initialising PyCOMPSs...\nTime: 0.500\nsome warning\nValidation: PASS\nBestSequential: 1.000\n"
        r = _run(output=out)
        assert r.is_valid is True
        assert abs(r.runtime - 0.500) < 1e-9

    def test_nonzero_exit_code_with_empty_output(self):
        r = RunOutput(1, "", "process crashed")
        assert r.exit_code == 1
        assert r.is_valid is None
        assert r.runtime is None


# ---------------------------------------------------------------------------
# _find_problem_size
# ---------------------------------------------------------------------------

class TestFindProblemSize:
    SIZES = {
        "matrix_mult": {"pycompss": "(1<<10)", "omp": "(1<<12)"},
        "sort_array":  {"pycompss": "(1<<15)"},
        "vec-add":     {"pycompss": "(1<<8)"},
    }
    DEFAULT = "(1<<18)"

    def test_exact_match(self):
        assert _find_problem_size(self.SIZES, "matrix_mult", "pycompss", self.DEFAULT) == "(1<<10)"

    def test_exact_match_different_model(self):
        assert _find_problem_size(self.SIZES, "matrix_mult", "omp", self.DEFAULT) == "(1<<12)"

    def test_missing_model_in_entry_returns_default(self):
        assert _find_problem_size(self.SIZES, "sort_array", "omp", self.DEFAULT) == self.DEFAULT

    def test_missing_problem_returns_default(self):
        assert _find_problem_size(self.SIZES, "nonexistent", "pycompss", self.DEFAULT) == self.DEFAULT

    def test_hyphen_to_underscore_normalization(self):
        # Entry stored as "vec-add"; lookup by "vec_add" should still match
        assert _find_problem_size(self.SIZES, "vec_add", "pycompss", self.DEFAULT) == "(1<<8)"

    def test_substring_fallback(self):
        # "sort_array" key is a superstring of "sort"; lookup by "sort" should match
        assert _find_problem_size(self.SIZES, "sort", "pycompss", self.DEFAULT) == "(1<<15)"

    def test_empty_sizes_returns_default(self):
        assert _find_problem_size({}, "anything", "pycompss", self.DEFAULT) == self.DEFAULT


# ---------------------------------------------------------------------------
# GeneratedTextResult
# ---------------------------------------------------------------------------

class TestGeneratedTextResult:
    def _make(self, did_build=True, run_outputs=None, relaxations_applied=None):
        build = _build(did_build)
        if run_outputs is None:
            run_outputs = [_run()] if did_build else None
        return GeneratedTextResult(True, build, run_outputs, relaxations_applied)

    def test_did_build_true(self):
        assert self._make(did_build=True).did_build() is True

    def test_did_build_false(self):
        assert self._make(did_build=False).did_build() is False

    def test_did_any_run_true_when_at_least_one_zero_exit(self):
        assert self._make(run_outputs=[_run(0), _run(1)]).did_any_run() is True

    def test_did_any_run_false_when_all_nonzero(self):
        assert self._make(run_outputs=[_run(1), _run(2)]).did_any_run() is False

    def test_did_all_run_true(self):
        assert self._make(run_outputs=[_run(0), _run(0)]).did_all_run() is True

    def test_did_all_run_false_when_partial(self):
        assert self._make(run_outputs=[_run(0), _run(1)]).did_all_run() is False

    def test_are_any_valid_true(self):
        result = self._make(run_outputs=[_run(0, PASS_OUT), _run(0, FAIL_OUT)])
        assert result.are_any_valid() is True

    def test_are_any_valid_false_when_all_fail(self):
        assert self._make(run_outputs=[_run(0, FAIL_OUT)]).are_any_valid() is False

    def test_are_all_valid_true(self):
        result = self._make(run_outputs=[_run(0, PASS_OUT), _run(0, PASS_OUT)])
        assert result.are_all_valid() is True

    def test_are_all_valid_false_when_partial(self):
        result = self._make(run_outputs=[_run(0, PASS_OUT), _run(0, FAIL_OUT)])
        assert result.are_all_valid() is False

    def test_best_sequential_runtime_returns_minimum(self):
        r1 = _run(0, "Time: 0.1\nBestSequential: 0.5\nValidation: PASS\n")
        r2 = _run(0, "Time: 0.2\nBestSequential: 0.3\nValidation: PASS\n")
        result = self._make(run_outputs=[r1, r2])
        assert abs(result.best_sequential_runtime() - 0.3) < 1e-9

    def test_best_sequential_runtime_none_when_not_built(self):
        assert self._make(did_build=False).best_sequential_runtime() is None

    def test_best_sequential_runtime_none_when_no_best_seq_line(self):
        r = _run(0, "Time: 0.1\nValidation: PASS\n")
        assert self._make(run_outputs=[r]).best_sequential_runtime() is None

    def test_relaxations_applied_stored(self):
        assert self._make(relaxations_applied=["rename_main"]).relaxations_applied == ["rename_main"]

    def test_relaxations_applied_defaults_to_empty(self):
        assert self._make().relaxations_applied == []


# ---------------------------------------------------------------------------
# _run_executed (pycompss-aware "did it run")
# ---------------------------------------------------------------------------

class TestRunExecuted:
    def test_clean_exit_counts_for_any_model(self):
        assert _run_executed(_run(0, PASS_OUT)) is True
        assert _run_executed(_run(0, FAIL_OUT)) is True

    def test_nonzero_exit_does_not_count_for_non_pycompss(self):
        # e.g. cpp/omp/mpi: nonzero exit means it did not run, even if validated
        assert _run_executed(_run(1, PASS_OUT), "omp") is False

    def test_nonzero_exit_but_validated_counts_for_pycompss(self):
        # timeout on teardown after the program already validated
        assert _run_executed(_run(1, PASS_OUT), "pycompss") is True

    def test_nonzero_exit_without_verdict_does_not_count_for_pycompss(self):
        # killed before producing a Validation line (e.g. NIO worker never started)
        assert _run_executed(_run(1, "no verdict here"), "pycompss") is False

    def test_pycompss_did_run_serialization_invariant(self):
        # is_valid True must never coexist with did_run False for pycompss
        result = GeneratedTextResult(True, _build(True), [_run(1, PASS_OUT)], parallelism_model="pycompss")
        assert result.did_any_run() is True


# ---------------------------------------------------------------------------
# PyCOMPSSValidator
# ---------------------------------------------------------------------------

class TestPyCOMPSSValidator:
    def test_true_when_task_decorator_present(self):
        assert PyCOMPSSValidator().validate("@task()\ndef f(): pass\n")

    def test_true_when_task_with_parameters(self):
        assert PyCOMPSSValidator().validate("@task(returns=int)\ndef compute(x): return x\n")

    def test_false_for_sequential_python(self):
        assert not PyCOMPSSValidator().validate("def f(x):\n    return x + 1\n")

    def test_false_for_empty_string(self):
        assert not PyCOMPSSValidator().validate("")

    def test_false_when_only_pycompss_import_present(self):
        code = "from pycompss.api.task import task\ndef f(): pass\n"
        assert not PyCOMPSSValidator().validate(code)

    def test_true_with_import_and_decorator(self):
        code = (
            "from pycompss.api.task import task\n"
            "@task(returns=list)\n"
            "def compute(data): return data\n"
        )
        assert PyCOMPSSValidator().validate(code)
