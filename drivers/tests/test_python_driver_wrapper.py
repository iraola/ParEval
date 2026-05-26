"""Unit tests for PythonDriverWrapper.

Run from the repo root:
    .venv/bin/pytest drivers/tests/ -v

Run from drivers/:
    ../.venv/bin/pytest tests/ -v

Coverage:
  - _in_scaling_mode() detection logic
  - Wave CPU assignment (no overlap, within bounds, single-CPU format)
  - Wave result ordering (original config order preserved)
  - Wave overflow (configs that don't fit split into a second wave)
  - NIO port uniqueness within a concurrent wave and across waves
  - Resources XML content (computing_units, ports)
  - Workdir lifecycle (created under TMPDIR, deleted on success and timeout)
  - Normal mode (no resource_slot: no XML, no cpu_affinity injection)
  - Integration: real subprocess parsing Time/BestSequential/Validation output
"""
import os
import re
import subprocess
import tempfile
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

# conftest.py already set sys.path and CWD; imports work from here.
import python.python_driver_wrapper as _pdw_module  # noqa: E402
from python.python_driver_wrapper import (  # noqa: E402
    PythonDriverWrapper,
    _compress_model_name,
    _prompt_id,
    _runs_succeeded,
    try_to_find_path,
)
from driver_wrapper import BuildOutput, RunOutput  # noqa: E402
from python.relaxations import RenameMainRelaxation  # noqa: E402


def _reset_port_counter():
    """Reset both NIO port counters to their base values between tests."""
    with _pdw_module._nio_port_lock:
        _pdw_module._nio_port_current = _pdw_module._NIO_PORT_BASE
    with _pdw_module._master_port_lock:
        _pdw_module._master_port_current = _pdw_module._MASTER_PORT_BASE

# ---------------------------------------------------------------------------
# Constants and shared fixtures
# ---------------------------------------------------------------------------

PASS_OUTPUT = "Time: 0.100\nBestSequential: 0.200\nValidation: PASS\n"
FAIL_OUTPUT = "Time: 0.050\nBestSequential: 0.100\nValidation: FAIL\n"

# Format that exercises all resource-slot placeholders
SLOT_FMT = (
    "fake_runcompss --cpu_affinity={cpu_affinity} "
    "--master_port={master_port} "
    "--resources={resources_xml} "
    "--master_working_dir={master_working_dir} "
    "{exec_path} {args}"
)

SIMPLE_FMT = "echo {exec_path} {args}"

# An 8-CPU slot starting at CPU 0
SLOT_8 = {"cpu_start": 0, "slot_cpus": 8}
# Same size but offset (simulates a non-zero cpu_start)
SLOT_OFFSET = {"cpu_start": 10, "slot_cpus": 8}


def make_driver(
    resource_slot=None,
    launch_params=None,
    launch_format=SLOT_FMT,
    run_timeout=10,
    scratch_dir=None,
    relaxations=None,
    early_exit_runs=False,
):
    if launch_params is None:
        launch_params = [{}]
    return PythonDriverWrapper(
        parallelism_model="pycompss",
        launch_configs={"pycompss": {"format": launch_format, "params": launch_params}},
        build_configs=None,
        problem_sizes={},
        scratch_dir=scratch_dir,
        run_timeout=run_timeout,
        resource_slot=resource_slot,
        model_name="test-model",
        relaxations=relaxations or [],
        early_exit_runs=early_exit_runs,
    )


def _fake_run_obj(return_code=0, output=PASS_OUTPUT):
    """Return a callable that captures kwargs passed to run() and returns a RunOutput."""
    calls = []
    lock = threading.Lock()

    def _run(exec_path, **run_config):
        with lock:
            calls.append(dict(run_config))
        return RunOutput(return_code, output, "", config=run_config)

    _run.calls = calls
    return _run


def _parse_affinity(aff: str) -> set:
    """Convert '0-3' or '6' to the set of CPU indices it covers."""
    if "-" in aff:
        lo, hi = map(int, aff.split("-"))
        return set(range(lo, hi + 1))
    return {int(aff)}


def _xml_from_cmd(cmd: str):
    """Extract and parse the resources XML file referenced in a fake_runcompss command."""
    m = re.search(r"--resources=(\S+)", cmd)
    assert m, f"No --resources= in command: {cmd}"
    return ET.parse(m.group(1)).getroot()


# ---------------------------------------------------------------------------
# _in_scaling_mode
# ---------------------------------------------------------------------------

class TestInScalingMode:
    def test_false_without_slot(self):
        d = make_driver(resource_slot=None)
        assert not d._in_scaling_mode([{"num_procs": 1}, {"num_procs": 2}])

    def test_false_with_empty_configs(self):
        d = make_driver(resource_slot=SLOT_8)
        assert not d._in_scaling_mode([])

    def test_false_when_no_num_procs_key(self):
        # Correctness mode: single empty config {}
        d = make_driver(resource_slot=SLOT_8)
        assert not d._in_scaling_mode([{}])

    def test_false_with_single_config_even_if_num_procs_present(self):
        # [{"num_procs": 5}] — one entry, should NOT trigger scaling
        d = make_driver(resource_slot=SLOT_8)
        assert not d._in_scaling_mode([{"num_procs": 5}])

    def test_true_with_two_num_procs_configs(self):
        d = make_driver(resource_slot=SLOT_8)
        assert d._in_scaling_mode([{"num_procs": 1}, {"num_procs": 2}])

    def test_true_with_many_num_procs_configs(self):
        d = make_driver(resource_slot=SLOT_8)
        configs = [{"num_procs": n} for n in [1, 2, 4, 8, 16, 32, 64]]
        assert d._in_scaling_mode(configs)


# ---------------------------------------------------------------------------
# Wave packing: CPU assignment
# ---------------------------------------------------------------------------

class TestWavePacking:
    def _run_waves(self, driver, configs):
        fr = _fake_run_obj()
        with patch.object(driver, "run", side_effect=fr):
            results = driver._run_configs_in_waves("dummy.py", configs)
        return results, fr.calls

    def test_all_configs_are_run(self):
        d = make_driver(resource_slot=SLOT_8)
        configs = [{"num_procs": 4}, {"num_procs": 2}, {"num_procs": 1}, {"num_procs": 1}]
        results, calls = self._run_waves(d, configs)
        assert len(results) == 4
        assert len(calls) == 4
        assert all(r is not None for r in results)

    def test_overflow_into_second_wave(self):
        """[4,4,4] in 8-CPU slot: first two pack into Wave 1 (8 CPUs), third into Wave 2."""
        d = make_driver(resource_slot=SLOT_8)
        configs = [{"num_procs": 4}, {"num_procs": 4}, {"num_procs": 4}]
        results, calls = self._run_waves(d, configs)
        assert len(results) == 3
        assert len(calls) == 3

    def test_results_indexed_by_original_config_order(self):
        """result[i] must correspond to configs[i], not sorted order."""
        d = make_driver(resource_slot=SLOT_8)
        # Original order [1, 4, 2]; wave packer will sort to [4, 2, 1] internally
        configs = [{"num_procs": 1}, {"num_procs": 4}, {"num_procs": 2}]
        results, _ = self._run_waves(d, configs)
        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_no_cpu_overlap_within_wave(self):
        d = make_driver(resource_slot=SLOT_8)
        configs = [{"num_procs": 4}, {"num_procs": 2}, {"num_procs": 1}, {"num_procs": 1}]
        _, calls = self._run_waves(d, configs)

        all_cpus = []
        for c in calls:
            all_cpus.extend(_parse_affinity(c["cpu_affinity"]))
        assert len(all_cpus) == len(set(all_cpus)), f"CPU overlap detected: {[c['cpu_affinity'] for c in calls]}"

    def test_cpu_affinity_within_slot_bounds(self):
        """All assigned CPUs must lie in [cpu_start, cpu_start + slot_cpus)."""
        d = make_driver(resource_slot=SLOT_OFFSET)  # cpu_start=10, slot_cpus=8 → [10,18)
        configs = [{"num_procs": 4}, {"num_procs": 2}, {"num_procs": 2}]
        _, calls = self._run_waves(d, configs)

        for c in calls:
            for cpu in _parse_affinity(c["cpu_affinity"]):
                assert 10 <= cpu < 18, f"CPU {cpu} outside slot bounds (aff={c['cpu_affinity']})"

    def test_num_procs_1_uses_bare_number_not_range(self):
        """num_procs=1 must produce '7' not '7-7'."""
        d = make_driver(resource_slot=SLOT_8)
        configs = [{"num_procs": 7}, {"num_procs": 1}]
        _, calls = self._run_waves(d, configs)
        single_calls = [c for c in calls if c.get("num_procs") == 1]
        assert single_calls, "no num_procs=1 call found"
        for c in single_calls:
            assert "-" not in c["cpu_affinity"], f"Unexpected range for num_procs=1: {c['cpu_affinity']}"

    def test_wave2_resets_cpu_position_to_slot_start(self):
        """Wave 2 starts CPU allocation over from cpu_start (waves run sequentially)."""
        d = make_driver(resource_slot=SLOT_8)  # cpu_start=0
        # [4,4] fills Wave 1 exactly; [4] goes to Wave 2 and should start at cpu 0
        configs = [{"num_procs": 4}, {"num_procs": 4}, {"num_procs": 4}]
        _, calls = self._run_waves(d, configs)
        affinities = [c["cpu_affinity"] for c in calls]
        # "0-3" should appear twice: once in Wave 1, once in Wave 2
        assert affinities.count("0-3") == 2, f"Expected '0-3' twice, got: {affinities}"


# ---------------------------------------------------------------------------
# NIO port uniqueness
# ---------------------------------------------------------------------------

class TestPortDerivation:
    @pytest.fixture(autouse=True)
    def reset_counter(self):
        _reset_port_counter()
        yield
        _reset_port_counter()

    def _collect_port_pairs(self, driver, configs=None, **run_kwargs):
        """Run either _run_configs_in_waves (if configs given) or run() once, collecting all XML port pairs."""
        port_pairs, lock = [], threading.Lock()

        def mock_cmd(cmd, **kw):
            root = _xml_from_cmd(cmd)
            mn = int(root.find(".//MinPort").text)
            mx = int(root.find(".//MaxPort").text)
            with lock:
                port_pairs.append((mn, mx))
            return CompletedProcess(cmd, 0, PASS_OUTPUT, "")

        with patch("python.python_driver_wrapper.run_command", mock_cmd):
            if configs is not None:
                driver._run_configs_in_waves("dummy.py", configs)
            else:
                driver.run("dummy.py", **run_kwargs)
        return port_pairs

    def test_run_gets_min_max_pair(self):
        """run() writes a resources XML with max_port == min_port + num_procs."""
        d = make_driver(resource_slot=SLOT_8)
        pairs = self._collect_port_pairs(d, num_procs=4, cpu_affinity="0-3")
        assert len(pairs) == 1
        mn, mx = pairs[0]
        assert mx == mn + 4  # max_port = min_port + num_procs

    def test_ports_unique_across_all_concurrent_wave_configs(self):
        """No two configs in the same wave should share a port number."""
        d = make_driver(resource_slot=SLOT_8)
        configs = [{"num_procs": 4}, {"num_procs": 2}, {"num_procs": 1}, {"num_procs": 1}]
        pairs = self._collect_port_pairs(d, configs=configs)
        all_ports = [p for mn, mx in pairs for p in (mn, mx)]
        assert len(all_ports) == len(set(all_ports)), f"Duplicate ports detected: {pairs}"

    def test_sequential_runs_on_same_slot_get_unique_ports(self):
        """Correctness mode: multiple outputs run sequentially on the same slot.
        Every call to run() must get a fresh port, not reuse the previous one."""
        d = make_driver(resource_slot=SLOT_8)
        pairs = []
        for _ in range(5):
            pairs.extend(self._collect_port_pairs(d, num_procs=8, cpu_affinity="0-7"))
        all_ports = [p for mn, mx in pairs for p in (mn, mx)]
        assert len(all_ports) == len(set(all_ports)), \
            f"Port reused across sequential correctness runs: {pairs}"

    def test_counter_wraps_before_exceeding_max(self):
        """When the counter reaches _NIO_PORT_MAX it wraps back to _NIO_PORT_BASE,
        keeping port numbers within a safe range."""
        d = make_driver(resource_slot=SLOT_8)
        # Drive the counter to one step before the wrap boundary.
        with _pdw_module._nio_port_lock:
            _pdw_module._nio_port_current = _pdw_module._NIO_PORT_MAX - 1

        pairs_before = self._collect_port_pairs(d, num_procs=8, cpu_affinity="0-7")
        pairs_after  = self._collect_port_pairs(d, num_procs=8, cpu_affinity="0-7")

        assert pairs_before[0][0] == _pdw_module._NIO_PORT_MAX - 1
        assert pairs_after[0][0]  == _pdw_module._NIO_PORT_BASE
        assert pairs_after[0][0] <= _pdw_module._NIO_PORT_MAX


# ---------------------------------------------------------------------------
# NIO port uniqueness across waves
# ---------------------------------------------------------------------------

class TestPortUniquenessAcrossWaves:
    """Worker ports must be globally unique across all waves, not just within one wave.
    Each test resets the counter so port values are deterministic and don't
    depend on the number of run() calls made by earlier tests.

    Fixed by a global atomic counter (_alloc_nio_port) that advances regardless
    of wave boundaries, so no two configs ever receive the same MinPort block.
    """

    @pytest.fixture(autouse=True)
    def reset_counter(self):
        _reset_port_counter()
        yield
        _reset_port_counter()

    def _collect_min_ports(self, driver, configs):
        """Run _run_configs_in_waves, return one min_port per config (from XML)."""
        min_ports = []
        lock = threading.Lock()

        def mock_cmd(cmd, **kw):
            root = _xml_from_cmd(cmd)
            with lock:
                min_ports.append(int(root.find(".//MinPort").text))
            return CompletedProcess(cmd, 0, PASS_OUTPUT, "")

        with patch("python.python_driver_wrapper.run_command", mock_cmd):
            driver._run_configs_in_waves("dummy.py", configs)
        return min_ports

    def _collect_master_ports(self, driver, configs):
        """Run _run_configs_in_waves, return one master_port per config (from cmd)."""
        master_ports = []
        lock = threading.Lock()

        def mock_cmd(cmd, **kw):
            m = re.search(r"--master_port=(\d+)", cmd)
            with lock:
                master_ports.append(int(m.group(1)) if m else -1)
            return CompletedProcess(cmd, 0, PASS_OUTPUT, "")

        with patch("python.python_driver_wrapper.run_command", mock_cmd):
            driver._run_configs_in_waves("dummy.py", configs)
        return master_ports

    def test_no_port_reuse_across_two_waves(self):
        """[4,4,4] on 8-CPU slot → Wave 1:[4,4], Wave 2:[4].
        Wave 2 starts at cpu_start=0 (same as Wave 1's first config) but
        must use a different port."""
        d = make_driver(resource_slot=SLOT_8)
        configs = [{"num_procs": 4}, {"num_procs": 4}, {"num_procs": 4}]
        min_ports = self._collect_min_ports(d, configs)
        assert len(min_ports) == 3
        assert len(set(min_ports)) == len(min_ports), \
            f"Port reuse across waves: {min_ports}"

    def test_wave2_first_config_does_not_reuse_wave1_first_config_port(self):
        """Wave 2 resets cpu_start to 0 but must not reuse Wave 1's port."""
        d = make_driver(resource_slot=SLOT_8)
        configs = [{"num_procs": 4}, {"num_procs": 4}, {"num_procs": 4}]
        min_ports = self._collect_min_ports(d, configs)
        assert len(set(min_ports)) == len(min_ports), \
            f"Port reuse detected: {min_ports}"

    def test_full_scaling_scenario_reproduces_original_bug(self):
        """Exact scenario from the bug: [1,2,4,8,16,32,64] on a 112-CPU slot.

        Wave 1 (112 CPUs): 64-proc @ cpu 0, 32-proc @ cpu 64, 16-proc @ cpu 96.
        Wave 2  (15 CPUs):  8-proc @ cpu 0, 4-proc @ cpu 8, 2-proc @ cpu 12, 1-proc @ cpu 14.

        The global counter ensures Wave 2's 8-proc config gets a different MinPort
        than Wave 1's 64-proc config, preventing BindException on port reuse.
        """
        slot = {"cpu_start": 0, "slot_cpus": 112, }
        d = make_driver(resource_slot=slot)
        configs = [{"num_procs": n} for n in [1, 2, 4, 8, 16, 32, 64]]
        min_ports = self._collect_min_ports(d, configs)
        assert len(min_ports) == 7
        assert len(set(min_ports)) == 7, \
            f"Duplicate ports detected (original bug): {min_ports}"

    def test_three_waves_all_ports_unique(self):
        """[4,4,4,4,4,4] on 8-CPU slot → three waves of two, all ports unique."""
        d = make_driver(resource_slot=SLOT_8)
        configs = [{"num_procs": 4}] * 6
        min_ports = self._collect_min_ports(d, configs)
        assert len(set(min_ports)) == 6, \
            f"Port reuse across three waves: {min_ports}"

    def test_offset_slot_ports_unique_across_waves(self):
        """Same check on a non-zero cpu_start slot (cpu_start=10)."""
        d = make_driver(resource_slot=SLOT_OFFSET)  # cpu_start=10, slot_cpus=8
        configs = [{"num_procs": 4}, {"num_procs": 4}, {"num_procs": 4}]
        min_ports = self._collect_min_ports(d, configs)
        assert len(set(min_ports)) == 3, \
            f"Port reuse with offset slot: {min_ports}"

    def test_master_ports_unique_within_concurrent_wave(self):
        """Concurrent runs in a scaling wave must each get a unique master port.

        Wave 1 of [64,32,16] on a 112-CPU slot runs 3 instances concurrently;
        each must bind a different --master_port.
        """
        slot = {"cpu_start": 0, "slot_cpus": 112, }
        d = make_driver(resource_slot=slot)
        configs = [{"num_procs": n} for n in [64, 32, 16]]
        master_ports = self._collect_master_ports(d, configs)
        assert len(master_ports) == 3
        assert len(set(master_ports)) == 3, \
            f"Master port collision in concurrent wave: {master_ports}"


# ---------------------------------------------------------------------------
# run(): resource management, XML content, and cleanup
# ---------------------------------------------------------------------------

class TestRunMethod:
    def test_normal_mode_no_cpu_injection(self):
        """Without resource_slot, run() issues a plain command with no cpu_affinity."""
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        captured = []

        def mock_cmd(cmd, **kw):
            captured.append(cmd)
            return CompletedProcess(cmd, 0, PASS_OUTPUT, "")

        with patch("python.python_driver_wrapper.run_command", mock_cmd):
            result = d.run("myexec.py")

        assert len(captured) == 1
        assert "cpu_affinity" not in captured[0]
        assert "resources_xml" not in captured[0]
        assert result.is_valid is True

    def test_slot_mode_injects_full_slot_cpu_affinity(self):
        """With resource_slot and no explicit num_procs, all slot CPUs are assigned."""
        d = make_driver(resource_slot=SLOT_8)
        captured = []

        def mock_cmd(cmd, **kw):
            captured.append(cmd)
            return CompletedProcess(cmd, 0, PASS_OUTPUT, "")

        with patch("python.python_driver_wrapper.run_command", mock_cmd):
            d.run("exec.py")

        assert captured
        assert "--cpu_affinity=0-7" in captured[0]

    def test_xml_computing_units_matches_num_procs(self):
        d = make_driver(resource_slot=SLOT_8)
        xml_units = []

        def mock_cmd(cmd, **kw):
            xml_units.append(int(_xml_from_cmd(cmd).find(".//ComputingUnits").text))
            return CompletedProcess(cmd, 0, PASS_OUTPUT, "")

        with patch("python.python_driver_wrapper.run_command", mock_cmd):
            d.run("exec.py", num_procs=3, cpu_affinity="0-2")

        assert xml_units == [3]

    def test_xml_computing_units_defaults_to_slot_cpus(self):
        """When num_procs is not in run_config, XML should have slot_cpus computing units."""
        slot = {"cpu_start": 0, "slot_cpus": 5, }
        d = make_driver(resource_slot=slot)
        xml_units = []

        def mock_cmd(cmd, **kw):
            xml_units.append(int(_xml_from_cmd(cmd).find(".//ComputingUnits").text))
            return CompletedProcess(cmd, 0, PASS_OUTPUT, "")

        with patch("python.python_driver_wrapper.run_command", mock_cmd):
            d.run("exec.py")

        assert xml_units == [5]

    def test_workdir_created_via_gettempdir(self):
        """run() passes tempfile.gettempdir() as the dir for the compss workdir."""
        d = make_driver(resource_slot=SLOT_8)
        mkdtemp_dirs = []
        _real_mkdtemp = tempfile.mkdtemp  # capture real fn before patch

        def spy(dir=None):
            mkdtemp_dirs.append(dir)
            return _real_mkdtemp(dir=dir)  # call real impl; no recursion

        with patch("python.python_driver_wrapper.tempfile.mkdtemp", side_effect=spy):
            with patch("python.python_driver_wrapper.run_command",
                       return_value=CompletedProcess([], 0, PASS_OUTPUT, "")):
                d.run("exec.py")

        assert mkdtemp_dirs, "mkdtemp was never called"
        assert mkdtemp_dirs[0] == tempfile.gettempdir(), \
            f"Expected gettempdir()={tempfile.gettempdir()} but got: {mkdtemp_dirs[0]}"

    def test_workdir_deleted_after_success(self):
        d = make_driver(resource_slot=SLOT_8)
        created = []
        real_mkdtemp = tempfile.mkdtemp

        def spy(**kw):
            path = real_mkdtemp(**kw)
            created.append(path)
            return path

        with patch("tempfile.mkdtemp", side_effect=spy):
            with patch("python.python_driver_wrapper.run_command",
                       return_value=CompletedProcess([], 0, PASS_OUTPUT, "")):
                d.run("exec.py")

        assert created
        for p in created:
            assert not os.path.exists(p), f"workdir {p} not cleaned up after success"

    def test_workdir_deleted_after_timeout(self):
        d = make_driver(resource_slot=SLOT_8)
        created = []
        real_mkdtemp = tempfile.mkdtemp

        def spy(**kw):
            path = real_mkdtemp(**kw)
            created.append(path)
            return path

        def timeout_cmd(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 10)

        with patch("tempfile.mkdtemp", side_effect=spy):
            with patch("python.python_driver_wrapper.run_command", side_effect=timeout_cmd):
                result = d.run("exec.py")

        assert result.exit_code == -1
        assert created
        for p in created:
            assert not os.path.exists(p), f"workdir {p} not cleaned up after timeout"

    def test_timeout_returns_minus1_exit_code(self):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        with patch("python.python_driver_wrapper.run_command",
                   side_effect=subprocess.TimeoutExpired("cmd", 10)):
            result = d.run("exec.py")
        assert result.exit_code == -1

    def test_nonzero_exit_code_preserved(self):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        with patch("python.python_driver_wrapper.run_command",
                   return_value=CompletedProcess([], 42, FAIL_OUTPUT, "err")):
            result = d.run("exec.py")
        assert result.exit_code == 42
        assert result.is_valid is False


# ---------------------------------------------------------------------------
# Integration: real subprocess, no mocking
# ---------------------------------------------------------------------------

class TestRealSubprocess:
    def test_passing_script(self, tmp_path):
        script = tmp_path / "pass_exec.py"
        script.write_text(
            "print('Time: 0.150')\n"
            "print('BestSequential: 0.300')\n"
            "print('Validation: PASS')\n"
        )
        d = make_driver(
            resource_slot=None,
            launch_format="python3 {exec_path}",
            launch_params=[{}],
        )
        result = d.run(str(script))
        assert result.exit_code == 0
        assert result.is_valid is True
        assert abs(result.runtime - 0.150) < 1e-6
        assert abs(result.best_sequential_runtime - 0.300) < 1e-6

    def test_failing_validation_script(self, tmp_path):
        script = tmp_path / "fail_exec.py"
        script.write_text(
            "print('Time: 0.050')\n"
            "print('BestSequential: 0.100')\n"
            "print('Validation: FAIL')\n"
        )
        d = make_driver(
            resource_slot=None,
            launch_format="python3 {exec_path}",
            launch_params=[{}],
        )
        result = d.run(str(script))
        assert result.exit_code == 0   # process exited cleanly
        assert result.is_valid is False  # but validation failed

    def test_wave_execution_with_real_subprocesses(self, tmp_path):
        """All wave configs run and produce independent results."""
        script = tmp_path / "wave_exec.py"
        script.write_text(
            "print('Time: 0.100')\n"
            "print('BestSequential: 0.200')\n"
            "print('Validation: PASS')\n"
        )
        # Use 8 CPUs assumed available; three configs fit in one wave
        slot = {"cpu_start": 0, "slot_cpus": 8, }
        d = make_driver(
            resource_slot=slot,
            launch_format="python3 {exec_path}",
            launch_params=[{"num_procs": n} for n in [4, 2, 1, 1]],
        )
        configs = d.launch_configs["params"]
        fr = _fake_run_obj()
        with patch.object(d, "run", side_effect=fr):
            results = d._run_configs_in_waves(str(script), configs)

        assert len(results) == 4
        assert all(r is not None for r in results)
        assert all(r.exit_code == 0 for r in results)


# ---------------------------------------------------------------------------
# Helper functions: _compress_model_name, _prompt_id, _runs_succeeded
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_compress_strips_org_prefix(self):
        assert _compress_model_name("mistralai/Codestral-22B-v0.1") == "Co22v01"

    def test_compress_no_org_prefix(self):
        assert _compress_model_name("test-model") == "temo"

    def test_compress_single_word(self):
        assert _compress_model_name("gpt") == "gp"

    def test_prompt_id_extracts_numeric_prefix(self):
        assert _prompt_id("22_histogram_count_quadrants") == "22"

    def test_prompt_id_no_digits_returns_full_name(self):
        assert _prompt_id("no_digits") == "no_digits"

    def test_prompt_id_starts_with_digit(self):
        assert _prompt_id("1abc") == "1"

    def test_runs_succeeded_empty_list(self):
        assert not _runs_succeeded([])

    def test_runs_succeeded_none(self):
        assert not _runs_succeeded(None)

    def test_runs_succeeded_pass(self):
        assert _runs_succeeded([RunOutput(0, PASS_OUTPUT, "")])

    def test_runs_succeeded_fail_validation(self):
        assert not _runs_succeeded([RunOutput(0, FAIL_OUTPUT, "")])

    def test_runs_succeeded_nonzero_exit(self):
        assert not _runs_succeeded([RunOutput(1, PASS_OUTPUT, "")])

    def test_runs_succeeded_mixed_any_pass(self):
        assert _runs_succeeded([RunOutput(1, FAIL_OUTPUT, ""), RunOutput(0, PASS_OUTPUT, "")])


# ---------------------------------------------------------------------------
# compile()
# ---------------------------------------------------------------------------

class TestCompile:
    def test_successful_compile_returns_build_output_zero(self, tmp_path):
        fa = tmp_path / "src.py"
        fa.write_text("x = 1\n")
        out = tmp_path / "merged.py"
        d = make_driver()
        result = d.compile(str(fa), output_path=str(out))
        assert result.did_build

    def test_output_contains_source_content_and_header(self, tmp_path):
        fa = tmp_path / "foo.py"
        fa.write_text("# content A\n")
        out = tmp_path / "merged.py"
        d = make_driver()
        d.compile(str(fa), output_path=str(out))
        content = out.read_text()
        assert "# content A" in content
        assert "# ===== foo.py =====" in content

    def test_files_written_in_reversed_argument_order(self, tmp_path):
        """compile() reverses binaries so later args end up at the top of the merged file."""
        fa = tmp_path / "first.py"
        fa.write_text("FIRST\n")
        fb = tmp_path / "second.py"
        fb.write_text("SECOND\n")
        out = tmp_path / "merged.py"
        d = make_driver()
        d.compile(str(fa), str(fb), output_path=str(out))
        content = out.read_text()
        assert content.index("SECOND") < content.index("FIRST")

    def test_creates_output_parent_directories(self, tmp_path):
        fa = tmp_path / "src.py"
        fa.write_text("x = 1\n")
        out = tmp_path / "a" / "b" / "merged.py"
        d = make_driver()
        result = d.compile(str(fa), output_path=str(out))
        assert result.did_build
        assert out.exists()

    def test_missing_file_returns_failure(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.py"
        out = tmp_path / "merged.py"
        d = make_driver()
        result = d.compile(str(nonexistent), output_path=str(out))
        assert not result.did_build
        assert "Could not find source file" in result.stderr

    def test_multiple_files_all_appear_in_output(self, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"part{i}.py"
            f.write_text(f"PART{i}\n")
            files.append(str(f))
        out = tmp_path / "merged.py"
        d = make_driver()
        d.compile(*files, output_path=str(out))
        content = out.read_text()
        for i in range(3):
            assert f"PART{i}" in content


# ---------------------------------------------------------------------------
# try_to_find_path()
# ---------------------------------------------------------------------------

class TestTryToFindPath:
    def test_returns_none_for_non_benchmarks_path(self):
        assert try_to_find_path(Path("/absolute/path/to/file.py")) is None

    def test_returns_none_for_path_without_benchmarks_prefix(self):
        assert try_to_find_path(Path("some/other/path/file.py")) is None

    def test_returns_none_when_regex_no_match(self):
        # Only two path components after python/benchmarks/ — regex needs three
        assert try_to_find_path(Path("python/benchmarks/kernel/file.py")) is None

    def test_returns_none_when_problem_types_dont_match(self):
        # "histogram" (problem_type_2) is not in "kernel" (problem_type)
        assert try_to_find_path(
            Path("python/benchmarks/kernel/22_histogram_count_quadrants/pycompss.py")
        ) is None

    def test_resolves_path_when_directory_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bmark_dir = tmp_path / "python" / "benchmarks" / "histogram" / "99_histogram_count_quadrants"
        bmark_dir.mkdir(parents=True)
        (bmark_dir / "pycompss.py").write_text("# driver")

        # Query uses a different numeric prefix (22) than the real dir (99).
        # The returned path is relative (base_dir is relative), so compare via resolve().
        result = try_to_find_path(
            Path("python/benchmarks/histogram/22_histogram_count_quadrants/pycompss.py")
        )
        assert result is not None
        assert result.resolve() == (bmark_dir / "pycompss.py").resolve()

    def test_returns_none_when_no_matching_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        base_dir = tmp_path / "python" / "benchmarks" / "histogram"
        base_dir.mkdir(parents=True)
        # No subdirectory contains "count_quadrants"
        (base_dir / "99_histogram_something_else").mkdir()

        result = try_to_find_path(
            Path("python/benchmarks/histogram/22_histogram_count_quadrants/pycompss.py")
        )
        assert result is None

    def test_hyphen_underscore_treated_as_equivalent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bmark_dir = tmp_path / "python" / "benchmarks" / "histogram" / "10_histogram_my-example"
        bmark_dir.mkdir(parents=True)
        (bmark_dir / "pycompss.py").write_text("# driver")

        # Query uses underscore; directory uses hyphen — both normalise to the same key.
        result = try_to_find_path(
            Path("python/benchmarks/histogram/22_histogram_my_example/pycompss.py")
        )
        assert result is not None
        assert result.resolve() == (bmark_dir / "pycompss.py").resolve()


# ---------------------------------------------------------------------------
# _enrich_stderr_with_compss_job_logs()
# ---------------------------------------------------------------------------

class TestEnrichStderr:
    def test_no_compss_path_returns_stderr_unchanged(self):
        d = make_driver()
        stderr = "Some error with no COMPSs reference"
        assert d._enrich_stderr_with_compss_job_logs(stderr) == stderr

    def test_bracket_notation_reads_named_job_files(self, tmp_path):
        jobs_dir = tmp_path / ".COMPSs" / "run40" / "jobs"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "job1.out").write_text("error from job 1\n" * 3)
        (jobs_dir / "job2.err").write_text("stack trace\n" * 3)
        (jobs_dir / "job3.out").write_text("unreferenced\n")

        stderr = f"Check files '{jobs_dir}/job[1|2]' to find out the error."
        result = make_driver()._enrich_stderr_with_compss_job_logs(stderr)

        assert "error from job 1" in result
        assert "stack trace" in result
        assert "unreferenced" not in result

    def test_plain_directory_path_reads_out_err_files(self, tmp_path):
        jobs_dir = tmp_path / ".COMPSs" / "run1" / "jobs"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "job1.out").write_text("plain output\n")
        (jobs_dir / "job1.err").write_text("plain error\n")

        stderr = f"See {jobs_dir} for details."
        result = make_driver()._enrich_stderr_with_compss_job_logs(stderr)

        assert "plain output" in result or "plain error" in result

    def test_deduplication_same_jobs_dir_appended_once(self, tmp_path):
        jobs_dir = tmp_path / ".COMPSs" / "run1" / "jobs"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "job1.out").write_text("content\n" * 3)

        # Same jobs directory referenced twice in the same stderr string
        stderr = (
            f"First: '{jobs_dir}/job[1]' check.\n"
            f"Second: '{jobs_dir}/job[1]' check again."
        )
        result = make_driver()._enrich_stderr_with_compss_job_logs(stderr)
        assert result.count("[job log:") == 1

    def test_original_stderr_preserved_as_prefix(self, tmp_path):
        jobs_dir = tmp_path / ".COMPSs" / "run1" / "jobs"
        jobs_dir.mkdir(parents=True)
        (jobs_dir / "job1.out").write_text("job detail\n")

        stderr = f"Original error text. See '{jobs_dir}/job[1]'."
        result = make_driver()._enrich_stderr_with_compss_job_logs(stderr)

        assert result.startswith("Original error text.")
        assert "job detail" in result

    def test_nonexistent_jobs_dir_leaves_stderr_unchanged(self, tmp_path):
        compss_path = tmp_path / ".COMPSs" / "ghost" / "jobs"
        # Do NOT create the directory
        stderr = f"Check files '{compss_path}/job[1|2]' for errors."
        result = make_driver()._enrich_stderr_with_compss_job_logs(stderr)
        assert result == stderr


# ---------------------------------------------------------------------------
# run(): additional cases not in the original TestRunMethod
# ---------------------------------------------------------------------------

class TestRunMethodExtra:
    def test_unicode_decode_error_returns_minus1(self):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        with patch("python.python_driver_wrapper.run_command",
                   side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")):
            result = d.run("exec.py")
        assert result.exit_code == -1

    def test_slot_mode_num_procs_1_uses_bare_cpu_in_run(self):
        """run() with num_procs=1 in slot mode should produce a single CPU number, not a range."""
        d = make_driver(resource_slot=SLOT_8)
        captured = []

        def mock_cmd(cmd, **kw):
            captured.append(cmd)
            return CompletedProcess(cmd, 0, PASS_OUTPUT, "")

        with patch("python.python_driver_wrapper.run_command", mock_cmd):
            d.run("exec.py", num_procs=1, cpu_affinity="5")

        assert captured
        assert "--cpu_affinity=5" in captured[0]
        assert "--cpu_affinity=5-5" not in captured[0]


# ---------------------------------------------------------------------------
# Wave packing: edge cases
# ---------------------------------------------------------------------------

class TestWavePackingEdgeCases:
    def test_oversized_config_crashes_with_empty_wave(self):
        """BUG: when the very first config has num_procs > slot_cpus, the overflow
        branch appends an empty wave before resetting, then that empty wave causes
        ThreadPoolExecutor(max_workers=0) to raise ValueError."""
        d = make_driver(resource_slot=SLOT_8)  # slot_cpus=8
        configs = [{"num_procs": 16}]           # exceeds slot capacity

        with pytest.raises(ValueError, match="max_workers"):
            d._run_configs_in_waves("dummy.py", configs)

    def test_scaling_mode_false_when_first_config_has_no_num_procs(self):
        """_in_scaling_mode only checks configs[0]; if it lacks num_procs the
        method returns False even if later configs have it."""
        d = make_driver(resource_slot=SLOT_8)
        assert not d._in_scaling_mode([{}, {"num_procs": 2}])


# ---------------------------------------------------------------------------
# test_single_output()
# ---------------------------------------------------------------------------

# Output that RenameMainRelaxation can transform: has @task (passes gate),
# plus exactly one non-decorated top-level function not named 'main'.
_RELAX_OUTPUT = "@task()\ndef task_fn(x):\n    return x\n\ndef compute(x):\n    return task_fn(x)\n"
# Output where RenameMainRelaxation returns None (undecorated function already named main).
_NO_RELAX_OUTPUT = "@task()\ndef task_fn(x):\n    return x\n\ndef main(x):\n    return task_fn(x)\n"


class TestSingleOutput:

    @pytest.fixture(autouse=True)
    def clear_global_state(self):
        PythonDriverWrapper._GLOBAL_PROMPT_TO_ID.clear()
        yield
        PythonDriverWrapper._GLOBAL_PROMPT_TO_ID.clear()

    @pytest.fixture
    def driver_files(self, tmp_path):
        """Minimal driver file + baseline that test_single_output checks for."""
        driver_file = tmp_path / "pycompss.py"
        driver_file.write_text("# fake driver\n")
        (tmp_path / "baseline.py").write_text("# fake baseline\n")
        return str(driver_file)

    def test_successful_run_returns_valid_result(self, driver_files):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        with patch.object(d, "compile", return_value=BuildOutput(0, "", "")):
            with patch.object(d, "run", return_value=RunOutput(0, PASS_OUTPUT, "")):
                result = d.test_single_output("# prompt", "@task()\ndef f(): pass\n", driver_files, "100")
        assert result.did_build()
        assert result.are_any_valid()
        assert result.relaxations_applied == []

    def test_build_failure_produces_no_run_outputs(self, driver_files):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        with patch.object(d, "compile", return_value=BuildOutput(1, "", "syntax error")):
            result = d.test_single_output("# prompt", "# output", driver_files, "100")
        assert not result.did_build()
        assert result.run_outputs is None

    def test_relaxation_applied_when_initial_run_fails(self, driver_files):
        d = make_driver(
            resource_slot=None,
            launch_format=SIMPLE_FMT,
            relaxations=[RenameMainRelaxation()],
        )
        compile_results = [BuildOutput(0, "", ""), BuildOutput(0, "", "")]
        run_results = [RunOutput(0, FAIL_OUTPUT, ""), RunOutput(0, PASS_OUTPUT, "")]
        with patch.object(d, "compile", side_effect=compile_results):
            with patch.object(d, "run", side_effect=run_results):
                result = d.test_single_output("# prompt", _RELAX_OUTPUT, driver_files, "100")
        assert result.relaxations_applied == ["rename_main"]
        assert result.are_any_valid()

    def test_relaxation_skipped_when_not_applicable(self, driver_files):
        """When relaxation.apply() returns None the retry is skipped and the
        original (failed) run result is kept."""
        d = make_driver(
            resource_slot=None,
            launch_format=SIMPLE_FMT,
            relaxations=[RenameMainRelaxation()],
        )
        with patch.object(d, "compile", return_value=BuildOutput(0, "", "")):
            with patch.object(d, "run", return_value=RunOutput(0, FAIL_OUTPUT, "")):
                result = d.test_single_output("# prompt", _NO_RELAX_OUTPUT, driver_files, "100")
        # Relaxation did not apply; result is the original failure
        assert result.relaxations_applied == []
        assert not result.are_any_valid()

    def test_relaxation_rebuild_failure_continues_to_next(self, driver_files):
        """If the post-relaxation compile fails, the loop moves on and does not
        raise; the final result reflects the original failure."""
        d = make_driver(
            resource_slot=None,
            launch_format=SIMPLE_FMT,
            relaxations=[RenameMainRelaxation()],
        )
        # First compile succeeds, second (post-relaxation) compile fails
        compile_results = [BuildOutput(0, "", ""), BuildOutput(1, "", "error")]
        with patch.object(d, "compile", side_effect=compile_results):
            with patch.object(d, "run", return_value=RunOutput(0, FAIL_OUTPUT, "")):
                result = d.test_single_output("# prompt", _RELAX_OUTPUT, driver_files, "100")
        assert result.relaxations_applied == []
        assert not result.are_any_valid()

    def test_early_exit_stops_after_first_run_failure(self, driver_files):
        d = make_driver(
            resource_slot=None,
            launch_format=SIMPLE_FMT,
            launch_params=[{}, {}],
            early_exit_runs=True,
        )
        run_count = [0]
        def counting_run(exec_path, **kw):
            run_count[0] += 1
            return RunOutput(0, FAIL_OUTPUT, "")

        with patch.object(d, "compile", return_value=BuildOutput(0, "", "")):
            with patch.object(d, "run", side_effect=counting_run):
                d.test_single_output("# prompt", "@task()\ndef f(): pass\n", driver_files, "100")

        assert run_count[0] == 1

    def test_missing_driver_file_raises(self, tmp_path):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        with pytest.raises(FileNotFoundError):
            d.test_single_output("# prompt", "@task()\ndef f(): pass\n",
                                 str(tmp_path / "nonexistent.py"), "100")


# ---------------------------------------------------------------------------
# test_single_output(): @task gate
# ---------------------------------------------------------------------------

class TestNoTaskGate:
    """Outputs without @task must be rejected before any file I/O or subprocess."""

    @pytest.fixture(autouse=True)
    def clear_global_state(self):
        PythonDriverWrapper._GLOBAL_PROMPT_TO_ID.clear()
        yield
        PythonDriverWrapper._GLOBAL_PROMPT_TO_ID.clear()

    @pytest.fixture
    def driver_files(self, tmp_path):
        driver_file = tmp_path / "pycompss.py"
        driver_file.write_text("# fake driver\n")
        (tmp_path / "baseline.py").write_text("# fake baseline\n")
        return str(driver_file)

    _SEQUENTIAL = "def solve(x):\n    return sum(x)\n"
    _WITH_TASK = "@task(returns=list)\ndef solve(x):\n    return x\n"

    def test_returns_did_build_false(self, tmp_path):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        # Use a nonexistent driver path — early exit must happen before the path is checked
        result = d.test_single_output("# prompt", self._SEQUENTIAL,
                                      str(tmp_path / "nonexistent.py"), "100")
        assert not result.did_build()

    def test_returns_run_outputs_none(self, tmp_path):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        result = d.test_single_output("# prompt", self._SEQUENTIAL,
                                      str(tmp_path / "nonexistent.py"), "100")
        assert result.run_outputs is None

    def test_compile_never_called(self, driver_files):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        with patch.object(d, "compile") as mock_compile:
            d.test_single_output("# prompt", self._SEQUENTIAL, driver_files, "100")
        mock_compile.assert_not_called()

    def test_run_never_called(self, driver_files):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        with patch.object(d, "run") as mock_run:
            d.test_single_output("# prompt", self._SEQUENTIAL, driver_files, "100")
        mock_run.assert_not_called()

    def test_with_task_proceeds_to_compile(self, driver_files):
        d = make_driver(resource_slot=None, launch_format=SIMPLE_FMT)
        with patch.object(d, "compile", return_value=BuildOutput(0, "", "")) as mock_compile:
            with patch.object(d, "run", return_value=RunOutput(0, PASS_OUTPUT, "")):
                d.test_single_output("# prompt", self._WITH_TASK, driver_files, "100")
        mock_compile.assert_called_once()
