""" Wrapper for calling python drivers
    author: 
    date: November 2025
"""
# std imports
import copy
import difflib
import glob
import logging
import os
from os import PathLike
import re
import subprocess
import sys
import tempfile
import threading
import shutil
from pathlib import Path
# local imports
sys.path.append("..")
from drivers.driver_wrapper import DriverWrapper, BuildOutput, RunOutput, GeneratedTextResult
from util import run_command

def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


""" Map parallelism models to driver files """
DRIVER_MAP = {
    "pycompss": "pycompss_driver.py",
}


def try_to_find_path(src_path):
    """
    Try to resolve a benchmark file path when the numeric prefix of the benchmark
    directory doesn't match the expected name. Searches for a folder under
    python/benchmarks/<problem_type>/ whose name contains the example name and
    returns the path to the file within it. Returns None if the path cannot be resolved.
    """
    src_str = str(src_path)
    if "benchmarks" in src_str and src_str.startswith("python/benchmarks"):
        # Match: python/benchmarks/<problem_type>/<orig_folder>/<file>
        import re
        m = re.match(
            r"(python/benchmarks/)([^/]+)/(\d+_)?([^_/]+)_(.+?)/([^/]+)$",
            src_str)
        if m:
            base_dir = Path(m.group(1)) / m.group(2)
            problem_type = m.group(2)
            problem_type_2 = m.group(4)
            example_name = m.group(5)
            file_name = m.group(6)
            # Make sure problem types match
            if not problem_type_2 in problem_type:
                return None
            # Find actual benchmark dir with the same example name
            for folder in base_dir.iterdir():
                ref_folder = str(folder.name).replace("-", "_")
                recvd_folder = example_name.replace("-", "_")
                if recvd_folder in ref_folder:
                    return folder / file_name
            else:
                return None
        else:
            return None
    else:
        return None


def _compress_model_name(model_name: str) -> str:
    name = model_name.split("/")[-1]
    return "".join(p[:2] for p in re.split(r"[-_]", name) if p)


def _prompt_id(prompt_name: str) -> str:
    """Return the numeric prefix of a prompt name (e.g. '22' from '22_histogram_count_quadrants')."""
    m = re.match(r"(\d+)", prompt_name)
    return m.group(1) if m else prompt_name


def _runs_succeeded(run_results) -> bool:
    """Return True if at least one run completed and validated successfully."""
    if not run_results:
        return False
    return any(r.exit_code == 0 and r.is_valid for r in run_results)


class PythonDriverWrapper(DriverWrapper):

    # GLOBAL TRACKER: This stays alive across all instances of the class
    # to ensure folder _0, _1, _2 are assigned correctly.
    _GLOBAL_PROMPT_TO_ID = {}
    _GLOBAL_PROMPT_TO_ID_LOCK = threading.Lock()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_driver_file = os.path.join("python", "models", DRIVER_MAP[self.parallelism_model])

    def write_source(self, content: str, fpath: PathLike) -> bool:
        """ Write the given python source to the given file. """
        with open(fpath, "w") as fp:
            fp.write(content)
        return True

    """
    def patch_prompt(self, content: str) -> str:
        # Add NO_INLINE to the given source code.
        # the last line of content should be: return_type function_name(args) {
        # we want to add NO_INLINE after the return_typewwhe
        parts = content.split("\n")[-1].split(" ")
        assert len(parts) > 1, f"Could not parse return type from {parts}"
        parts.insert(1, "NO_INLINE")
        return "\n".join(content.split("\n")[:-1] + [" ".join(parts)])
    """

    def compile(
        self,
        *binaries: PathLike,
        output_path: PathLike = "a_out.py",
    ) -> BuildOutput:
        """
        Merge multiple Python source files into a single runnable Python file
        for PyCOMPSs execution. This mimics C++ compilation.
        
        Parameters:
        - binaries: e.g., (pycompss_driver.py, cpu.py)
        - output_path: path to the combined file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as out_fp:
                for src in binaries[::-1]:
                    src_path = Path(src)
                    if not src_path.exists():
                        src_path = try_to_find_path(src_path)
                        if src_path is None:
                            return BuildOutput(1, "", f"Could not find source file {src}")

                    with open(src_path, "r") as in_fp:
                        content = in_fp.read()
                        out_fp.write(f"# ===== {src_path.name} =====\n")
                        out_fp.write(content + "\n\n")

            logging.debug(f"Merged {len(binaries)} files into {output_path}")
            return BuildOutput(0, f"Merged files into {output_path}", "")

        except Exception as e:
            logging.error(f"Failed to merge files: {e}")
            return BuildOutput(1, "", str(e))

    @staticmethod
    def _enrich_stderr_with_compss_job_logs(stderr: str, tail_lines: int = 40) -> str:
        """If stderr references .COMPSs job files, read them and append a snippet.

        COMPSs error messages look like:
            Check files '/home/.../.COMPSs/a_out.py_40/jobs/job[1|2]' to find out the error.

        The bracket notation job[1|2] names the specific job IDs to inspect —
        there can be hundreds of other job files in the same directory, so we
        only open the ones explicitly named.  For each ID we glob
        job<id>*.out / job<id>*.err in the jobs directory.

        When no bracket notation is present (e.g. a plain directory path) we
        fall back to the first few *.out/*.err files in that directory.
        """
        pattern = re.compile(r'((?:~|/\S+)?\.COMPSs/\S+)', re.IGNORECASE)
        appended: list[str] = []
        seen_dirs: set[str] = set()

        for m in pattern.finditer(stderr):
            raw = m.group(1).rstrip(".,;'\"")

            # ── Case 1: bracket notation  e.g. .../jobs/job[1|2|3] ──────────
            bracket_m = re.search(r'\[([^\]]+)\]', raw)
            if bracket_m:
                bracket_pos = raw.index('[')
                path_prefix = raw[:bracket_pos]           # e.g. .../jobs/job
                parent_dir  = os.path.expanduser(os.path.dirname(path_prefix))
                file_prefix = os.path.basename(path_prefix)  # e.g. "job"
                job_ids     = [i.strip() for i in bracket_m.group(1).split('|')]

                if not os.path.isdir(parent_dir) or parent_dir in seen_dirs:
                    continue
                seen_dirs.add(parent_dir)

                candidates: list[str] = []
                for job_id in job_ids:
                    # Match exact name (job1.err) and underscore-suffixed variants
                    # (job1_NEW.err).  Avoid job1*.err which would also match
                    # job10.err, job11.err, etc.
                    for ext in ("out", "err"):
                        candidates += sorted(glob.glob(
                            os.path.join(parent_dir, f"{file_prefix}{job_id}.{ext}")))
                        candidates += sorted(glob.glob(
                            os.path.join(parent_dir, f"{file_prefix}{job_id}_*.{ext}")))

            # ── Case 2: plain path ────────────────────────────────────────────
            else:
                path = os.path.expanduser(raw)
                if os.path.isfile(path):
                    candidates = [path]
                elif os.path.isdir(path):
                    if path in seen_dirs:
                        continue
                    seen_dirs.add(path)
                    candidates = (sorted(glob.glob(os.path.join(path, "*.out")))
                                + sorted(glob.glob(os.path.join(path, "*.err"))))[:4]
                else:
                    continue

            for job_file in candidates:
                try:
                    with open(job_file, errors="replace") as f:
                        lines = f.readlines()
                    snippet = "".join(lines[-tail_lines:])
                    if snippet.strip():
                        appended.append(f"\n[job log: {job_file}]\n{snippet}")
                except OSError:
                    pass

        return stderr + "".join(appended)

    def run(self, executable: PathLike, **run_config) -> RunOutput:
        """ Run the given executable. """
        launch_format = self.launch_configs["format"]
        compss_workdir = None
        if self.resource_slot:
            compss_workdir = tempfile.mkdtemp(dir=self.scratch_dir)
            run_config = {**run_config, **self.resource_slot, "master_working_dir": compss_workdir}
        launch_cmd = launch_format.format(exec_path=executable, args="", **run_config).strip()
        try:
            run_process = run_command(launch_cmd, timeout=self.run_timeout, dry=self.dry, parallelism_model=self.parallelism_model)
        except subprocess.TimeoutExpired as e:
            def _decode(b) -> str:
                if b is None:
                    return ""
                return b.decode("utf-8", errors="replace") if isinstance(b, bytes) else str(b)
            stderr = self._enrich_stderr_with_compss_job_logs(f"[Timeout] {_decode(e.stderr)}")
            return RunOutput(-1, _decode(e.stdout), stderr, config=run_config)
        except UnicodeDecodeError as e:
            logging.warning(f"UnicodeDecodeError: {str(e)}\nRunnning command: {launch_cmd}")
            return RunOutput(-1, "", f"UnicodeDecodeError: {str(e)}", config=run_config)
        finally:
            if compss_workdir:
                shutil.rmtree(compss_workdir, ignore_errors=True)
        stderr = self._enrich_stderr_with_compss_job_logs(run_process.stderr)
        return RunOutput(run_process.returncode, run_process.stdout, stderr, config=run_config)

    def test_single_output(self, prompt: str, output: str, test_driver_file: PathLike, problem_size: str, prompt_name: str = "prompt", problem_type: str = "problem", output_index: int = 0) -> GeneratedTextResult:
        """ Test a single generated output for PyCOMPSs.
            Writes generated_code.py, stages a harness (existing cpu_harness.py/cpu.py or a tiny adapter),
            and launches the shared model driver under runcompss via a small runner.py.
        """
        preview = output[:500] + ("..." if len(output) > 500 else "")
        logging.debug("code preview:\n%s", _indent(preview))

        with PythonDriverWrapper._GLOBAL_PROMPT_TO_ID_LOCK:
            if prompt not in PythonDriverWrapper._GLOBAL_PROMPT_TO_ID:
                PythonDriverWrapper._GLOBAL_PROMPT_TO_ID[prompt] = len(PythonDriverWrapper._GLOBAL_PROMPT_TO_ID)
            p_idx = PythonDriverWrapper._GLOBAL_PROMPT_TO_ID[prompt]

        artifact_stem = f"py{output_index}_{_compress_model_name(self.model_name)}_{_prompt_id(prompt_name)}"

        with tempfile.TemporaryDirectory(dir=self.scratch_dir) as tmpdir:
            src_path = os.path.join(tmpdir, f"{artifact_stem}.py")
            write_success = self.write_source(prompt + "\n" + output, src_path)

            # Setup directory
            dst_dir = None
            if self.save_generated_dir:
                dst_dir = os.path.join(self.save_generated_dir, problem_type, f"{prompt_name}_{p_idx}")
                os.makedirs(dst_dir, exist_ok=True)

                # Save metadata
                prompt_file = os.path.join(dst_dir, "prompt.txt")
                if not os.path.exists(prompt_file):
                    with open(prompt_file, "w") as f:
                        f.write(prompt)

                # Save raw code
                shutil.copyfile(src_path, os.path.join(dst_dir, f"{artifact_stem}.py"))

            # Create config file explicitly for this prompt run
            config_path = os.path.join(tmpdir, "driver_config.py")
            with open(config_path, "w") as f:
                f.write(f"# Auto-generated configuration\n")
                f.write(f"DRIVER_PROBLEM_SIZE = {problem_size}\n")
                f.write(f"MAX_VALIDATION_ATTEMPTS = 5\n")

            # Build
            exec_path = os.path.join(tmpdir, f"{artifact_stem}_merged.py")
            if not Path(test_driver_file).exists():
                resolved = try_to_find_path(Path(test_driver_file))
                if resolved is not None:
                    test_driver_file = str(resolved)
                else:
                    raise FileNotFoundError(f"pycompss.py driver file not found: {test_driver_file}")
            driver_dir = os.path.dirname(test_driver_file)
            baseline_path = os.path.join(driver_dir, "baseline.py")
            if not os.path.exists(baseline_path):
                raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
            
            sources = [self.model_driver_file]
            sources.append(test_driver_file)
            sources.append(baseline_path)
            sources.append(src_path)
            sources.append(config_path)
            
            build_result = self.compile(*sources, output_path=exec_path)

            merged_path = None
            if dst_dir and build_result.did_build:
                merged_path = os.path.join(dst_dir, f"{artifact_stem}_merged.py")
                shutil.copyfile(exec_path, merged_path)

            if build_result.did_build:
                save_note = f"  saved: {merged_path}" if merged_path else ""
                logging.info("[build] OK%s", save_note)
            else:
                logging.info("[build] FAILED")
                logging.debug("[build] stderr:\n%s", _indent(build_result.stderr[:300]))

            # run the code
            configs = self.launch_configs["params"]
            n_configs = len(configs)
            if build_result.did_build:
                run_results = []
                for j, c in enumerate(configs):
                    logging.debug("[run %d/%d]", j + 1, n_configs)
                    run_result = self.run(exec_path, **c)
                    run_results.append(run_result)
                    time_str = f"{run_result.runtime:.3f}s" if run_result.runtime is not None else "–"
                    logging.info("[run %d/%d] exit=%d  valid=%s  time=%s",
                                 j + 1, n_configs, run_result.exit_code, run_result.is_valid, time_str)
                    if self.display_runs:
                        logging.debug(
                            "[run %d/%d output]\n%s",
                            j + 1, n_configs,
                            _indent(f"--- stdout ---\n{run_result.stdout}\n--- stderr ---\n{run_result.stderr}"),
                        )
                    if self.early_exit_runs and (run_result.exit_code != 0 or not run_result.is_valid):
                        break
            else:
                run_results = None

            # Relaxation retry: if the run failed (or didn't happen) and relaxations
            # are configured, try each one by modifying generated_code.py, rebuilding,
            # and rerunning. Stop at the first relaxation that produces a passing run.
            applied_relaxations = []
            if self.relaxations and not _runs_succeeded(run_results):
                n_relaxations = len(self.relaxations)
                logging.debug("[relaxations] initial run failed — trying %d relaxation(s)", n_relaxations)
                for k, relaxation in enumerate(self.relaxations):
                    logging.debug("── relaxation %d/%d: %s ──────────────────────────",
                                  k + 1, n_relaxations, relaxation.name)
                    modified_output = relaxation.apply(output)
                    if modified_output is None:
                        logging.debug("does not apply, skipping")
                        continue

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        diff = "".join(difflib.unified_diff(
                            output.splitlines(keepends=True),
                            modified_output.splitlines(keepends=True),
                            fromfile="original",
                            tofile=f"after_{relaxation.name}",
                            n=2,
                        ))
                        if diff:
                            logging.debug("relaxation diff:\n%s",
                                          _indent(f"--- diff ---\n{diff}--- end diff ---"))

                    self.write_source(prompt + "\n" + modified_output, src_path)
                    new_build = self.compile(*sources, output_path=exec_path)

                    if not new_build.did_build:
                        logging.debug("[build] FAILED after transform")
                        continue
                    logging.debug("[build] OK")

                    new_runs = []
                    for j, c in enumerate(configs):
                        logging.debug("[run %d/%d]", j + 1, n_configs)
                        run_result = self.run(exec_path, **c)
                        new_runs.append(run_result)
                        time_str = f"{run_result.runtime:.3f}s" if run_result.runtime is not None else "–"
                        logging.info("[run %d/%d] exit=%d  valid=%s  time=%s",
                                     j + 1, n_configs, run_result.exit_code, run_result.is_valid, time_str)
                        if self.display_runs:
                            logging.debug(
                                "[run %d/%d output]\n  --- stdout ---\n%s\n  --- end stdout ---\n"
                                "  --- stderr ---\n%s\n  --- end stderr ---",
                                j + 1, n_configs, run_result.stdout, run_result.stderr,
                            )
                        if self.early_exit_runs and (run_result.exit_code != 0 or not run_result.is_valid):
                            break

                    if _runs_succeeded(new_runs):
                        logging.debug("succeeded")
                        build_result = new_build
                        run_results = new_runs
                        applied_relaxations.append(relaxation.name)
                        break
                    else:
                        logging.debug("still failed")

            # Per-output outcome summary
            if _runs_succeeded(run_results):
                best = next((r for r in run_results if r.exit_code == 0 and r.is_valid), None)
                time_str = f"{best.runtime:.3f}s" if best and best.runtime is not None else "–"
                relax_note = f"  [via relaxation: {applied_relaxations[0]}]" if applied_relaxations else ""
                logging.info("output %d: PASS  time=%s%s", output_index, time_str, relax_note)
            elif not build_result.did_build:
                logging.info("output %d: FAIL  build error", output_index)
            elif run_results is not None and all(r.exit_code != 0 for r in run_results):
                logging.info("output %d: FAIL  runtime error", output_index)
            elif run_results is not None and not any(r.is_valid for r in run_results if r.exit_code == 0):
                logging.info("output %d: FAIL  validation error", output_index)
            else:
                relax_note = " — all relaxations exhausted" if self.relaxations else ""
                logging.info("output %d: FAIL%s", output_index, relax_note)

            return GeneratedTextResult(write_success, build_result, run_results, relaxations_applied=applied_relaxations)