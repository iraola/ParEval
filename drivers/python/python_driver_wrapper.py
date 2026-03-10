""" Wrapper for calling python drivers
    author: 
    date: November 2025
"""
# std imports
import copy
import logging
import os
from os import PathLike
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
# local imports
sys.path.append("..")
from drivers.driver_wrapper import DriverWrapper, BuildOutput, RunOutput, GeneratedTextResult
from util import run_command

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


def _runs_succeeded(run_results) -> bool:
    """Return True if at least one run completed and validated successfully."""
    if not run_results:
        return False
    return any(r.exit_code == 0 and r.is_valid for r in run_results)


class PythonDriverWrapper(DriverWrapper):
    
    # GLOBAL TRACKER: This stays alive across all instances of the class
    # to ensure folder _0, _1, _2 are assigned correctly.
    _GLOBAL_PROMPT_TO_ID = {}

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

            logging.info(f"Merged {len(binaries)} files into {output_path}")
            return BuildOutput(0, f"Merged files into {output_path}", "")

        except Exception as e:
            logging.error(f"Failed to merge files: {e}")
            return BuildOutput(1, "", str(e))

    def run(self, executable: PathLike, **run_config) -> RunOutput:
        """ Run the given executable. """
        launch_format = self.launch_configs["format"]
        launch_cmd = launch_format.format(exec_path=executable, args="", **run_config).strip()
        try:
            run_process = run_command(launch_cmd, timeout=self.run_timeout, dry=self.dry, parallelism_model=self.parallelism_model)
        except subprocess.TimeoutExpired as e:
            return RunOutput(-1, str(e.stdout), f"[Timeout] {str(e.stderr)}", config=run_config)
        except UnicodeDecodeError as e:
            logging.warning(f"UnicodeDecodeError: {str(e)}\nRunnning command: {launch_cmd}")
            return RunOutput(-1, "", f"UnicodeDecodeError: {str(e)}", config=run_config)
        return RunOutput(run_process.returncode, run_process.stdout, run_process.stderr, config=run_config)

    def test_single_output(self, prompt: str, output: str, test_driver_file: PathLike, problem_size: str, prompt_name: str = "prompt", problem_type: str = "problem", output_index: int = 0) -> GeneratedTextResult:
        """ Test a single generated output for PyCOMPSs.
            Writes generated_code.py, stages a harness (existing cpu_harness.py/cpu.py or a tiny adapter),
            and launches the shared model driver under runcompss via a small runner.py.
        """
        logging.debug(f"Testing output (python/pycompss):\n{output[:500]}{'...' if len(output)>500 else ''}")

        # 1. Use the Class-level global tracker
        if prompt not in PythonDriverWrapper._GLOBAL_PROMPT_TO_ID:
            PythonDriverWrapper._GLOBAL_PROMPT_TO_ID[prompt] = len(PythonDriverWrapper._GLOBAL_PROMPT_TO_ID)
        
        p_idx = PythonDriverWrapper._GLOBAL_PROMPT_TO_ID[prompt]

        with tempfile.TemporaryDirectory(dir=self.scratch_dir) as tmpdir:
            src_path = os.path.join(tmpdir, "generated_code.py")
            write_success = self.write_source(prompt + "\n" + output, src_path)

            # 2. Setup Directory
            dst_dir = None
            if self.save_generated_dir:
                safe_type = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in problem_type)
                safe_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in prompt_name)
                dst_dir = os.path.join(self.save_generated_dir, safe_type, f"{safe_name}_{p_idx}")
                os.makedirs(dst_dir, exist_ok=True)

                # Save metadata
                prompt_file = os.path.join(dst_dir, "prompt.txt")
                if not os.path.exists(prompt_file):
                    with open(prompt_file, "w") as f:
                        f.write(prompt)

                # Save raw code
                shutil.copyfile(src_path, os.path.join(dst_dir, f"{self.parallelism_model}_{output_index}.py"))

            # Create config file explicitly for this prompt run
            config_path = os.path.join(tmpdir, "driver_config.py")
            with open(config_path, "w") as f:
                f.write(f"# Auto-generated configuration\n")
                f.write(f"DRIVER_PROBLEM_SIZE = {problem_size}\n")
                f.write(f"MAX_VALIDATION_ATTEMPTS = 5\n")

            # 3. Build
            exec_path = os.path.join(tmpdir, "a_out.py")
            driver_dir = os.path.dirname(test_driver_file)
            baseline_path = os.path.join(driver_dir, "baseline.py")
            
            sources = [self.model_driver_file]
            sources.append(test_driver_file)
            if os.path.exists(baseline_path):
                sources.append(baseline_path)
            sources.append(src_path)
            sources.append(config_path)
            
            build_result = self.compile(*sources, output_path=exec_path)

            # Save Merged Code immediately after build, so it's available for inspection
            if dst_dir and build_result.did_build:
                merged_fname = f"{self.parallelism_model}_{output_index}_merged_a_out.py"
                merged_path = os.path.join(dst_dir, merged_fname)
                shutil.copyfile(exec_path, merged_path)
                logging.info(f"Saved merged executable to {merged_path}")

            # run the code
            configs = self.launch_configs["params"]
            if build_result.did_build:
                run_results = []
                for c in configs:
                    run_result = self.run(exec_path, **c)
                    run_results.append(run_result)
                    if self.display_runs:
                        logging.debug(run_result.stderr)
                        logging.debug(run_result.stdout)
                    if self.early_exit_runs and (run_result.exit_code != 0 or not run_result.is_valid):
                        break
            else:
                run_results = None

            logging.debug(f"Run results: {run_results}")
            if run_results:
                for rr in run_results:
                    if rr.exit_code != 0:
                        logging.debug(f"Outputs for failed run:\n\tstdout: {rr.stdout}\n\tstderr: {rr.stderr}")

            # Relaxation retry: if the run failed (or didn't happen) and relaxations
            # are configured, try each one by modifying generated_code.py, rebuilding,
            # and rerunning. Stop at the first relaxation that produces a passing run.
            applied_relaxations = []
            if self.relaxations and not _runs_succeeded(run_results):
                for relaxation in self.relaxations:
                    modified_output = relaxation.apply(output)
                    if modified_output is None:
                        logging.debug(f"Relaxation '{relaxation.name}' does not apply, skipping.")
                        continue

                    logging.info(f"Applying relaxation '{relaxation.name}' for {prompt_name}[{output_index}].")
                    self.write_source(prompt + "\n" + modified_output, src_path)
                    new_build = self.compile(*sources, output_path=exec_path)

                    if not new_build.did_build:
                        logging.debug(f"Relaxation '{relaxation.name}': build failed after transform.")
                        continue

                    new_runs = []
                    for c in configs:
                        run_result = self.run(exec_path, **c)
                        new_runs.append(run_result)
                        if self.display_runs:
                            logging.debug(run_result.stderr)
                            logging.debug(run_result.stdout)
                        if self.early_exit_runs and (run_result.exit_code != 0 or not run_result.is_valid):
                            break

                    if _runs_succeeded(new_runs):
                        logging.info(f"Relaxation '{relaxation.name}' succeeded for {prompt_name}[{output_index}].")
                        build_result = new_build
                        run_results = new_runs
                        applied_relaxations.append(relaxation.name)
                        break
                    else:
                        logging.debug(f"Relaxation '{relaxation.name}': run still failed after transform.")

            return GeneratedTextResult(write_success, build_result, run_results, relaxations_applied=applied_relaxations)