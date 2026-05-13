#!/bin/python3
""" Run all the generated code.
    author: Daniel Nichols
    date: October 2023
"""
# std imports
from argparse import ArgumentParser
import json
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# tpl imports
from tqdm import tqdm

# local imports
from driver_wrapper import DriverWrapper
from cpp.cpp_driver_wrapper import CppDriverWrapper
from python.python_driver_wrapper import PythonDriverWrapper
from python.relaxations import ALL_RELAXATIONS, RELAXATION_MAP
from util import await_input, load_json

""" Map language names to driver wrappers """
LANGUAGE_DRIVERS = {
    "cpp": CppDriverWrapper,
    "python": PythonDriverWrapper,
}


class ThreadBufferingHandler(logging.Handler):
    """Buffer log records per worker thread; flush atomically to a target handler.

    Main-thread records are emitted immediately. Worker records accumulate
    in a per-thread list and are written out all at once when flush_thread()
    is called, preventing interleaving between concurrent prompts.
    """

    def __init__(self, target: logging.Handler):
        super().__init__()
        self.target = target
        self._buffers: dict[int, list] = {}
        self._flush_lock = threading.Lock()

    def emit(self, record):
        if threading.current_thread() is threading.main_thread():
            with self._flush_lock:
                self.target.emit(record)
        else:
            tid = threading.get_ident()
            if tid not in self._buffers:
                self._buffers[tid] = []
            self._buffers[tid].append(record)

    def flush_thread(self):
        tid = threading.get_ident()
        records = self._buffers.pop(tid, [])
        if records:
            with self._flush_lock:
                for record in records:
                    self.target.emit(record)

def get_args():
    parser = ArgumentParser(description="Run all the generated code.")
    parser.add_argument("input_json", type=str, help="Input JSON file containing the test cases.")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file containing the results.")
    parser.add_argument("--scratch-dir", type=str, help="If provided, put scratch files here.")
    parser.add_argument("--artifacts-dir", type=str, metavar="DIR", help="Directory to save per-output evaluation artifacts (raw generated code and merged executable). Defaults to 'artifacts/' under the driver root.")
    parser.add_argument("--driver-root", type=str, help="Where to look for the driver files, if not in cwd.")
    parser.add_argument("--launch-configs", type=str, default="launch-configs.json",
        help="config for how to run samples.")
    parser.add_argument("--build-configs", type=str, default="build-configs.json",
        help="config for how to build samples. If not provided, will use the default build settings for each model.")
    parser.add_argument("--problem-sizes", type=str, default="problem-sizes.json",
        help="config for how to run samples.")
    parser.add_argument("--yes-to-all", action="store_true", help="If provided, automatically answer yes to all prompts.")
    parser.add_argument("--dry", action="store_true", help="Dry run. Do not actually run the code snippets.")
    parser.add_argument("--resume", action="store_true", help="If the output file already exists, load it as starting data instead of input_json, skipping already-evaluated entries.")
    parser.add_argument("--overwrite", action="store_true", help="If ouputs are already in DB for a given prompt, \
        then overwrite them. Default behavior is to skip existing results.")
    parser.add_argument("--hide-progress", action="store_true", help="If provided, do not show progress bar.")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--exclude-models", nargs="+", type=str, choices=["serial", "omp", "mpi", "mpi+omp", "kokkos", "cuda", "hip", "pycompss"],
        help="Exclude the given parallelism models from testing.")
    model_group.add_argument("--include-models", nargs="+", type=str, choices=["serial", "omp", "mpi", "mpi+omp", "kokkos", "cuda", "hip", "pycompss"],
        help="Only test the given parallelism models.")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--problem", type=str, help="Only test this probem if provided.")
    model_group.add_argument("--problem-type", type=str, help="Only test problems of this type if provided.")
    parser.add_argument("--early-exit-runs", action="store_true", help="If provided, stop evaluating a model output after the first run configuration fails.")
    parser.add_argument("--build-timeout", type=int, default=30, help="Timeout in seconds for building a program.")
    parser.add_argument("--run-timeout", type=int, default=120, help="Timeout in seconds for running a program.")
    parser.add_argument("--log", choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"], default="INFO",
        type=str.upper, help="logging level")
    parser.add_argument("--log-build-errors", action="store_true", help="On build error, display the stderr of the build process.")
    parser.add_argument("--log-runs", action="store_true", help="Display the stderr and stdout of runs.")
    relaxation_names = list(RELAXATION_MAP.keys())
    parser.add_argument(
        "--relaxations",
        nargs="+",
        metavar="RELAXATION",
        choices=relaxation_names + ["all"],
        default=None,
        help=(
            "Apply source-code relaxations when a run fails, then retry. "
            "Use 'all' to enable every relaxation, or pass one or more names to enable specific ones. "
            "Available: " + ", ".join(relaxation_names) + ". "
            "Disabled by default."
        ),
    )
    parser.add_argument("--resource-slots", type=str, metavar="FILE",
        help="JSON file listing resource slots for parallel execution (keys: cpu_affinity, resources_xml). One thread per slot.")
    return parser.parse_args()

def resolve_relaxations(relaxations_arg):
    """ Resolve the --relaxations CLI argument to a list of Relaxation instances. """
    if relaxations_arg is None:
        return []
    if "all" in relaxations_arg:
        return list(ALL_RELAXATIONS)
    return [RELAXATION_MAP[name] for name in relaxations_arg]


def get_driver(
    prompt: dict,
    scratch_dir: Optional[os.PathLike],
    launch_configs: dict,
    build_configs: dict,
    problem_sizes: dict,
    dry: bool,
    **kwargs
) -> DriverWrapper:
    """ Get the language drive wrapper for this prompt """
    driver_cls = LANGUAGE_DRIVERS[prompt["language"]]
    return driver_cls(parallelism_model=prompt["parallelism_model"], launch_configs=launch_configs,
        build_configs=build_configs, problem_sizes=problem_sizes, scratch_dir=scratch_dir, dry=dry, **kwargs)

def already_has_results(prompt: dict) -> bool:
    """ Check if a prompt already has results stored in it. """
    if "outputs" not in prompt or not isinstance(prompt["outputs"], list):
        raise ValueError(f"Prompt {prompt.get('name', 'unknown')} does not have any outputs.")
    
    outputs = prompt["outputs"]
    if len(outputs) == 0 or all(isinstance(o, str) for o in outputs):
        return False

    if len(outputs) > 0 and all(isinstance(o, dict) for o in outputs):
        return True

    raise ValueError(f"Prompt {prompt.get('name', 'unknown')} has invalid outputs.")

def main():
    args = get_args()

    # setup logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(args.log))
    log_format = "%(asctime)s [%(levelname)s] [%(threadName)s] -- %(message)s" if args.resource_slots \
        else "%(asctime)s [%(levelname)s] -- %(message)s"
    logging.basicConfig(format=log_format, datefmt="%H:%M:%S", level=numeric_level)

    # warn user before continuing
    logging.warning("This script will compile and run code generated by an LLM. " +
        "It is recommended that you run this script in a sandboxed environment.")
    if not args.yes_to_all:
        response = await_input("Continue? [y/n] ", lambda r: r.lower() in ["y", "n", "yes", "no"])
        if response.lower() in ["n", "no"]:
            logging.info("Exiting.")
            return

    # load in the generated text, resuming from a partial output file if requested
    if args.resume:
        if not args.output or args.output == '-':
            logging.warning("--resume was set but no output file was specified (-o); ignoring --resume and loading from input_json.")
            data = load_json(args.input_json)
            logging.info(f"Loaded {len(data)} prompts from {args.input_json}.")
        elif not os.path.isfile(args.output):
            logging.warning(f"--resume was set but output file {args.output} does not exist yet; loading from input_json.")
            data = load_json(args.input_json)
            logging.info(f"Loaded {len(data)} prompts from {args.input_json}.")
        else:
            data = load_json(args.output)
            logging.info(f"Resuming from existing output file {args.output} ({len(data)} prompts).")
    else:
        data = load_json(args.input_json)
        logging.info(f"Loaded {len(data)} prompts from {args.input_json}.")

    # derive model name from the input filename (e.g. "output-gpt-oss-20b.json" → "gpt-oss-20b")
    input_stem = os.path.splitext(os.path.basename(args.input_json))[0]
    model_name = input_stem[len("output-"):] if input_stem.startswith("output-") else input_stem

    # load launch configs
    launch_configs = load_json(args.launch_configs)
    logging.info(f"Loaded launch configs from {args.launch_configs}.")

    # load build configs
    build_configs = load_json(args.build_configs)
    logging.info(f"Loaded build configs from {args.build_configs}.")

    # load problem sizes
    problem_sizes = load_json(args.problem_sizes)
    logging.info(f"Loaded problem sizes from {args.problem_sizes}.")

    # set driver root; If provided, use user argument. If it's not provided, then check if the PAREVAL_ROOT environment
    # variable is set, then use "${PAREVAL_ROOT}/drivers" as the root. If neither is set, then use the location of 
    # this script as the root.
    if args.driver_root:
        DRIVER_ROOT = args.driver_root
    elif "PAREVAL_ROOT" in os.environ:
        DRIVER_ROOT = os.path.join(os.environ["PAREVAL_ROOT"], "drivers")
    else:
        DRIVER_ROOT = os.path.dirname(os.path.abspath(__file__))
    logging.info(f"Using driver root: {DRIVER_ROOT}")

    # change to driver root once, before spawning any threads; all relative paths depend on this
    os.chdir(DRIVER_ROOT)

    artifacts_dir = args.artifacts_dir if args.artifacts_dir else os.path.join(DRIVER_ROOT, "artifacts")

    # gather the list of parallelism models to test
    models_to_test = args.include_models if args.include_models else ["serial", "omp", "mpi", "mpi+omp", "kokkos", "cuda", "hip", "pycompss"]
    if args.exclude_models:
        models_to_test = [m for m in models_to_test if m not in args.exclude_models]

    # resolve relaxations
    relaxations = resolve_relaxations(args.relaxations)
    if relaxations:
        logging.info(f"Relaxations enabled: {[r.name for r in relaxations]}")

    # load resource slots for parallel mode
    resource_slots = None
    if args.resource_slots:
        resource_slots = load_json(args.resource_slots)
        logging.info(f"Loaded {len(resource_slots)} resource slots from {args.resource_slots}.")

    # build the list of prompts to run
    prompts_to_run = []
    for prompt in data:
        if prompt["parallelism_model"] not in models_to_test:
            logging.debug(f"Skipping prompt {prompt['name']} because it uses {prompt['parallelism_model']}.")
            continue

        if args.problem and prompt["name"] != args.problem:
            logging.debug(f"Skipping prompt {prompt['name']} because it is not {args.problem}.")
            continue

        if args.problem_type and prompt["problem_type"] != args.problem_type:
            logging.debug(f"Skipping prompt {prompt['name']} because it is not {args.problem_type}.")
            continue

        if already_has_results(prompt):
            if args.overwrite:
                logging.debug(f"Prompt {prompt['name']} already has results. Overwriting.")
                prompt["outputs"] = [p["generated_output"] for p in prompt["outputs"]]
            else:
                logging.debug(f"Skipping prompt {prompt['name']} because it already has results.")
                continue
        prompts_to_run.append(prompt)

    driver_kwargs = dict(
        launch_configs=launch_configs,
        build_configs=build_configs,
        problem_sizes=problem_sizes,
        scratch_dir=args.scratch_dir,
        dry=args.dry,
        display_build_errors=args.log_build_errors,
        display_runs=args.log_runs,
        early_exit_runs=args.early_exit_runs,
        build_timeout=args.build_timeout,
        run_timeout=args.run_timeout,
        save_generated_dir=artifacts_dir,
        relaxations=relaxations,
        model_name=model_name,
    )

    def write_output():
        if args.output and args.output != '-':
            with open(args.output, "w") as fp:
                json.dump(data, fp, indent=4)

    if resource_slots:
        # parallel mode: thread pool + resource queue
        json_lock = threading.Lock()
        rq = queue.Queue()
        for slot in resource_slots:
            rq.put(slot)

        n_workers = rq.qsize()
        logging.info(f"Parallel mode: {n_workers} workers ({n_workers} resource slots).")

        # replace root logging handler with a buffering wrapper so each thread's output
        # is flushed atomically when its prompt completes
        root_logger = logging.getLogger()
        real_handler = root_logger.handlers[0]
        buf_handler = ThreadBufferingHandler(real_handler)
        buf_handler.setLevel(real_handler.level)
        root_logger.removeHandler(real_handler)
        root_logger.addHandler(buf_handler)

        progress = tqdm(total=len(prompts_to_run), desc="Testing prompts", disable=args.hide_progress)

        def process_prompt(prompt):
            threading.current_thread().name = prompt["name"]
            slot = rq.get()
            try:
                driver = get_driver(prompt, resource_slot=slot, **driver_kwargs)
                driver.test_all_outputs_in_prompt(prompt)
            finally:
                rq.put(slot)
                buf_handler.flush_thread()
            with json_lock:
                write_output()
                if not args.hide_progress:
                    progress.update(1)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_prompt, p): p for p in prompts_to_run}
            for f in as_completed(futures):
                f.result()

        progress.close()

    else:
        # sequential mode (default / laptop)
        all_prompts = prompts_to_run if args.hide_progress else tqdm(prompts_to_run, desc="Testing prompts")
        for prompt in all_prompts:
            driver = get_driver(prompt, **driver_kwargs)
            driver.test_all_outputs_in_prompt(prompt)
            write_output()
            logging.debug(f"Wrote intermediate results to {args.output}.")

    # write final results
    if args.output and args.output != '-':
        with open(args.output, "w") as fp:
            json.dump(data, fp, indent=4)
        logging.info(f"Wrote results to {args.output}.")
    else:
        print(json.dumps(data, indent=4))

if __name__ == "__main__":
    main()
