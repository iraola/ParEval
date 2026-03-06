import pathlib
import json
import logging
import asyncio
import re
import tempfile

logger = logging.getLogger(__name__)

class Runner(object):
    """
    Runner class to execute tasks in parallel using asyncio and subprocesses.
    A task is defined by a code snippet and a name, and the runner executes
    the code with arguments specified in a JSON file. The execution results are saved
    in a specified directory.
    """

    def __init__(self, exec_command: str, args_file: str, *, base_dir: str = "."):
        """
        Initializes the Runner with the command to execute, the path to the arguments file,
        and the base directory for resolving input/output paths.

        The args_file contains the macros `INPUT_FILE`, `INPUT_DIR`, `OUTPUT_FILE`, and `OUTPUT_DIR` followed by a
        relative path. Paths with the macro `INPUT_FILE` or `INPUT_DIR` will be resolved according the base directory.
        Paths with the macro `OUTPUT_FILE` or `OUTPUT_DIR` will be resolved to a temporary directory. See
        _process_argument for more details on how the arguments are processed.

        Args:
            exec_command (str): The command to execute the code snippets.
            args_file (str): Path to the JSON file containing arguments for each task.
            base_dir (str): The base directory to resolve input/output file paths.
        Raises:
            ValueError: If the arguments file does not exist or is not a file.
        """
        args_path: pathlib.Path = pathlib.Path(args_file)
        if not args_path.exists() or not args_path.is_file():
            raise ValueError(f"File {args_file} does not exist or is not a file.")
        self.args: dict[str, str] = json.loads(args_path.read_text())
        self.exec_command: list[str] = exec_command.split()
        self.tasks: list[tuple[str, str]] = []
        self.base_dir: pathlib.Path = pathlib.Path(base_dir).resolve()
        self.save_dir: pathlib.Path | None = None # Directory where execution results (stderr, stdout) will be saved
        self.waiting_for: list[str] = [] # List of task names that are still waiting for completion
        logger.info(f"Initialized with {self.exec_command = }, {args_file = }, {self.base_dir = }")

    def add_task(self, code: str, name: str) -> None:
        """
        Adds a task to the runner with the given code and name. The code is expected to be a string
        representing the code snippet to be executed, and the name is a unique identifier for the task.
        Args:
            code (str): The code snippet to be executed.
            name (str): A unique name for the task.
        Raises:
            ValueError: If a task with the same name already exists.
        """
        if any(name == task_name for _, task_name in self.tasks):
            raise ValueError(f"Task with name {name} already exists.")
        self.tasks.append((code, name))
        logger.info(f"Added task for code with name: {name}")

    def run_tasks(self, save_dir: str) -> None:
        """
        Method to bridge the synchronous interface to the asynchronous runner. Results will be saved in the specified
        directory.
        Args:
            save_dir (str): The directory where the execution results will be saved.
        """
        self.save_dir = pathlib.Path(save_dir)
        asyncio.run(self._run_tasks())


    async def _run_tasks(self) -> None:
        """
        Runs the tasks asynchronously using asyncio and subprocesses. Each task is executed in a separate subprocess,
        and the results (stdout and stderr) are collected in a queue. The results are then processed by a consumer
        that saves them to a JSON file in the specified save directory.
        The tasks are expected to be added using the `add_task` method before calling this method.
        """
        logger.info(f"Running {len(self.tasks)} tasks in asynchronous mode.")
        self.waiting_for = [name for _, name in self.tasks] # All tasks are pending to finish
        logger.info(f"Waiting for {self.waiting_for} tasks to complete.")
        queue = asyncio.Queue()
        await asyncio.gather(
            *[self._run_subprocesses(queue, code, name) for code, name in self.tasks],
            self._consumer(queue),
        )

    async def _run_subprocesses(self, queue, code: str, name: str) -> None:
        """
        Runs a subprocess for the given code and name, processing the arguments from the args dictionary.
        The code is expected to be a string representing the code snippet to be executed, and the name is a unique identifier for the task.
        The arguments for the subprocess are processed using the `_process_argument` method, which resolves input/output paths based on the base directory.

        If no arguments are provided for the task, it puts a tuple with the name and None in the queue, which will be
        treated separately by the consumer.
        Args:
            queue (asyncio.Queue): The queue to put the results of the subprocess execution.
            code (str): The code snippet to be executed.
            name (str): A unique name for the task.
        """
        if name in self.args:
            args = self.args[name]
            precessed_args = [self._process_argument(arg) for arg in args]
            logger.info(f"Start process for {name} with arguments: {precessed_args}")
            subprocess_args = self.exec_command + [code] + precessed_args
            process = await asyncio.create_subprocess_exec(*subprocess_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            await queue.put(
                (name, stdout.decode(), stderr.decode())
            )
        else:
            await queue.put((name, None))
            logger.info(f"No arguments found for {name}")

    async def _consumer(self, queue: asyncio.Queue) -> None:
        """
        Consumer that processes the results from the queue. Progressively collects the results of the tasks
        executed by the subprocesses and saves them to a JSON file in the specified save directory.
        The results are stored in a dictionary with the task name as the key and a dictionary containing
        the stdout and stderr as the value. If no arguments were provided for a task, it
        saves an empty stdout and a message indicating no arguments were provided in the stderr.
        Args:
            queue (asyncio.Queue): The queue from which to consume the results of the subprocess executions
        """
        execution_results = {}
        num_tasks = len(self.tasks)
        i = 0
        while i < num_tasks:
            result = await queue.get()
            logger.info(f"Consumer received result {i+1}/{num_tasks}: {result[0]}")
            if result[1] is None:
                execution_results[result[0]] = {"stdout": "", "stderr": "No arguments provided for this task."}
            else:
                execution_results[result[0]] = {"stdout": result[1], "stderr": result[2]}
            i += 1
            self.waiting_for.remove(result[0])
            logger.info(f"Waiting for {self.waiting_for} tasks to complete.")
        with open(self.save_dir / "execution_results.json", "w") as f:
            json.dump(execution_results, f, indent=4)
        logger.info(f"Execution results saved to {self.save_dir / 'execution_results.json'}")

    def _process_argument(self, arg: str) -> str:
        """
        Processes the argument string to resolve input/output paths based on the base directory.
        The argument is expected to be a string that starts with a macro like `INPUT_FILE`,
        `INPUT_DIR`, `OUTPUT_FILE`, or `OUTPUT_DIR` followed by a relative path.
        If the argument matches one of these macros, it resolves the path accordingly:
        - `INPUT_FILE` or `INPUT_DIR`: Resolves to the base directory followed by the relative path.
        - `OUTPUT_FILE` or `OUTPUT_DIR`: Resolves to a temporary directory followed by the relative path. The
        temporary directory can be found later by the user (look at logging.log for the path).
        If the argument does not match any of these macros, it is returned as a pure string.
        Args:
            arg (str): The argument string to be processed.
        Returns:
            str: The processed argument string with resolved paths.
        Raises:
            ValueError: If the argument does not match with an unexpected macro.
        Example:
            >>> runner = Runner("python -c", "args.json", base_dir="/path/to/base") # Make sure args.json exists
            >>> runner._process_argument("`INPUT_FILE`data/input.txt") # will return "/path/to/base/data/input.txt"
            >>> runner._process_argument("`OUTPUT_DIR`results/output.txt") # will return a temporary path like "/tmp/tmp123456/results/output.txt"
            >>> runner._process_argument("Just a string") # will return "Just a string"

        """
        match = re.match(r"`([A-Z_]+)`(.*)", arg)
        if match is None:
            logger.info(f"Argument {arg} is pure string.")
            return arg
        elif match.group(1) in ("INPUT_FILE", "INPUT_DIR"):
            path = self.base_dir / pathlib.Path(match.group(2))
            full_path = path.resolve()
            logger.info(f"Argument {arg} is a input file/dir -> {full_path}.")
            return str(full_path)
        elif match.group(1) in ("OUTPUT_FILE", "OUTPUT_DIR"):
            tmpdir = pathlib.Path(tempfile.mkdtemp()) # Temporary directory will not be deleted
            path = pathlib.Path(tmpdir) / match.group(2)
            full_path = path.resolve()
            logger.info(f"Argument {arg} is a output file/dir -> {full_path}.")
            return str(full_path)
        else:
            raise ValueError(f"Argument {arg} is not valid.")