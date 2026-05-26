import json
import logging
import os
import signal
from os import PathLike
import shlex
import subprocess
from subprocess import CompletedProcess
from typing import Optional


def all_equal(iterable) -> bool:
    """ Returns true if all values in iterable are equal """
    return len(set(iterable)) <= 1

def await_input(prompt: str, is_valid_input) -> str:
    """ Repeatedly ask the user for input until it is valid. """
    response = input(prompt)
    while not is_valid_input(response):
        response = input(prompt)
    return response

def load_json(fpath: PathLike) -> dict:
    """ Load the given json file into a dict """
    with open(fpath, "r") as fp:
        return json.load(fp)

def mean(iterable) -> float:
    """ Returns the mean of the given iterable """
    if not hasattr(iterable, "__len__"):
        iterable = list(iterable)
    return sum(iterable) / len(iterable) if len(iterable) > 0 else 0

def run_command(cmd: str, timeout: Optional[int] = None, dry: bool = False, parallelism_model: Optional[str] = None) -> CompletedProcess:
    """ Run the given command on the system and return the result """
    logging.debug(f"Running command: {cmd}")
    if dry:
        return CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    cmd_list = shlex.split(cmd)
    if parallelism_model == "pycompss":
        # COMPSs spawns worker JVMs that outlive the runcompss master process.
        # Put the whole tree in its own process group so stragglers are killed
        # when the master exits, preventing gradual heap accumulation.
        proc = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        pgid = proc.pid  # after setsid, child's PGID == its own PID
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(cmd_list, timeout,
                                            output=stdout.encode(),
                                            stderr=stderr.encode())
        finally:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
        return CompletedProcess(cmd_list, proc.returncode, stdout, stderr)
    else:
        return subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout)
