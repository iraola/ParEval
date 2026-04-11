#!/usr/bin/env python3
"""
Estimate runtimes for each benchmark problem in an examples repository.

Walks repo_path/category/problem/ directories, runs each solution with plain
Python, and reports wall time.

Usage:
  python bin/estimate_runtimes.py /path/to/examples/repo
  python bin/estimate_runtimes.py /path/to/examples/repo --timeout 60
  python bin/estimate_runtimes.py /path/to/examples/repo --problem graph/count_components
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

DEFAULT_TIMEOUT = 30


def find_solution_file(problem_dir: Path) -> Path | None:
    for candidate in ["solution.py", "main.py"]:
        f = problem_dir / candidate
        if f.exists():
            return f
    dir_name = problem_dir.name.replace("-", "_")
    for f in sorted(problem_dir.glob("*.py")):
        if f.name == "__init__.py":
            continue
        if f.stem.replace("-", "_") == dir_name:
            return f
    py_files = [f for f in sorted(problem_dir.glob("*.py")) if f.name != "__init__.py"]
    return py_files[0] if len(py_files) == 1 else None


def discover_problems(repo_path: Path) -> list[tuple[str, str, Path]]:
    problems = []
    for cat_dir in sorted(repo_path.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("."):
            continue
        for prob_dir in sorted(cat_dir.iterdir()):
            if not prob_dir.is_dir() or prob_dir.name.startswith("."):
                continue
            problems.append((cat_dir.name, prob_dir.name, prob_dir))
    return problems


def run_problem(solution: Path, timeout: float) -> tuple[float | None, str]:
    try:
        t0 = time.perf_counter()
        result = subprocess.run(
            [PYTHON, str(solution)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(solution.parent),
        )
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            return None, result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "non-zero exit"
        return elapsed, ""
    except subprocess.TimeoutExpired:
        return None, f"timeout (>{timeout}s)"
    except Exception as e:
        return None, str(e)


def fmt_time(t: float) -> str:
    if t < 1.0:
        return f"{t * 1000:.0f}ms"
    return f"{t:.2f}s"


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo_path", type=Path)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="Seconds before aborting a run (default: %(default)s).")
    p.add_argument("--problem", type=str, default=None, metavar="CATEGORY/NAME",
                   help="Only run this problem, e.g. graph/count_components.")
    return p.parse_args()


def main() -> None:
    args = get_args()
    repo_path = args.repo_path.resolve()
    if not repo_path.exists():
        print(f"error: {repo_path} does not exist", file=sys.stderr)
        sys.exit(1)

    problems = discover_problems(repo_path)
    if args.problem:
        cat, name = args.problem.split("/", 1)
        problems = [(c, n, d) for c, n, d in problems if c == cat and n == name]
        if not problems:
            print(f"error: '{args.problem}' not found", file=sys.stderr)
            sys.exit(1)

    print(f"{'Problem':<50}  {'Time':>8}  Note")
    print("-" * 72)

    for category, name, problem_dir in problems:
        label = f"{category}/{name}"
        solution = find_solution_file(problem_dir)
        if solution is None:
            print(f"{label:<50}  {'---':>8}  no solution file found")
            continue

        elapsed, err = run_problem(solution, args.timeout)
        if elapsed is None:
            print(f"{label:<50}  {'---':>8}  {err}")
        else:
            print(f"{label:<50}  {fmt_time(elapsed):>8}")


if __name__ == "__main__":
    main()
