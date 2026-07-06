import sys
import time
import random
sys.setrecursionlimit(100000)

NITER = 3
MAX_VALIDATION_ATTEMPTS = 2

def _seeded_reset(ctx, i):
    """Seed the RNG before resetting data for timing iteration ``i``.

    Ensures every num_procs run (and the sequential baseline) times the
    same inputs, matching the deterministic behaviour of the original C++
    benchmark (whose unseeded ``rand()`` produced a fixed sequence). Falls
    back to a fixed seed if the config has no DRIVER_SEED.
    """
    try:
        base = DRIVER_SEED
    except NameError:
        base = 1234
    random.seed(base + i)
    reset(ctx)

def run_benchmark():
    # Setup
    ctx = init()

    # Validation
    print("Validating...")
    if not validate(ctx):
        print("Validation: FAIL")
        return
    print("Validation: PASS")

    # Benchmark parallel
    print("Running compute() benchmark...")
    total_parallel = 0.0
    for i in range(NITER):
        _seeded_reset(ctx, i)
        start = time.perf_counter()

        compute(ctx)
        iteration_time = time.perf_counter() - start
        total_parallel += iteration_time
        print(f"Parallel Iteration time: {iteration_time:.6f}")
    
    print(f"Time: {total_parallel / NITER:.6f}")

    # Benchmark sequential
    print("Running best() benchmark...")
    total_seq = 0.0
    for i in range(NITER):
        _seeded_reset(ctx, i)
        start = time.perf_counter()

        best(ctx)
        iteration_time = time.perf_counter() - start
        total_seq += iteration_time
        print(f"Sequential Iteration time: {iteration_time:.6f}")

    print(f"BestSequential: {total_seq / NITER:.6f}")

if __name__ == "__main__":
    run_benchmark()