"""
PyCOMPSs driver
"""

import time

NITER = 5

def run_benchmark():

    # --- INIT ---
    ctx = init()

    # --- VALIDATE ---
    print("Validating...")
    is_valid = validate(ctx)
    print("Validation:", "PASS" if is_valid else "FAIL")

    if not is_valid:
        destroy(ctx)
        return

    # --- BENCHMARK COMPUTE ---
    total_time = 0.0
    print("Running compute() benchmark...")

    for _ in range(NITER):
        start = time.time()
        compute(ctx)

        total_time += time.time() - start
        reset(ctx)

    print("Time:", total_time / NITER)

    # --- BENCHMARK BEST SEQUENTIAL ---
    total_time = 0.0
    print("Running best() benchmark...")

    for _ in range(NITER):
        start = time.time()
        best(ctx)
        total_time += time.time() - start
        reset(ctx)

    print("BestSequential:", total_time / NITER)

    # --- CLEANUP ---
    destroy(ctx)


if __name__ == "__main__":
    run_benchmark()
