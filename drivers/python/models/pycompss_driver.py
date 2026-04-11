import time
NITER = 5
MAX_VALIDATION_ATTEMPTS = 2

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
    for _ in range(NITER):
        reset(ctx)
        start = time.perf_counter()
      
        compute(ctx)
        iteration_time = time.perf_counter() - start
        total_parallel += iteration_time
        print(f"Parallel Iteration time: {iteration_time:.6f}")
    
    print(f"Time: {total_parallel / NITER:.6f}")

    # Benchmark sequential
    print("Running best() benchmark...")
    total_seq = 0.0
    for _ in range(NITER):
        reset(ctx)
        start = time.perf_counter()
        
        best(ctx)
        iteration_time = time.perf_counter() - start
        total_seq += iteration_time
        print(f"Sequential Iteration time: {iteration_time:.6f}")

    print(f"BestSequential: {total_seq / NITER:.6f}")

if __name__ == "__main__":
    run_benchmark()