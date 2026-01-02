import time
NITER = 5

def run_benchmark():
    # 1. Setup
    ctx = init()

    # 2. Validation (Single point of failure)
    print("Validating...")
    if not validate(ctx):
        print("Validation: FAIL")
        return
    print("Validation: PASS")

    # 3. Benchmark Parallel (Compute)
    print("Running compute() benchmark...")
    total_parallel = 0.0
    for _ in range(NITER):
        reset(ctx)
        start = time.time()
      
        result = compute(ctx)
        
        total_parallel += (time.time() - start)
    
    print(f"Time: {total_parallel / NITER:.6f}")

    # 4. Benchmark Sequential (Best)
    print("Running best() benchmark...")
    total_seq = 0.0
    for _ in range(NITER):
        reset(ctx)
        start = time.time()
        
        best(ctx)
        
        total_seq += (time.time() - start)

    print(f"BestSequential: {total_seq / NITER:.6f}")

if __name__ == "__main__":
    run_benchmark()