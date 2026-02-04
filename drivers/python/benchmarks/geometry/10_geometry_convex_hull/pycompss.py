# Driver for 10_geometry_convex_hull
# """ Compute the convex hull of a set of points.

import random
# --- CONTEXT CLASS -----------------------------------------------

class Context:
    """
    Holds the state for the benchmark
    """
    def __init__(self, size=10):
        try:
            self.size = DRIVER_PROBLEM_SIZE
        except NameError:
            self.size = size

        self.points = []
        self.reset_data()

    def reset_data(self):
        self.points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(self.size)]
# --- DRIVER INTERFACE --------------------------------------------

def init():
    """ 
    Initializes context as a Class Instance.
    """
    return Context()

def reset(ctx: Context):
    """
    Wrapper to call the class method. 
    Maintains compatibility with the generic driver.
    """
    ctx.reset_data()

# --- COMPUTE -----------------------------------------------------

def compute(ctx: Context):
    """
    Launches parallel ReLU tasks.
    Accesses data via dot notation (ctx.x).
    """
    # Assuming 'relu' is the @task defined elsewhere
    return main(ctx.points)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_relu' is defined elsewhere
    return correct_main(ctx.points)

def points_equal(p1, p2, eps=1e-6):
    # Handles comparing the individual points
    return abs(p1[0] - p2[0]) <= eps and abs(p1[1] - p2[1]) <= eps

def hulls_equal(hull_a, hull_b, eps=1e-6):
    if len(hull_a) != len(hull_b):
        return False
    
    # 1. Normalize: find the 'lexicographically smallest' point to start from
    # (e.g., the point with the smallest X, then smallest Y)
    start_point = min(hull_a)
    
    # 2. Reorder hull_a to start from that point
    idx = hull_a.index(start_point)
    hull_a = hull_a[idx:] + hull_a[:idx]
    
    # 3. Check both orientations (CW and CCW) against hull_b
    # This handles the "direction" problem
    def check_match(ha, hb):
        return all(points_equal(p1, p2, eps) for p1, p2 in zip(ha, hb))

    # Try forward match
    if check_match(hull_a, hull_b): return True
    # Try reverse match
    if check_match(hull_a, hull_b[::-1]): return True
    
    return False

# --- VALIDATE ----------------------------------------------------

def validate(ctx: Context):
    """ Verifies parallel execution matches sequential execution. """
    try:
        max_attempts = MAX_VALIDATION_ATTEMPTS
    except NameError:
        max_attempts = 5
        
    for _ in range(max_attempts):
        # Reset the data within the context for a new validation run
        reset(ctx)
        
        # Create copies to avoid modifying the source data in place during comparison
        # if the tasks operate in-place.
        par_res = ctx.points[:]
        seq_res = ctx.points[:]

        # Parallel execution
        par_res = main(par_res)
        
        # Sequential execution
        seq_res = correct_main(seq_res)
        
        # Check equality
        if not hulls_equal(par_res, seq_res, eps=1e-6):
            return False
            
    return True