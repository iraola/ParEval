# Driver for 52_stencil_1d_jacobi_3-point_stencil
# """ Perform a 1D Jacobi stencil with a 3-point kernel. Each output element is the average of itself and its immediate neighbors. For boundary elements, missing neighbors are treated as 0.

import random
from python.utilities import fillRand, fequal 
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

        self.input = []
        self.N = self.size
        self.reset_data()

    def reset_data(self):
        self.input = [[0] * self.N for _ in range(self.N)]
        for i in range(len(self.input)):
            fillRand(self.input[i], -50, 50)
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
    Launches parallel tasks.
    Accesses data via dot notation (ctx.x).
    """
    # Assuming 'main' is the @task defined elsewhere
    return main(ctx.input)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.input)

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

        # Parallel execution
        par_res = main(ctx.input)
        
        # Sequential execution
        seq_res = correct_main(ctx.input)

        # Check equality
        for i in range(len(par_res)):
            if not fequal(par_res[i], seq_res[i], 1e-6):
                return False
            
    return True