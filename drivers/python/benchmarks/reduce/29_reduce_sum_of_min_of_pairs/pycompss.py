# Driver for 29_reduce_sum_of_min_of_pairs
# """ Compute the sum of the minimum of pairs of elements in two lists.

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

        self.x = []
        self.y = []
        self.reset_data()

    def reset_data(self):
        self.x = [0] * self.size
        self.y = [0] * self.size
        fillRand(self.x, 0, 50)
        fillRand(self.y, 0, 50)

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

    # Assuming 'main' is the @task defined elsewhere
    return main(ctx.x, ctx.y)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.x, ctx.y)

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

        # Parallel execution
        par_res = main(ctx.x, ctx.y)
        
        # Sequential execution
        seq_res = correct_main(ctx.x, ctx.y)
        
        # Check equality
        if abs(par_res - seq_res) > 1e-9:
            return False
            
    return True