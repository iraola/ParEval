# Driver for 02_dense_la_gemm
# """ Perform matrix multiplication.
# """

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

        self.A = []
        self.B = []
        self.reset_data()

    def reset_data(self):
        self.A = [[0.0] * self.size for _ in range(self.size)]
        for row in self.A:
            fillRand(row, -10.0, 10.0)
            
        self.B = [[0.0] * self.size for _ in range(self.size)]
        for row in self.B:
            fillRand(row, -10.0, 10.0)
        

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
    return main(ctx.A, ctx.B)

def best(ctx: Context):
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.A, ctx.B)
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
        par_res = main(ctx.A, ctx.B)
        
        # Sequential execution
        seq_res = correct_main(ctx.A, ctx.B)

        # Check equality
        for i in range (len(seq_res)):
            if not fequal(par_res[i], seq_res[i], eps=1e-6):
                return False
            
    return True