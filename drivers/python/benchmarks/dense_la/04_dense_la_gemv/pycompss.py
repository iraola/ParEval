# Driver for 04_dense_la_gemv
# """ Perform GEMV operation.
# """

import random
from python.utilities import fillRand, fequal 

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
        self.x = []
        self.reset_data()

    def reset_data(self):
        self.A = [[0.0] * self.size for _ in range(self.size)]
        for row in self.A:
            fillRand(row, -10.0, 10.0)
        self.x = [0.0] * self.size
        fillRand(self.x, -10.0, 10.0)
        


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


def compute(ctx: Context):
    # Assuming 'main' is the @task defined elsewhere
    return main(ctx.A, ctx.x)

def best(ctx: Context):
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.A, ctx.x)

def validate(ctx: Context):
    """ Verifies parallel execution matches sequential execution. """
    try:
        max_attempts = MAX_VALIDATION_ATTEMPTS
    except NameError:
        max_attempts = 5
        
    for _ in range(max_attempts):
        # Reset the data within the context for a new validation run
        reset(ctx)

        print(ctx.A)
        # Parallel execution
        par_res = main(ctx.A, ctx.x)
        print(ctx.A)
        # Sequential execution
        seq_res = correct_main(ctx.A, ctx.x)

        print("Parallel Result: ", par_res)
        print("Sequential Result: ", seq_res)
        # Check equality
        if not fequal(par_res, seq_res, eps=1e-6):
            return False
            
    return True