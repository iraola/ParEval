# Driver for 05_fft_inverse_fft
# """ Compute the inverse FFT of a set of complex numbers.
# """

import random
from python.utilities import fillRand, fequal 
import copy

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
        self.N = self.size
        self.reset_data()

    def reset_data(self):
        self.A = [[0.0] * self.N for _ in range(self.N)]
        for row in self.A:
            fillRand(row, -50.0, 50.0)


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
    """
    Launches parallel ReLU tasks.
    Accesses data via dot notation (ctx.x).
    """
    # Assuming 'relu' is the @task defined elsewhere
    return main(ctx.A, ctx.N)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_relu' is defined elsewhere
    return correct_main(ctx.A, ctx.N)


def validate(ctx: Context):
    """ Verifies parallel execution matches sequential execution. """
    try:
        max_attempts = MAX_VALIDATION_ATTEMPTS
    except NameError:
        max_attempts = 5
        
    for _ in range(max_attempts):
        # Reset the data within the context for a new validation run
        reset(ctx)
        
        ctx_par = copy.deepcopy(ctx)

        # Parallel execution
        par_res = main(ctx_par.A, ctx_par.N)
        
        # Sequential execution
        seq_res = correct_main(ctx.A, ctx.N)

        # Check equality
        for i in range (len(par_res)):
            if not fequal(par_res[i], seq_res[i], eps=1e-6):
                return False
            
    return True