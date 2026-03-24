# Driver for 05_fft_inverse_fft
# """ Compute the inverse FFT of a set of complex numbers.
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

        self.x = []
        self.real = []
        self.imag = []
        self.reset_data()

    def reset_data(self):
        self.real = [0.0] * self.size
        self.imag = [0.0] * self.size
        fillRand(self.real, -1.0, 1.0)
        fillRand(self.imag, -1.0, 1.0)
        self.x = [complex(r, i) for r, i in zip(self.real, self.imag)]


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
    return main(ctx.x)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_relu' is defined elsewhere
    return correct_main(ctx.x)


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
        par_res = ctx.x[:]
        seq_res = ctx.x[:]

        # Parallel execution
        par_res = main(par_res)
        
        # Sequential execution
        seq_res = correct_main(seq_res)
        
        # Check equality
        if not fequal(par_res, seq_res, eps=1e-6):
            return False
            
    return True