# Driver for 20_histogram_pixel_histogram
# """ Compute the histogram of pixel intensities in an image.

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

        self.image = []
        self.reset_data()

    def reset_data(self):
        self.image = [0] * self.size
        fillRand(self.image, 0, 256)

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
    return main(ctx.image)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_relu' is defined elsewhere
    return correct_main(ctx.image)


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
        par_res = ctx.image[:]
        seq_res = ctx.image[:]

        # Parallel execution
        par_res = main(par_res)
        
        # Sequential execution
        seq_res = correct_main(seq_res)
        
        # Check equality
        if not fequal(par_res, seq_res, eps=1e-6):
            return False
            
    return True