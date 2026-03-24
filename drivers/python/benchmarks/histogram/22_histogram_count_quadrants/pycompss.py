# Driver for 22_histogram_count_quadrants
# """ Compute the histogram of an image with 4 bins for each quadrant. """

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

        self.points = []
        self.x = []
        self.y = []
        self.reset_data()

    def reset_data(self):
        self.x = [0-0] * self.size
        self.y = [0.0] * self.size
        fillRand(self.x, -50.0, 50.0)
        fillRand(self.y, -50.0, 50.0)
        self.points = list(zip(self.x, self.y))

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
    return main(ctx.points)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.points)


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
        par_res = main(ctx.points)
        
        # Sequential execution
        seq_res = correct_main(ctx.points)
        
        # Check equality
        if not fequal(par_res, seq_res, eps=1e-6):
            return False
            
    return True