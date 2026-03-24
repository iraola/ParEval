# Driver for 26_reduce_product_of_inverses
# """ Compute the product of even elements and inverses of odd elements in a list.

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
        self.reset_data()

    def reset_data(self):
        self.x = [0] * self.size
        fillRand(self.x, -10, 11)
        if 0 in self.x:
            # Ensure no zeros to avoid division by zero in the product of inverses
            self.x = [val if val != 0 else 1 for val in self.x]

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
    Launches parallel tasks to compute the product of even elements and inverses of odd elements in a list.
    Accesses data via dot notation (ctx.x).
    """
    # Assuming 'main' is the @task defined elsewhere
    return main(ctx.x)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
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

        # Parallel execution
        par_res = main(ctx.x)
        
        # Sequential execution
        seq_res = correct_main(ctx.x)
        
        # Check equality
        if abs(par_res - seq_res) > 1e-9:
            return False
            
    return True