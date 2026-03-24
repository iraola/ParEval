# Driver for 51_stencil_edge_kernel
# """ Perform dot product of each element and its neighbours with the edge kernel which has 8 in the center and -1 in the surrounding positions.
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

        self.matrix = []
        self.N = self.size
        self.reset_data()

    def reset_data(self):
        self.matrix = [[0] * self.N for _ in range(self.N)]

        for i in range(len(self.matrix)):
            fillRand(self.matrix[i], 0, 256)  # Random integers 0 to 255

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
    Launches parallel tasks.
    Accesses data via dot notation (ctx.x).
    """
    # Assuming 'main' is the @task defined elsewhere
    return main(ctx.matrix, ctx.N)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.matrix, ctx.N)


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
        par_res = main(ctx.matrix, ctx.N)
        
        # Sequential execution
        seq_res = correct_main(ctx.matrix, ctx.N)

        # Check equality
        for i in range(len(par_res)):
                if not fequal(par_res[i], seq_res[i]):
                    return False
            
    return True