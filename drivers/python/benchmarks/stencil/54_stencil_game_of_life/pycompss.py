# Driver for 54_stencil_game_of_life
# """ Perform Game of Life with 8 potential neighbors in a 2D grid. A cell becomes alive if it has exactly 3 live neighbors, and remains alive if it has 2 or 3 live neighbors. Otherwise, it dies.
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

        self.input = []
        self.N = self.size
        self.reset_data()

    def reset_data(self):
        self.input = [[0] * self.N for _ in range(self.N)]

        # Fill the adjacency matrix with random 0s and 1s
        for i in range(len(self.input)):
            fillRand(self.input[i], 0, 2)  # Random integers 0 or 1

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
    return main(ctx.input)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.input)


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
        if not par_res == seq_res:
            return False
            
    return True