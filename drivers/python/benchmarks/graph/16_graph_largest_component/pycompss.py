# Driver for 16_graph_largest_component
# """ Find the largest connected component in a graph represented by an adjacency matrix.
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
        self.reset_data()

    def reset_data(self):
        # Initialize the matrix
        self.A = [[0] * self.size for _ in range(self.size)]

        # Fill the adjacency matrix with random 0s and 1s using your function
        for i in range(self.size):
            fillRand(self.A[i], 0, 2)

        # Force the matrix to be symmetric (undirected)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                # Overwrite the lower half with the upper half
                self.A[j][i] = self.A[i][j] 
                
            self.A[i][i] = 0
            
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
    return main(ctx.A)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.A)

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
        par_res = main(ctx.A)
        
        # Sequential execution
        seq_res = correct_main(ctx.A)
        
        # Check equality
        if abs(par_res - seq_res) > 1e-6:
            return False
            
    return True