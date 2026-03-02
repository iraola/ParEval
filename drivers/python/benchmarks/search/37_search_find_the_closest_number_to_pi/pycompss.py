# Driver for 37_search_find_the_closest_number_to_pi
# """ Find the number in an array closest to pi.
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

        self.x = []
        # Initialize data immediately
        self.reset_data()

    def reset_data(self):
        """Refills the list with random floating-point numbers."""
        self.x = [0.0] * self.size
        fillRand(self.x, -10.0, 10.0)

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
    """
    Launches parallel tasks to find the number in the array closest to pi.
    Accesses data via dot notation (ctx.x).
    """
    return main(ctx.x)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.x)

# --- VALIDATE ----------------------------------------------------

def validate(ctx: Context):
    """ Verifies parallel tasks match sequential tasks. """
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
        if par_res != seq_res:
            return False
            
    return True