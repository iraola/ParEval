# Driver for 39_search_xor_contains
# """ Check if an array contains a specific value, but only in one of the two arrays (XOR condition).
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
        self.y = []
        self.val = None
        # Initialize data immediately
        self.reset_data()

    def reset_data(self):
        """Refills the list with random integers."""
        self.x = [0] * self.size
        self.y = [0] * self.size
        self.val = random.randint(-30, 30)
        fillRand(self.x, -30, 30)
        fillRand(self.y, -30, 30)
        

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
    return main(ctx.x, ctx.y, ctx.val)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.x, ctx.y, ctx.val)

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
        par_res = main(ctx.x, ctx.y, ctx.val)
        
        # Sequential execution
        seq_res = correct_main(ctx.x, ctx.y, ctx.val)
        
        # Check equality
        if par_res != seq_res:
            return False
            
    return True