# Driver for 36_search_check_if_array_contains_value
# """ Check if an array contains a specific value.
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
        self.target = None
        # Initialize data immediately
        self.reset_data()

    def reset_data(self):
        """Refills the list with random integers and sets a random target value."""
        self.x = [0] * self.size
        fillRand(self.x, -10, 10)
        self.target = random.randint(-10, 10)

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
    Launches parallel tasks to check if the array contains the target value.
    Accesses data via dot notation (ctx.x) and the target via ctx.target.
    """
    return main(ctx.x, ctx.target)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.x, ctx.target)

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
        par_res = main(ctx.x, ctx.target)
        
        # Sequential execution
        seq_res = correct_main(ctx.x, ctx.target)
        
        # Check equality
        if par_res != seq_res:
            return False
            
    return True