# Driver for 40_sort_sort_an_array_of_complex_numbers_by_magnitude
# """ Sort an array of complex numbers by their magnitude.

import random
from python.utilities import fillRand, fequal 
# --- CONTEXT CLASS -----------------------------------------------

class Context:
    """
    Holds the state for the benchmark
    """
    def __init__(self, size=10):
        self.size = size
        self.x = []
        # Initialize data immediately
        self.reset_data()

    def reset_data(self):
        """
        Refills the list with random complex numbers.
        Constructs complex numbers using random floats for real and imag parts.
        """
        self.x = []
        for _ in range(self.size):
            # Generate random real and imaginary components
            r = random.uniform(-50.0, 50.0)
            i = random.uniform(-50.0, 50.0)
            self.x.append(complex(r, i))

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

# --- VALIDATE ----------------------------------------------------

def validate(ctx: Context):
    """ Verifies parallel ReLU matches sequential ReLU. """
    for _ in range(5):
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