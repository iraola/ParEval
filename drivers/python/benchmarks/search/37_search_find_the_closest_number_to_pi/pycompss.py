# Driver for 37_search_find_the_closest_number_to_pi
# """ Find the closest number to pi in an array.
#     Use PyCOMPSs to compute in parallel.
# """


from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.api.api import compss_wait_on
import random
from python.utilities import fillRand, fequal, fillRandString

# --- CONTEXT -----------------------------------------------------

class Context:
    def __init__(self, size=5):
        self.size = size
        self.x = [0.0] * size

def reset(ctx):
    """Reset context with random data"""
    fillRand(ctx.x, -50.0, 50.0)

def init():
    """Initialize context with default size"""
    ctx = Context(size=5)
    reset(ctx)
    return ctx

# --- COMPUTE -----------------------------------------------------

def compute(ctx):
    """
    Run the generated PyCOMPSs task.
    findLastShortBook returns a PyCOMPSs Future → must wait later.
    """
    findClosestToPi(ctx.x)

def best(ctx):
    """
    Run sequential baseline version
    """
    correct_findClosestToPi(ctx.x)

# --- VALIDATE -----------------------------------------------------

def validate(ctx):
    for _ in range(5):
        # Create test data with random books
        test = [0.0] * 5
        fillRand(test, -50.0, 50.0)
        
        # Compute reference (sequential)
        correct_result = correct_findClosestToPi(test)

        # Compute PyCOMPSs version
        test_result = findClosestToPi(test)
        
        # Compare results (bool)
        return correct_result == test_result

# --- DESTROY -----------------------------------------------------

def destroy(ctx):
    del ctx
