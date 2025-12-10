# Driver for 35_search_search_for_last_struct_by_key
# """ Find last book with less than 100 pages.
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
        self.x = [0] * size
        self.target = 0

def reset(ctx):
    """Reset context with random data"""
    fillRand(ctx.x, 0, 20)
    ctx.target = random.randint(0, 20)

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
    contains(ctx.x, ctx.target)

def best(ctx):
    """
    Run sequential baseline version
    """
    correct_contains(ctx.x, ctx.target)

# --- VALIDATE -----------------------------------------------------

def validate(ctx):
    for _ in range(5):
        # Create test data with random books
        test_x = [0] * 5
        fillRand(test_x, 0, 20)
        test_target = random.randint(0, 20)
        
        # Compute reference (sequential)
        correct_result = correct_contains(test_x, test_target)

        # Compute PyCOMPSs version
        test_result = contains(test_x, test_target)
        
        # Compare results (bool)
        return correct_result == test_result

# --- DESTROY -----------------------------------------------------

def destroy(ctx):
    del ctx
