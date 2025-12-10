# Driver for 39_search_xor_contains
# """ Find if value is contained in exactly one of two arrays.
#     Use PyCOMPSs to compute in parallel.
# """


from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.api.api import compss_wait_on
import random
from python.utilities import fillRand, fequal, fillRandString

# --- CONTEXT -----------------------------------------------------

class Context:
    def __init__(self, size_x=5, size_y=6):
        self.size_x = size_x
        self.size_y = size_y
        self.x = [0] * size_x
        self.y = [0] * size_y
        self.val = 0

def reset(ctx):
    """Reset context with random data"""
    fillRand(ctx.x, 0, 20)
    fillRand(ctx.y, 0, 20)
    ctx.val = random.randint(0, 20)
def init():
    """Initialize context with default size"""
    ctx = Context(size_x=5, size_y=6)
    reset(ctx)
    return ctx

# --- COMPUTE -----------------------------------------------------

def compute(ctx):
    """
    Run the generated PyCOMPSs task.
    findLastShortBook returns a PyCOMPSs Future → must wait later.
    """
    xorContains(ctx.x, ctx.y, ctx.val)

def best(ctx):
    """
    Run sequential baseline version
    """
    correct_xorContains(ctx.x, ctx.y, ctx.val)

# --- VALIDATE -----------------------------------------------------

def validate(ctx):
    for _ in range(5):
        # Create test data with random books
        test_x = [0] * 5
        test_y = [0] * 6
        fillRand(test_x, 0, 20)
        fillRand(test_y, 0, 20)
        val = random.randint(0, 20)
        
        # Compute reference (sequential)
        correct_result = correct_xorContains(test_x, test_y, val)
        # Compute PyCOMPSs version
        test_result = xorContains(test_x, test_y, val)
        
        # Compare results (bool)
        return correct_result == test_result

# --- DESTROY -----------------------------------------------------

def destroy(ctx):
    del ctx
