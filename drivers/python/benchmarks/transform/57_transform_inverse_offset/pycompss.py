# Driver for 57_transform_inverse_offset
# """ In the vector x compute 1 - (1 / x) for each element.
#     Use PyCOMPSs to compute in parallel.
# """

from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.api.api import compss_wait_on

from python.utilities import fillRand, fequal

# --- CONTEXT -----------------------------------------------------

class Context:
    def __init__(self, size=5):
        self.size = size
        self.x = [0] * size
        fillRand(self.x, -50, 50)

def reset(ctx):
    fillRand(ctx.x, -50, 50)
def init():
    return Context(size=5)

# --- COMPUTE -----------------------------------------------------

def compute(ctx):
    """
    Run the generated PyCOMPSs task.
    oneMinusInverse returns a PyCOMPSs Future → must wait later.
    """
    ctx.x = oneMinusInverse(ctx.x)

def best(ctx):
    """
    Run sequential baseline version
    """
    correct_oneMinusInverse(ctx.x)

# --- VALIDATE -----------------------------------------------------

def validate(ctx):
    import math

    for _ in range(5):
        test      = [0] * 5
        fillRand(test, -50, 50)

        correct   = test.copy()
        test_comp = test.copy()

        # Compute reference (sequential)
        correct_oneMinusInverse(correct)

        # Compute PyCOMPSs version
        test_result = oneMinusInverse(test_comp)
        # Compare
        if not fequal(correct, test_result, eps=1e-6):
            return False

    return True

# --- DESTROY -----------------------------------------------------

def destroy(ctx):
    del ctx
