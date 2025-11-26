# Driver for 58_transform_squaring
# """ In the vector x compute the square of each element.
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
    squareEach returns a PyCOMPSs Future → must wait later.
    """
    ctx.x = squareEach(ctx.x)

def best(ctx):
    """
    Run sequential baseline version
    """
    correct_squareEach(ctx.x)

# --- VALIDATE -----------------------------------------------------

def validate(ctx):
    import math

    for _ in range(5):
        test      = [0] * 5
        fillRand(test, -50, 50)

        correct   = test.copy()
        test_comp = test.copy()

        # Compute reference (sequential)
        correct_squareEach(correct)

        # Compute PyCOMPSs version
        test_result = squareEach(test_comp)
        # Compare
        if fequal(correct, test_result, eps=1e-6) is False:
            return False

    return True

# --- DESTROY -----------------------------------------------------

def destroy(ctx):
    del ctx
