# Driver for 57_transform_inverse_offset
# """ In the vector x compute 1 - (1 / x) for each element.
#     Use PyCOMPSs to compute in parallel.
# """

import random
from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.api.api import compss_wait_on

# --- CONTEXT -----------------------------------------------------

class Context:
    def __init__(self, size=5):
        self.size = size
        self.x = [random.randint(-50, 50) for _ in range(size)]

def reset(ctx):
    ctx.x = [random.randint(-50, 50) for _ in range(ctx.size)]

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
        test      = [random.randint(-50, 50) for _ in range(5)]
        correct   = test.copy()
        test_comp = test.copy()

        # Compute reference (sequential)
        correct_oneMinusInverse(correct)

        # Compute PyCOMPSs version
        test_result = oneMinusInverse(test_comp)
        # Compare
        if not all(math.isclose(a, b, abs_tol=1e-6)
                   for a, b in zip(correct, test_result)):
            return False

    return True

# --- DESTROY -----------------------------------------------------

def destroy(ctx):
    del ctx
