# Driver for 55_transform_relu
# """ Compute the ReLU function on every element of x. Elements less than zero become zero,
#     while elements greater than zero stay the same.
# """

import random
from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.api.api import compss_wait_on

# --- CONTEXT -----------------------------------------------------

class Context:
    def __init__(self, size=5):
        self.size = size
        self.x = [random.uniform(-50, 50) for _ in range(size)]

def reset(ctx):
    ctx.x = [random.uniform(-50, 50) for _ in range(ctx.size)]

def init():
    return Context(size=5)

# --- COMPUTE -----------------------------------------------------

def compute(ctx):
    """
    Run the generated PyCOMPSs task.
    relu_task returns a PyCOMPSs Future → must wait later.
    """
    ctx.x = relu(ctx.x)

def best(ctx):
    """
    Run sequential baseline version
    """
    correct_relu(ctx.x)

# --- VALIDATE -----------------------------------------------------

def validate(ctx):
    import math

    for _ in range(5):
        test      = [random.uniform(-50, 50) for _ in range(5)]
        correct   = test.copy()
        test_comp = test.copy()

        # Compute reference (sequential)
        correct_relu(correct)

        # Compute PyCOMPSs version
        test_result = relu(test_comp)

        # Compare
        if not all(math.isclose(a, b, abs_tol=1e-6)
                   for a, b in zip(correct, test_result)):
            return False

    return True

# --- DESTROY -----------------------------------------------------

def destroy(ctx):
    del ctx
