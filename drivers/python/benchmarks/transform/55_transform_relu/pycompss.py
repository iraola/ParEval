# Driver for 55_transform_relu
# """ Compute the ReLU function on every element of x. Elements less than zero become zero,
#     while elements greater than zero stay the same.
# """

from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.api.api import compss_wait_on

from python.utilities import fillRand, fequal

# --- CONTEXT -----------------------------------------------------

class Context:
    def __init__(self, size=5):
        self.size = size
        self.x = [0.0] * size
        fillRand(self.x, -50.0, 50.0)

def reset(ctx):
    fillRand(ctx.x, -50.0, 50.0)

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

    for _ in range(5):
        test = [0.0] * 5
        fillRand(test, -50.0, 50.0)

        correct   = test.copy()
        test_comp = test.copy()

        # Compute reference (sequential)
        correct_relu(correct)

        # Compute PyCOMPSs version
        test_result = relu(test_comp)

        # Compare
        if not fequal(correct, test_result, eps=1e-6):
            return False

    return True

# --- DESTROY -----------------------------------------------------

def destroy(ctx):
    del ctx
