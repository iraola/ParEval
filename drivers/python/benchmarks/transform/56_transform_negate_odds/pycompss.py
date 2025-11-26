# Driver for 56_transform_negate_odds
# """ In the vector x negate the odd values and divide the even values by 2.
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
    negateOddsAndHalveEvens returns a PyCOMPSs Future → must wait later.
    """
    ctx.x = negateOddsAndHalveEvens(ctx.x)

def best(ctx):
    """
    Run sequential baseline version
    """
    correct_negateOddsAndHalveEvens(ctx.x)

# --- VALIDATE -----------------------------------------------------

def validate(ctx):

    for _ in range(5):
        test      = [0] * 5
        fillRand(test, -50, 50)

        correct   = test.copy()
        test_comp = test.copy()

        # Compute reference (sequential)
        correct_negateOddsAndHalveEvens(correct)

        # Compute PyCOMPSs version
        test_result = negateOddsAndHalveEvens(test_comp)
        
        # Compare
        if not fequal(correct, test_result, eps=1e-6):
            return False

    return True

# --- DESTROY -----------------------------------------------------

def destroy(ctx):
    del ctx
