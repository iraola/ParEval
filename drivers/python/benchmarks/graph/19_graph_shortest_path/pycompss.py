# Driver for 19_graph_shortest_path    
# """ Find the shortest path in a graph represented by an adjacency matrix.
# """

import random
from python.utilities import fillRand, fequal 

class Context:
    """
    Holds the state for the benchmark
    """
    def __init__(self, size=10):
        try:
            self.size = DRIVER_PROBLEM_SIZE
        except NameError:
            self.size = size

        self.A = []
        self.source = 0
        self.destination = self.size - 1
        self.reset_data()

    def reset_data(self):
        self.A = [[0] * self.size for _ in range(self.size)]
        
        probability = 0.15 

        for i in range(self.size):
            for j in range(i + 1, self.size):
                if random.random() < probability:
                    self.A[i][j] = 1
                    self.A[j][i] = 1

        self.source = random.randint(0, self.size - 1)
        self.destination = random.randint(0, self.size - 1)


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


def compute(ctx: Context):
    # Assuming 'main' is the @task defined elsewhere
    return main(ctx.A, ctx.source, ctx.destination)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.A, ctx.source, ctx.destination)


def validate(ctx: Context):
    """ Verifies parallel execution matches sequential execution. """
    try:
        max_attempts = MAX_VALIDATION_ATTEMPTS
    except NameError:
        max_attempts = 5
        
    for _ in range(max_attempts):
        # Reset the data within the context for a new validation run
        reset(ctx)

        # Parallel execution
        par_res = main(ctx.A, ctx.source, ctx.destination)
        
        # Sequential execution
        seq_res = correct_main(ctx.A, ctx.source, ctx.destination)
        
        # Check equality
        if abs(par_res - seq_res) > 1e-6:
            return False
            
    return True