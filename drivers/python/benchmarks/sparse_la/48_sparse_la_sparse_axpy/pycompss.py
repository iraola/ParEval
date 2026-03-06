# Driver for 48_sparse_la_sparse_axpy
# """ Sparse z = alpha*x + y benchmark.

import copy
from python.utilities import fillRand, fequal 
import random

# --- CONTEXT CLASS -----------------------------------------------

class Context:

    def __init__(self, size=10, sparsity=0.2):
        # Safely fetch global benchmark variables or use defaults
        try:
            self.size = DRIVER_PROBLEM_SIZE
        except NameError:
            self.size = size
            
        try:
            self.sparsity = SPARSE_LA_SPARSITY
        except NameError:
            self.sparsity = sparsity

        self.alpha = 0.0
        self.x = []
        self.y = []
        self.N = self.size
        self.reset_data()

    def reset_data(self):
        self.alpha = random.uniform(-10.0, 10.0)
        
        # Initialize x and y with zeros
        self.x = [] * self.N
        self.y = [] * self.N
        
        # Fill x and y with random values based on sparsity
        for i in range(self.N):
            if random.random() < self.sparsity:
                self.x.append({'index': i, 'value': random.uniform(-10.0, 10.0)})
            if random.random() < self.sparsity:
                self.y.append({'index': i, 'value': random.uniform(-10.0, 10.0)})

        
# --- DRIVER INTERFACE --------------------------------------------

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

# --- COMPUTE -----------------------------------------------------

def compute(ctx: Context):
    return main(ctx.alpha, ctx.x, ctx.y, ctx.N)

def best(ctx: Context):
    return correct_main(ctx.alpha, ctx.x, ctx.y, ctx.N)

# --- VALIDATE ----------------------------------------------------

def validate(ctx: Context):
    """ Verifies parallel execution matches sequential execution. """
    try:
        max_attempts = MAX_VALIDATION_ATTEMPTS
    except NameError:
        max_attempts = 5
        
    for _ in range(max_attempts):
        # Reset the data within the context for a new validation run
        reset(ctx)
        
        # Deepcopy to ensure the parallel run doesn't mutate baseline data
        ctx_par = copy.deepcopy(ctx)

        # Execute both versions
        par_res = compute(ctx_par)
        seq_res = best(ctx)

        if not fequal(par_res, seq_res, eps=1e-6):
            return False
            
    return True