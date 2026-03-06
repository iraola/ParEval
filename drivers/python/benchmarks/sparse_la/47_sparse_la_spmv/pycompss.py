# Driver for 47_sparse_la_spmv
# """ Sparse z = alpha*A*x + beta*y(SpMV) benchmark.

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
        self.beta = 0.0

        self.A = []
        self.x = []
        self.y = []
        self.reset_data()

    def reset_data(self):
            """
            Generates random inputs for SpMV: z = alpha * A * x + beta * y
            """
            self.alpha = random.uniform(-10.0, 10.0)
            self.beta = random.uniform(-10.0, 10.0)

            self.x = [0.0] * self.size
            fillRand(self.x, -10.0, 10.0)
            
            self.y = [0.0] * (self.size + 2)
            fillRand(self.y, -10.0, 10.0)

            total_elements = (self.size + 2) * self.size
            nnz = int(self.sparsity * total_elements)
            
            flat_indices = random.sample(range(total_elements), nnz)
            
            values = [0.0] * nnz
            fillRand(values, -10.0, 10.0)
            
            self.A = []
            for i, idx in enumerate(flat_indices):
                r = idx // self.size
                c = idx % self.size
                self.A.append({'row': r, 'column': c, 'value': values[i]})
                
            self.A.sort(key=lambda item: (item['row'], item['column']))
        
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
    return main(ctx.alpha, ctx.beta, ctx.A, ctx.x, ctx.y)

def best(ctx: Context):
    return correct_main(ctx.alpha, ctx.beta, ctx.A, ctx.x, ctx.y)

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