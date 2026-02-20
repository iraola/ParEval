# Driver for 46_sparse_la_spmm
# """ Sparse matrix-matrix multiplication (SpMM) benchmark.

import copy
from python.utilities import fillRand, fequal 

# --- CONTEXT CLASS -----------------------------------------------

class Context:
    """
    Holds the state for the SpMM benchmark.
    Generates sparse matrices A (MxK) and X (KxN) in COO format.
    """
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

        # Calculate dimensions based on the C++ logic
        self.M = self.size
        self.K = self.size // 4
        self.N = self.size // 2

        self.A = []
        self.X = []
        
        self.reset_data()

    def reset_data(self):
        """ Populates A and X with random sparse coordinates and sorts them. """
        nVals_A = int(self.M * self.K * self.sparsity)
        nVals_X = int(self.K * self.N * self.sparsity)

        # 1. Pre-allocate and fill arrays for A
        a_rows = [0] * nVals_A
        a_cols = [0] * nVals_A
        a_vals = [0.0] * nVals_A
        
        # fillRand uses randint(min, max-1) for ints, so we pass self.M and self.K
        fillRand(a_rows, 0, self.M) 
        fillRand(a_cols, 0, self.K)
        fillRand(a_vals, -1.0, 1.0)

        # Pack into dictionaries
        self.A = [{'row': r, 'column': c, 'value': v} for r, c, v in zip(a_rows, a_cols, a_vals)]

        # 2. Pre-allocate and fill arrays for X
        x_rows = [0] * nVals_X
        x_cols = [0] * nVals_X
        x_vals = [0.0] * nVals_X
        
        fillRand(x_rows, 0, self.K)
        fillRand(x_cols, 0, self.N)
        fillRand(x_vals, -1.0, 1.0)

        # Pack into dictionaries
        self.X = [{'row': r, 'column': c, 'value': v} for r, c, v in zip(x_rows, x_cols, x_vals)]

        # 3. Sort elements: primarily by row, secondarily by column
        self.A.sort(key=lambda item: (item['row'], item['column']))
        self.X.sort(key=lambda item: (item['row'], item['column']))

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
    """
    Launches the parallel PyCOMPSs SpMM tasks.
    Assumes 'main' is your PyCOMPSs function.
    """
    return main(ctx.A, ctx.X, ctx.M, ctx.N)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    Assumes 'correct_main' is the pure Python baseline.
    """
    return correct_main(ctx.A, ctx.X, ctx.M, ctx.N)

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

        # Validate using the custom fequal, passing it row by row
        for r in range(ctx.M):
            if not fequal(par_res[r], seq_res[r], eps=1e-6):
                print(f"Mismatch found at row {r}.")
                return False
            
    return True