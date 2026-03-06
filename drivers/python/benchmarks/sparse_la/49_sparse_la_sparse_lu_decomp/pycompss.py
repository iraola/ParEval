# Driver for 49_sparse_la_sparse_lu_decomp
# """ Sparse LU decomposition benchmark.

import copy
from python.utilities import fillRand, fequal 
import random

# --- CONTEXT CLASS -----------------------------------------------

class Context:

    def __init__(self, size=10, sparsity=0.2):
        try:
            self.size = DRIVER_PROBLEM_SIZE
        except NameError:
            self.size = size
            
        try:
            self.sparsity = SPARSE_LA_SPARSITY
        except NameError:
            self.sparsity = sparsity

        self.A = []
        self.N = self.size
        self.reset_data()

    def reset_data(self):
        self.A = []
        for i in range(self.N):
            row_sum = 0.0
            row_entries = []
            
            for j in range(self.N):
                if i != j:
                    if random.random() < self.sparsity:
                        val = random.uniform(-10.0, 10.0)
                        row_entries.append({'row': i, 'column': j, 'value': val})
                        row_sum += abs(val)
            
            diag_val = row_sum + random.uniform(1.0, 5.0)
            
            if random.random() < 0.5:
                diag_val = -diag_val
                
            row_entries.append({'row': i, 'column': i, 'value': diag_val})
            
            self.A.extend(row_entries)

        
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
    return main(ctx.A, ctx.N)

def best(ctx: Context):
    return correct_main(ctx.A, ctx.N)

# --- VALIDATE ----------------------------------------------------

def compare_sparse_matrices(mat_par, mat_seq, eps=1e-6):
    par_dict = {(elem['row'], elem['column']): elem['value'] for elem in mat_par}
    seq_dict = {(elem['row'], elem['column']): elem['value'] for elem in mat_seq}
    
    all_coords = set(par_dict.keys()).union(set(seq_dict.keys()))
    
    for coord in all_coords:
        val_par = par_dict.get(coord, 0.0)
        val_seq = seq_dict.get(coord, 0.0)
        
        if abs(val_par - val_seq) > eps:
            return False
            
    return True

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
        par_res_L, par_res_U = compute(ctx_par)
        seq_res_L, seq_res_U = best(ctx)

        if not compare_sparse_matrices(par_res_L, seq_res_L) or not compare_sparse_matrices(par_res_U, seq_res_U):
            return False
            
    return True