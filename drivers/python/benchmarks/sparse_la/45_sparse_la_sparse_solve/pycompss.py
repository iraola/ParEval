# Driver for 45_sparse_la_sparse_solve
# """ Solve a sparse linear system.
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

        self.A_coo = []
        self.b = []
        self.N = self.size
        
        self.reset_data()

    def reset_data(self):
        """
        Generates a solvable sparse linear system in COO format with ~15% density.
        Always includes the full diagonal (nonzero) so the dense Gaussian-elimination
        baseline never hits a singular pivot, even at small N.
        """
        total_elements = self.N * self.N
        nnz = int(0.15 * total_elements)

        diagonal_indices = [i * self.N + i for i in range(self.N)]
        off_diagonal_indices = [idx for idx in range(total_elements) if idx % self.N != idx // self.N]
        n_extra = max(nnz - self.N, 0)
        extra_indices = random.sample(off_diagonal_indices, min(n_extra, len(off_diagonal_indices)))

        flat_indices = diagonal_indices + extra_indices

        values = [0.0] * len(flat_indices)
        fillRand(values, -10.0, 10.0)
        
        self.A_coo = []
        for i, idx in enumerate(flat_indices):
            r = idx // self.N
            c = idx % self.N
            self.A_coo.append((r, c, values[i]))

        self.A_coo.sort(key=lambda item: (item[0], item[1]))

        x = [0.0] * self.N
        fillRand(x, -10.0, 10.0)

        self.b = [0.0] * self.N
        for element in self.A_coo:
            self.b[element[0]] += element[2] * x[element[1]]


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
    return main(ctx.A_coo, ctx.b)

def best(ctx: Context):
    # Assuming 'correct_main' is defined elsewhere
    return correct_main(ctx.A_coo, ctx.b)


def validate(ctx: Context):
    """ Verifies parallel execution matches sequential execution. """
    try:
        max_attempts = MAX_VALIDATION_ATTEMPTS
    except NameError:
        max_attempts = 5
        
    for _ in range(max_attempts):
        # Reset the data within the context for a new validation run
        reset(ctx)
        
        import copy
        ctx_par = copy.deepcopy(ctx)

        # Parallel execution
        par_res = compute(ctx_par)
        
        # Sequential execution
        seq_res = best(ctx)
        
        # Check equality
        if not fequal(par_res, seq_res, eps=1e-6):
            return False
            
    return True