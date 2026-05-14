# Driver for 43_sort_sort_an_array_of_structs_by_key
# """ Given an array of {"startTime", "duration", "value"} dicts, sort the array by startTime. """

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

        self.x = []
        self.start_times = []
        self.durations = []
        self.values = []
        self.reset_data()

    def reset_data(self):
        self.start_times = [0] * self.size
        fillRand(self.start_times, 0, 25)
    
        self.durations = [0] * self.size
        fillRand(self.durations, 1, 11)

        self.values = [0.0] * self.size
        fillRand(self.values, 0.0, 50.0)

        self.x = [{"startTime": st, "duration": d, "value": v}
                  for st, d, v in zip(self.start_times, self.durations, self.values)]


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

    return main(ctx.x)

def best(ctx: Context):
    """
    Calls the sequential baseline.
    """
    return correct_main(ctx.x)


def validate(ctx: Context):
    """ Verifies parallel execution matches sequential execution. """
    try:
        max_attempts = MAX_VALIDATION_ATTEMPTS
    except NameError:
        max_attempts = 5
        
    for _ in range(max_attempts):
        # Reset the data within the context for a new validation run
        reset(ctx)
        
        # Create copies to avoid modifying the source data in place during comparison
        # if the tasks operate in-place.
        par_res = ctx.x[:]
        seq_res = ctx.x[:]

        # Parallel execution
        par_res = main(par_res)
        
        # Sequential execution
        seq_res = correct_main(seq_res)

        # Check equality
        for p, s in zip(par_res, seq_res):
            if p["startTime"] != s["startTime"] or p["duration"] != s["duration"] or not abs(p["value"] - s["value"]) < 1e-6:
                return False
            
    return True