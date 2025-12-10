# Driver for 35_search_search_for_last_struct_by_key
# """ Find last book with less than 100 pages.
#     Use PyCOMPSs to compute in parallel.
# """


from pycompss.api.task import task
from pycompss.api.parameter import INOUT
from pycompss.api.api import compss_wait_on
import random
from python.utilities import fillRand, fequal, fillRandString

# --- CONTEXT -----------------------------------------------------

class Context:
    def __init__(self, size=5):
        self.size = size
        self.books = [("", 0) for _ in range(size)]
        self.pages = [0] * size
        self.titles = [""] * size

def reset(ctx):
    """Reset context with random data, ensuring at least one book < 100 pages"""
    fillRandString(ctx.titles, 5, 15)
    fillRand(ctx.pages, 101, 1000)
    
    # Ensure at least one book with < 100 pages
    min_idx = 0
    max_idx = ctx.size // 4
    ctx.pages[random.randint(min_idx, max_idx)] = 72
    
    # Populate Book tuples
    for i in range(ctx.size):
        ctx.books[i] = (ctx.titles[i], ctx.pages[i])

def init():
    """Initialize context with default size"""
    ctx = Context(size=5)
    reset(ctx)
    return ctx

# --- COMPUTE -----------------------------------------------------

def compute(ctx):
    """
    Run the generated PyCOMPSs task.
    findLastShortBook returns a PyCOMPSs Future → must wait later.
    """
    findLastShortBook(ctx.books)

def best(ctx):
    """
    Run sequential baseline version
    """
    correct_findLastShortBook(ctx.books)

# --- VALIDATE -----------------------------------------------------

def validate(ctx):
    for _ in range(5):
        # Create test data with random books
        test_titles = [""] * 5
        test_pages = [0] * 5
        fillRandString(test_titles, 5, 15)
        fillRand(test_pages, 101, 1000)
        
        # Ensure at least one book with < 100 pages
        test_pages[random.randint(0, 1)] = random.randint(50, 99)
        
        # Build book tuples
        test = [(test_titles[i], test_pages[i]) for i in range(5)]
        # Compute reference (sequential)
        correct = test.copy()
        correct_result = correct_findLastShortBook(correct)

        # Compute PyCOMPSs version
        test_result = findLastShortBook(test)
        
        # Compare results (both should be tuples or None)
        if correct_result != test_result:
            return False
    
    return True

# --- DESTROY -----------------------------------------------------

def destroy(ctx):
    del ctx
