# Driver for 35_search_search_for_last_struct_by_key
# """ Find last book with less than 100 pages.
#     Use PyCOMPSs to compute in parallel.
# """
import random
from python.utilities import fillRand, fequal, fillRandString


class Context:
    """
    Holds the state (list of books) for the benchmark.
    """
    def __init__(self, size=10):
        try:
            self.size = DRIVER_PROBLEM_SIZE
        except NameError:
            self.size = size

        self.books = []
        # Initialize data immediately
        self.reset_data()

    def reset_data(self):
        """
        Populates the list of books. 
        Each book is a tuple: (title: str, pages: int).
        """
        titles = [""] * self.size
        pages = [0] * self.size
        
        fillRandString(titles, 5, 15)
        fillRand(pages, 101, 1000)
        
        # Guarantee at least one "short" book (< 100 pages)
        # to ensure the search logic is exercised.
        pages[random.randint(0, self.size - 1)] = random.randint(10, 99)
        
        # Store as a list of tuples
        self.books = list(zip(titles, pages))


def init():
    """ 
    Initializes context as a Class Instance.
    """
    return Context()

def reset(ctx: Context):
    """
    Wrapper to call the class method. 
    """
    ctx.reset_data()


def compute(ctx: Context):
    """ Calls the PyCOMPSs function/s using data from the context class. """
    return main(ctx.books)

def best(ctx: Context):
    """ Calls the sequential baseline using data from the context class. """
    return correct_main(ctx.books)

def validate(ctx: Context):
    """ Verifies that parallel output matches sequential output. """
    try:
        max_attempts = MAX_VALIDATION_ATTEMPTS
    except NameError:
        max_attempts = 5
    for _ in range(max_attempts):
        reset(ctx)

        test_res = compute(ctx)
        seq_res = best(ctx)
        
        if test_res != seq_res:
            print(f"Validation Failed: Parallel({test_res}) != Sequential({seq_res})")
            return False
    return True