import random
import string
from typing import List, Any


def fillRandString(vec: List[str], min_len: int, max_len: int) -> None:
    """
    Fill a list with random lowercase strings with lengths in [min_len, max_len).
    The list is modified in place.
    """
    for i in range(len(vec)):
        length = random.randint(min_len, max_len - 1)
        vec[i] = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def fillRand(vec: List[Any], minv, maxv) -> None:
    """
    Fill a list with random values between minv and maxv.
    - Floats → uniform distribution
    - Ints → integer range
    - Complex numbers → random real/imag components
    """
    for i in range(len(vec)):
        if isinstance(minv, float) or isinstance(maxv, float):
            vec[i] = random.uniform(minv, maxv)

        elif isinstance(minv, int) and isinstance(maxv, int):
            vec[i] = random.randint(minv, maxv - 1)

        elif isinstance(minv, complex) and isinstance(maxv, complex):
            real = random.uniform(minv.real, maxv.real)
            imag = random.uniform(minv.imag, maxv.imag)
            vec[i] = complex(real, imag)

        else:
            raise TypeError("Unsupported type for fillRand")


def fequal(a: List[float], b: List[float], eps: float = 1e-6) -> bool:
    """
    Compare two lists of floats with tolerance.
    Returns True if |a[i] - b[i]| <= eps for all i.
    """
    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if abs(x - y) > eps:
            return False

    return True
