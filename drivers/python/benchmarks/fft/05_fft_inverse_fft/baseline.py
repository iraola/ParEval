import cmath

def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def _dft_direct(x):
    # Direct O(n^2) DFT, used as a fallback for sizes the Cooley-Tukey
    # butterfly below can't handle (it requires n to be a power of two).
    n = len(x)
    output = [0.0j] * n
    for k in range(n):
        total = 0.0j
        for j in range(n):
            angle = -2j * cmath.pi * k * j / n
            total += x[j] * cmath.exp(angle)
        output[k] = total
    return output

def fft_iterative(x):
    n = len(x)
    if not _is_power_of_two(n):
        return _dft_direct(x)

    # Bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]

    # Cooley-Tukey butterfly operations
    length = 2
    while length <= n:
        angle = -2.0 * cmath.pi / length
        w_len = cmath.exp(complex(0, angle))
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for j in range(length // 2):
                u = x[i + j]
                v = x[i + j + length // 2] * w
                x[i + j] = u + v
                x[i + j + length // 2] = u - v
                w *= w_len
        length <<= 1
    return x

def correct_main(x):
    N = len(x)
    x = [val.conjugate() for val in x]
    x = fft_iterative(x)
    return [val.conjugate() / N for val in x]