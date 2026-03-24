def correct_main(x):
    # Extract and sort the non-zero elements
    non_zeros = sorted(v for v in x if v != 0)
    # Turn that sorted list into an iterator to consume in-order below
    it = iter(non_zeros)
    return [next(it) if v != 0 else 0 for v in x]