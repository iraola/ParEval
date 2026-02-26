def correct_main(x):
    # 1. Extract and sort the non-zero elements
    non_zeros = sorted(v for v in x if v != 0)
    
    # 2. Turn that sorted list into an iterator
    it = iter(non_zeros)
    
    # 3. Build a new list: pull from 'it' if the original wasn't 0, else keep 0
    return [next(it) if v != 0 else 0 for v in x]