def correct_isPowerOfTwo(x):
    for i in range(len(x)):
        x[i] = x[i] > 0 and (x[i] & (x[i] - 1)) == 0