def correct_oneMinusInverse(x):
    for i in range(len(x)):
        x[i] = 1 - 1 / x[i] if x[i] != 0 else 0