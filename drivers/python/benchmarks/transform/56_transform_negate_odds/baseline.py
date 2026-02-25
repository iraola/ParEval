def correct_main(x):
    for i in range(len(x)):
        if x[i] % 2 != 0:
            x[i] = -x[i]
        else:
            x[i] = x[i]//2

    return x