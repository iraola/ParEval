def correct_main(x):
    output = 1
    for i in range(len(x)):
        if i % 2 == 0:
            output *= x[i]
        else:
            output *= 1 / x[i]
    return output