def correct_main(x, y):
    output = 0
    for i in range(len(x)):
        output += min(x[i], y[i])
    return output