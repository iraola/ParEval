def correct_main(x):
    output = 0
    for i in range(len(x)):
        output += x[i]
    output /= len(x)
    return output