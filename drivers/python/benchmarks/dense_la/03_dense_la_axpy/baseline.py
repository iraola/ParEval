def correct_main(alpha, x, y):
    z = [0 for _ in range(len(x))]
    for i in range(len(x)):
        z[i] = alpha * x[i] + y[i]
                
    return z