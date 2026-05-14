def correct_main(alpha, x, y, N):
    z = [0.0] * N
    
    for elem in y:
        index, value = elem
        z[index] += value

    for elem in x:
        index, value = elem
        z[index] += alpha * value
        
    return z