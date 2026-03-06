def correct_main(alpha, x, y, N):
    z = [0.0] * N
    
    for elem in y:
        index = elem['index']
        value = elem['value']
        z[index] += value
        
    for elem in x:
        index = elem['index']
        value = elem['value']
        z[index] += alpha * value
        
    return z