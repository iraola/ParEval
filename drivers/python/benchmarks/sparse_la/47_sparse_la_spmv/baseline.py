def correct_main(alpha, beta, A, x, y):
    result_y = [beta * val for val in y]
    
    for element in A:
        row, col, val = element
        result_y[row] += alpha * val * x[col]
        
    return result_y