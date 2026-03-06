def correct_main(A, x):
    z = [0 for _ in range(len(A))]
    for row in range(len(A)):
        for i in range(len(x)):
            z[row] += A[row][i] * x[i]                
    return z