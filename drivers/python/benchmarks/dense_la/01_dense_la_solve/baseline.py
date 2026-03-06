def correct_main(A, b):
    N = len(b)
    
    if len(A) == N * N and not isinstance(A[0], list):
        A_matrix = [[A[i * N + j] for j in range(N)] for i in range(N)]
    else:
        A_matrix = [list(row) for row in A]
        
    b_vec = list(b)
    
    for i in range(N):
        max_row = i
        for k in range(i + 1, N):
            if abs(A_matrix[k][i]) > abs(A_matrix[max_row][i]):
                max_row = k
                
        A_matrix[i], A_matrix[max_row] = A_matrix[max_row], A_matrix[i]
        b_vec[i], b_vec[max_row] = b_vec[max_row], b_vec[i]
        
        for k in range(i + 1, N):
            factor = A_matrix[k][i] / A_matrix[i][i]
            for j in range(i, N):
                A_matrix[k][j] -= factor * A_matrix[i][j]
            b_vec[k] -= factor * b_vec[i]
            
    x = [0.0] * N
    for i in range(N - 1, -1, -1):
        sum_ax = 0.0
        for j in range(i + 1, N):
            sum_ax += A_matrix[i][j] * x[j]
        
        x[i] = (b_vec[i] - sum_ax) / A_matrix[i][i]
        
    return x