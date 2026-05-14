def correct_main(A_coo, b):
    N = len(b)
    matrix = [[0.0] * N for _ in range(N)]

    b_copy = list(b)

    for element in A_coo:
        r, c, v = element
        matrix[r][c] = v

    x = [0.0] * N

    for i in range(N):
        max_el = abs(matrix[i][i])
        max_row = i
        for k in range(i + 1, N):
            val = abs(matrix[k][i])
            if val > max_el:
                max_el = val
                max_row = k

        matrix[max_row], matrix[i] = matrix[i], matrix[max_row]
        b_copy[max_row], b_copy[i] = b_copy[i], b_copy[max_row]


        for k in range(i + 1, N):
            if matrix[i][i] == 0: continue 
            
            c = -matrix[k][i] / matrix[i][i]
            
            for j in range(i, N):
                if i == j:
                    matrix[k][j] = 0.0
                else:
                    matrix[k][j] += c * matrix[i][j]
            
            b_copy[k] += c * b_copy[i]

    for i in range(N - 1, -1, -1):
        if matrix[i][i] == 0:
            raise ValueError("Matrix is singular")
            
        x[i] = b_copy[i] / matrix[i][i]
        
        for k in range(i - 1, -1, -1):
            b_copy[k] -= matrix[k][i] * x[i]

    return x