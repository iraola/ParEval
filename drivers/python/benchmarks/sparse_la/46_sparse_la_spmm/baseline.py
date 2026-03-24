def correct_main(A, X, M, N):
    """
    Compute the matrix multiplication Y = AX. 
    A and X are sparse matrices in COO format (lists of dictionaries).
    Y is returned as a dense 2D list (M rows by N columns).
    """
    
    Y = [[0.0 for _ in range(N)] for _ in range(M)]

    for a in A:
        for x in X:
            if a['column'] == x['row']:
                Y[a['row']][x['column']] += a['value'] * x['value']
                
    return Y