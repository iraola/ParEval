def correct_main(A, X, M, N):
    """
    Compute the matrix multiplication Y = AX.
    A and X are sparse matrices in COO format (lists of (row, col, value) tuples).
    Y is returned as a dense 2D list (M rows by N columns).
    """

    Y = [[0.0 for _ in range(N)] for _ in range(M)]

    for a in A:
        for x in X:
            if a[1] == x[0]:
                Y[a[0]][x[1]] += a[2] * x[2]

    return Y