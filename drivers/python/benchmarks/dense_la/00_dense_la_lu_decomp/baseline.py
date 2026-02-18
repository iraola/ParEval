def correct_main(A, N):
    """
    Factorize the matrix A into A=LU where L is lower triangular 
    and U is upper triangular. Results are stored in-place in A.
    
    A is a list of lists (Array of Arrays).
    Example: [[4.0, 3.0], [6.0, 3.0]]
    """    
    for k in range(N):
        # iterate over rows below the pivot
        for i in range(k + 1, N):
            
            # Calculate the factor (L part)
            factor = A[i][k] / A[k][k]
            
            # Store the factor in the lower triangle
            A[i][k] = factor
            
            # Update the remaining elements in row i (U part)
            for j in range(k + 1, N):
                A[i][j] -= factor * A[k][j]
    
    return A