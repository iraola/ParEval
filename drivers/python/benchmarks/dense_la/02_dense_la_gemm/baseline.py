def correct_main(A, B):

    M = len(A)
    K = len(A[0])
    N = len(B[0])
    
    C = [[0 for _ in range(N)] for _ in range(M)]
    
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j] += A[i][k] * B[k][j]
                
    return C