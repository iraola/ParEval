def correct_main(A):
    edges = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] == 1:
                edges += 1
    return edges