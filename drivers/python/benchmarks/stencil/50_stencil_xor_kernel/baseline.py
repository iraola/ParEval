def correct_main(matrix, N):
    # Initialize an NxN output matrix with zeros
    output = [[0] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            neighbor_count = 0
            
            # Check all 8 potential neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    # Skip the cell itself
                    if di == 0 and dj == 0:
                        continue
                    
                    ni, nj = i + di, j + dj
                    
                    # Ensure neighbor is within grid boundaries
                    if 0 <= ni < N and 0 <= nj < N:
                        if matrix[ni][nj] == 1:
                            neighbor_count += 1
            
            # Set output to 1 if exactly one neighbor is 1
            if neighbor_count == 1:
                output[i][j] = 1
            else:
                output[i][j] = 0
                
    return output