def dfs(A, node, N, visited):
    visited[node] = True    
    count = 1

    for i in range(N):
        if A[node][i] == 1 and not visited[i]:
            count += dfs(A, i, N, visited)
            
    return count

def correct_main(A):
    max_count = 0
    N = len(A)
    visited = [False] * N
    
    for i in range(N):
        if not visited[i]:
            current_component_size = dfs(A, i, N, visited)
            max_count = max(max_count, current_component_size)
            
    return max_count