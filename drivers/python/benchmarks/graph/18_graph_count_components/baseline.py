def correct_main(A):
    if not A:
        return 0
        
    N = len(A)
    visited = [False] * N
    components_count = 0
    
    def dfs(node):
        visited[node] = True
        for neighbor in range(N):
            if A[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor)
                
    for i in range(N):
        if not visited[i]:
            components_count += 1
            dfs(i)
            
    return components_count