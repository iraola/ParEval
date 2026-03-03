from collections import deque

def correct_main(A, source, dest):
    if source == dest:
        return 0
        
    N = len(A)
    
    queue = deque([(source, 0)])
    
    visited = {source}
    
    while queue:
        current_node, dist = queue.popleft()
        
        for neighbor in range(N):
            if A[current_node][neighbor] == 1:
                if neighbor == dest:
                    return dist + 1
                    
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    
    return -1