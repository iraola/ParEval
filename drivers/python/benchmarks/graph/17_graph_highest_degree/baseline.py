def correct_main(A):
    max_count = 0
    for row in A:
        count = sum(row)
        max_count = max(max_count, count)
            
    return max_count