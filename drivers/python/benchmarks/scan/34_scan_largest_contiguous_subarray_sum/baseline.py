def correct_main(x):
    current_sum = max_sum = x[0]
    
    for num in x[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum