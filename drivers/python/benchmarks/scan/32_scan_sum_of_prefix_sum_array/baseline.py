def correct_main(x):
    prefix_sum = []
    current_sum = 0
    for num in x:
        current_sum += num
        prefix_sum.append(current_sum)
    return sum(prefix_sum)