def correct_main(x):
    output = []
    current_sum = 0
    for num in reversed(x):
        current_sum += num
        output.append(current_sum)
    return output