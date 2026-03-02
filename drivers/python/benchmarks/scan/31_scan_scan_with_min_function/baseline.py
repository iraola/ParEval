def correct_main(x):
    output = []
    current_min = float('inf')
    for num in x:
        current_min = min(current_min, num)
        output.append(current_min)
    return output