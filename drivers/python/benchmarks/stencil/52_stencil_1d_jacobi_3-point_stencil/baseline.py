def correct_main(input):
    output = [0] * len(input)
    for i in range(len(input)):
        left = input[i - 1] if i - 1 >= 0 else 0
        center = input[i]
        right = input[i + 1] if i + 1 < len(input) else 0
        output[i] = (left + center + right) / 3
            
    return output