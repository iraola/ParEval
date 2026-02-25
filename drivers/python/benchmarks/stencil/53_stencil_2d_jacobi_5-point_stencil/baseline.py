def correct_main(input):
    output = [[0] * len(input) for _ in range(len(input))]
    for i in range(len(input)):
        for j in range(len(input[i])):
            left = input[i - 1][j] if i - 1 >= 0 else 0
            center = input[i][j]
            right = input[i + 1][j] if i + 1 < len(input) else 0
            up = input[i][j - 1] if j - 1 >= 0 else 0
            down = input[i][j + 1] if j + 1 < len(input[i]) else 0
            output[i][j] = (left + center + right + up + down) / 5
            
    return output