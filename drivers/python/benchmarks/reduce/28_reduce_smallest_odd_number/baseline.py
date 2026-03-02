def correct_main(x):
    min = float('inf')
    for num in x:
        if num % 2 == 1 and num < min:
            min = num
    return min if min != float('inf') else -1