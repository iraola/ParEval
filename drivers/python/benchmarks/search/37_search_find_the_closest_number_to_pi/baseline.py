def correct_findClosestToPi(x):
    closest_index = -1
    closest_diff = float('inf')
    pi = 3.141592653589793
    for i, value in enumerate(x):
        diff = abs(value - pi)
        if diff < closest_diff:
            closest_diff = diff
            closest_index = i
    return closest_index