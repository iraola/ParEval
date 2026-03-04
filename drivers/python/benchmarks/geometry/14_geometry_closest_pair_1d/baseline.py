import itertools

def correct_main(points):
    min_distance = float('inf')
    for p1, p2 in itertools.combinations(points, 2):
        distance = abs(p1 - p2)
        if distance < min_distance:
            min_distance = distance
    return min_distance