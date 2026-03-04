import itertools

def correct_main(points):
    min_distance = float('inf')
    for p1, p2 in itertools.combinations(points, 2):
        distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
    return min_distance