import itertools

def correct_main(points):
    min_area = float('inf')
    for p1, p2, p3 in itertools.combinations(points, 3):
        area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                         p2[0] * (p3[1] - p1[1]) + 
                         p3[0] * (p1[1] - p2[1]))
        if area < min_area:
            min_area = area

    return min_area