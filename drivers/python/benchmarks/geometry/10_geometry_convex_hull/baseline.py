def correct_main(points):
    """
    Computes the convex hull of a list of (x, y) tuples.
    """
    # Sort points lexicographically (by x, then by y)
    n = len(points)
    if n <= 2:
        return points
    
    points.sort()

    # cross_product > 0: Left turn (Counter-clockwise)
    # cross_product = 0: Collinear
    # cross_product < 0: Right turn (Clockwise)
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate lower and upper hull, removing the last point of each because it's repeated
    return lower[:-1] + upper[:-1]