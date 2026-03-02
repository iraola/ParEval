def correct_main(image):
    bins = [0] * 4
    for x, y in image:
        if x >= 0 and y >= 0:
            bins[0] += 1  # Quadrant I
        elif x < 0 and y >= 0:
            bins[1] += 1  # Quadrant II
        elif x < 0 and y < 0:
            bins[2] += 1  # Quadrant III
        elif x >= 0 and y < 0:
            bins[3] += 1  # Quadrant IV
    return bins