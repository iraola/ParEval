def correct_main(image):
    bins = [0] * 10
    for value in image:
        bin_index = min(value // 10, 9)
        bins[bin_index] += 1
    return bins