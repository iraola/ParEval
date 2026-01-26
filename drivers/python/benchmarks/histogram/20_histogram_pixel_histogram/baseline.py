def correct_main(image):
    bins = [0] * 256
    for pixel in image:
        bins[pixel] += 1
    return bins