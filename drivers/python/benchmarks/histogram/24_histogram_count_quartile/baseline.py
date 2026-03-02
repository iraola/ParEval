def correct_main(x):
    bins = [0] * 4
    for value in x:
        fractional_part = value - int(value)
        if fractional_part < 0.25:
            bins[0] += 1
        elif fractional_part < 0.5:
            bins[1] += 1
        elif fractional_part < 0.75:
            bins[2] += 1
        else:
            bins[3] += 1
    return bins