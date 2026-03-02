def correct_main(strings):
    bins = [0] * 26
    for s in strings:
        if s:  # Check if the string is not empty
            first_char = s[0].lower()
            if 'a' <= first_char <= 'z':
                bin_index = ord(first_char) - ord('a')
                bins[bin_index] += 1
    return bins