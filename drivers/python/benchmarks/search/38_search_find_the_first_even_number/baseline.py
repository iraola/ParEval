def correct_main(x):
    for index, value in enumerate(x):
        if value % 2 == 0:
            return index
    return -1