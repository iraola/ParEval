def correct_main(books):
    index = -1
    for i in range(len(books)):
        if books[i][1] < 100 and i > index:
            index = i
    return index