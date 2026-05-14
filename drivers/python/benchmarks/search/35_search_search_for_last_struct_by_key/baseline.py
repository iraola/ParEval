def correct_main(books):
    index = -1
    for i in range(len(books)):
        if books[i]["pages"] < 100 and i > index:
            index = i
    return index