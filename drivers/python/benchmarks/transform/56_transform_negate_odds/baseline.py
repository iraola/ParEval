def correct_negateOddsAndHalveEvens(x):
    for i in range(len(x)):
        x[i] = -x[i] if x[i] % 2 != 0 else x[i] // 2