def correct_main(x, y, val):
    found_in_x = False
    found_in_y = False
    
    for value in x:
        if value == val:
            found_in_x = True
            break
            
    for value in y:
        if value == val:
            found_in_y = True
            break
    
    return found_in_x != found_in_y