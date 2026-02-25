def correct_main(image_in, n):
    
    # Initialize output image as an NxN 2D list filled with 0
    image_out = [[0 for _ in range(n)] for _ in range(n)]
    
    # Define the 3x3 edge kernel
    kernel = [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]
    
    for r in range(n):
        for c in range(n):
            pixel_sum = 0
            
            # Iterate through the 3x3 kernel neighbors
            for i in range(3):
                for j in range(3):
                    # Calculate neighbor coordinates (relative to center r, c)
                    # i-1 and j-1 shift the range [0, 1, 2] to [-1, 0, 1]
                    nr, nc = r + (i - 1), c + (j - 1)
                    
                    # Boundary Check: If within bounds, add weighted pixel value
                    # If out of bounds, we implicitly treat it as 0
                    if 0 <= nr < n and 0 <= nc < n:
                        pixel_sum += image_in[nr][nc] * kernel[i][j]
            
            # Clip the output to [0, 255]
            image_out[r][c] = max(0, min(255, pixel_sum))
            
    return image_out