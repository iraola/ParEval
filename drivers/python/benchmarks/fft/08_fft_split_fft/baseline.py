import cmath

def correct_main(x):
    N = len(x)
    r = []
    i = []
    
    for k in range(N):
        X_k = 0.0j
        for n in range(N):
            # Standard DFT calculation
            angle = -2j * cmath.pi * k * n / N
            X_k += x[n] * cmath.exp(angle)
            
        # Separate the real and imaginary components into their respective lists
        r.append(X_k.real)
        i.append(X_k.imag)
        
    # Return both lists as a tuple
    return r, i