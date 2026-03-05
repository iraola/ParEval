import cmath

def correct_main(x):
    N = len(x)
    output = []
    
    for k in range(N):
        X_k = 0.0j
        for n in range(N):
            # Standard DFT formula matching your PyCOMPSs implementation
            angle = -2j * cmath.pi * k * n / N
            X_k += x[n] * cmath.exp(angle)
            
        output.append(X_k)
        
    return output