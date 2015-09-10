import numpy as np

def Back_Sub(A, b):
    
    n = len(A) - 1
    x = np.zeros(3)
    x[n] = b[n]/float(A[n][n])
    
    for i in range(n-1,-1,-1):
        x[i] = (b[i] - A[i,i:n+1].dot(x[i:n+1])) / A[i][i]
    return x