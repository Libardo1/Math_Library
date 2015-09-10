import numpy as np

def Forward_Sub(A, b):
    
    x = [b[0]/float(A[0][0])]
    
    for i in range(1,len(A)):
         x.append((b[i] - A[i,:i].dot(x[:i])) / A[i][i])
    return x