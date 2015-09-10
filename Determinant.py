import numpy as np
from PLUDecomposition import PLU_Factorization

def Determinant(M):
    '''
    input: square matrix
    Return: determinant
    '''
    
    L, U, P = PLU_Factorization(M)
    n = len(L)

    # diagonal elements of matrix L
    dia_elements = (L*np.eye(n)).dot(np.ones(n))
    
    # count the number of permutation made to M
    P_count = np.sum(np.argmax(P, axis=0) != range(3))
    
    # determinant of P (permutation matrix)
    if P_count%2 == 0:
        P_sign = -1
    else:
        P_sign = 1
        
    # the determiant is simply the product of the diagonal elements of matrix L times the determiant of P
    return reduce(lambda a, b: a*b, dia_elements) * P_sign

if __name__ == "__main__":
    A = [[0,3,-1],
         [3,5,10],
         [0,6,3]]
    #A = [[1,2],[3,4]]
    print "Matrix:\n", np.asarray(A)
    print "\nDeterminant is:\n", Determinant(A)
    print "\nnumpy check:\n", np.linalg.det(A)