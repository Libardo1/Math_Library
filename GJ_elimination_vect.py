import numpy as np
from PLUDecomposition import PLU_Factorization
from forward_sub import Forward_Sub
from back_sub import Back_Sub

def GJ_elimination(A_augmented):
    
    A_augmented = np.asarray(A_augmented)
    A = A_augmented[:,:-1]
         
    L, U, P = PLU_Factorization(A) # LU Factorization
    
    b = A_augmented[:,-1]
    P_b = P.dot(b)
    L_P_b = Forward_Sub(L, P_b)
    x = Back_Sub(U, L_P_b)

    print "x:\n", x
    return x

if __name__ == '__main__':
    A = [[1,1,1,5],
        [2,3,5,8],
        [4,0,5,2]]
    GJ_elimination(A)