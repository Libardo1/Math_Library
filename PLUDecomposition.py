import numpy as np
from permutation_matrix import permutation_matrix

def PLU_Factorization(M):
    '''
    Input: square matrix
    return: PLU Factorization
    '''
    
    M = np.asarray(M)
    assert M.shape[0]  == M.shape[1] # Assert matrix is square
    
    P = permutation_matrix(M)
    
    # pivot matrix
    M = P.dot(M)
    #print M
    
    # Dimension of square matrix
    n = len(M)
    
    # Initialize L and U matrix
    L = np.zeros((n,n))
    U = np.eye(3)
    
    # 1st column of L
    L[:,0] = M[:,0]
    
    # 1st row of U
    U[0,1:] = M[0,1:] / L[0][0]
    
    # loop over rows
    for row_i in range(1,n):
        
        # loop over row_i+1 columns
        for col_j in range(1,row_i+1):
            L_e = L[row_i,:col_j].dot(U[:col_j,col_j]) # Elimination term
            L[row_i,col_j] = M[row_i,col_j] - L_e
        
        # loop over columns from row_i+1 to n
        for col_j in range(row_i+1,n):
            U_e = L[row_i,:row_i].dot(U[:row_i,col_j]) # Elimination term
            U[row_i,col_j] = (M[row_i,col_j] - U_e)/ L[row_i,row_i]
            
    return L, U, P

if __name__ == "__main__":
    A = [[5,3,-1],
         [3,-2,4],
         [1,-5,3]]
    L,U, P = PLU_Factorization(A)
    print "\nA:\n", np.asarray(A)
    print "\nL:\n", L
    print "\nU:\n", U
    print "\nP:\n", P
    print "\nPLU:\n", P.dot(L.dot(U)) # we get back to the original Matrixb