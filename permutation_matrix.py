import numpy as np

def permutation_matrix(M):
    '''Compute the permutation matrix to realign max numbers on the diagonal'''
    
    n = len(M)
    P = np.eye(3)
    
    for col_j in range(n):
        # row index of max values by column
        row_i = max(range(col_j,n), key = lambda row_i: abs(M[row_i, col_j]))
        
        # swap rows
        if row_i != col_j:
            P[[row_i,col_j], :] = P[[col_j,row_i], :]
    return P