import numpy as np

def GJ_elimination(A):
    
    A = np.asarray(A)
    n = A.shape[0]
    m = A.shape[1]
    x = []
    assert m == n+1
    
#     for i, row in enumerate(A):
#         for j, value in enumerate(row):
#             print " A[{}][{}]:".format(i,j)
#             print "Value: ", value
            
    for j, col in enumerate(A[:,:-1].T):
        for i, value in enumerate(col):
            if i != j:
                c = A[i][j] / float(A[j][j])
                for k in range(m):
                    A[i][k] -= c * A[j][k]

    A[n-1] = A[n-1] / A[n-1][n-1]
    
    print "A:\n", A
    print "x:\n", A[:,m-1]
    return None

if __name__ == '__main__':
    A = [[1,1,1,5],
        [2,3,5,8],
        [4,0,5,2]]
    GJ_elimination(A)
    