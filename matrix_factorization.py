
import numpy as np
import random as rd
from scipy.optimize import fmin_cg

def matrix_factorization(R, P, Q, K, steps = 5000, alpha = 0.0002, beta = 0.02):

	"""
	INPUT:
	    R     : a matrix to be factorized, dimension N x M
	    P     : an initial matrix of dimension N x K
	    Q     : an initial matrix of dimension M x K
	    K     : the number of latent features
	    steps : the maximum number of steps to perform the optimisation
	    alpha : the learning rate
	    beta  : the regularization parameter
	OUTPUT:
	    the final matrices P and Q
	"""
	Q = Q.T
	for step in xrange(steps):
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					eij = R[i][j] - P[i,:].dot(Q[:,j])
					for k in xrange(K):
						P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j]) 
		eR = P.dot(Q)
		e = 0
		for i in xrange(len(R)):
			for j in xrange(len(R[i])):
				if R[i][j] > 0:
					e += (R[i][j] - P[i,:].dot(Q[:,j]))**2
					for k in xrange(K):
						e += (beta/2.0) * (P[i][k]**2 + Q[k][j]**2)					
		if e < 0.001:
			break
	return P, Q.T 


def matrix_factorization_vectorized(R, P, Q, K, steps = 5000, alpha = 0.0002, beta = 0.02):
	
	I = R.copy()
	I[I != 0] = 1

	params = np.concatenate((P.flatten(),Q.flatten()))
	args = (R, I, beta, P.shape, Q.shape)

	def cost_function(params, *args):
		R, I, beta, Pdim, Qdim = args

		n = Pdim[0] * Pdim[1]
		P = params[:n].reshape(Pdim)
		Q = params[n:].reshape(Qdim)

		Err = I * (P.dot(Q.T) - R)
		sqErr = Err**2

		reg = (beta/2.0) * (np.linalg.norm(P, 'fro')**2 + np.linalg.norm(Q, 'fro')**2)

		cost = sqErr.sum() + reg
		return cost

	def gradient_function(params, *args):
		R, I, beta, Pdim, Qdim = args

		n = Pdim[0] * Pdim[1]
		P = params[:n].reshape(Pdim)
		Q = params[n:].reshape(Qdim)

		Err = I * (P.dot(Q.T) - R)
		P_grad = Err.dot(Q) - (beta*P)
		Q_grad = Err.T.dot(P) - (beta*Q)
		return np.concatenate((P_grad.flatten(), Q_grad.flatten()))

	theta = fmin_cg(cost_function,
					x0 = params,
					fprime = gradient_function,
					args = args,
					epsilon = 0.001,
					maxiter = 5000)

	n = P.shape[0] * P.shape[1]
	P = params[:n].reshape(P.shape)
	Q = params[n:].reshape(Q.shape)
	return P, Q


if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = np.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    rd.seed(3)
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)
    print nP.dot(nQ.T)

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    Pn , Qn =  matrix_factorization_vectorized(R, P, Q, K)
    print Pn.dot(Qn.T)


