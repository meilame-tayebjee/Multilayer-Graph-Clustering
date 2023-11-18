import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
from utils.utils import *


#------------ALGORITHMS OF THE PAPER: SC_GED and SC_SR------------------#


#Corps de l'algo à commenter
def SC_GED(W,k, verbose = False):

    """
    Implements the SC-GED algorithm.
    Corresponds to the Algorithm 2 of the paper.

    Parameters:
    ----------
    W (array shape (n,n,M)): The M adjacency matrices of a multi-layer graph.
    k (int): The number of clusters to compute.

    verbose (bool): If True, print the intermediate results of the algorithm.

    Returns:
    -------
    numpy.ndarray shape(n,1): The SC-GED cluster assignement for each node.

    """


    alpha = 10
    beta = 100
    iter = 0
    iter_max = 20
    stopping_condition = False
    condition = 0.001

    n,_,M = W.shape
    L = np.zeros(np.shape(W))
    delta = np.zeros(np.shape(W))
    
    for i in range(M):
        W_i = W[: , : ,i]
        D = computeDegreeMatrix(W_i)
        L[:, :, i] = np.linalg.inv(D)@(D-W_i)
        eigenvalues,eigenvectors = np.linalg.eig(L[:, :, i])
        delta[: , :, i] = np.diag(eigenvalues)

    eigenvalues,eigenvectors = np.linalg.eig(L[:, :, 0])
    P = eigenvectors
    Q = np.linalg.inv(P)

    while(stopping_condition == False and iter<iter_max):

        if verbose:
            print("iter : ",iter)

        def function_to_optimize_P(P_vectorized):
            P = P_vectorized.reshape((n,n))
            res = 0.5*alpha*(np.linalg.norm(P,'fro')**2 + np.linalg.norm(Q,'fro')**2) + 0.5*beta*np.linalg.norm(P@Q-np.identity(n),'fro')**2
            for i in range(M):
                res += 0.5*np.linalg.norm(L[: , :, i] - P@delta[:  ,:, i]@Q,'fro')**2
            return res

        def function_to_optimize_P_gradient(P_vectorized):
            P = P_vectorized.reshape((n,n))
            res = alpha*P + beta*(P@Q - np.identity(n))@np.transpose(Q)
            for i in range(M):
                res = res - (L[:, :, i] - P@delta[: , :, i]@Q)@np.transpose(Q)@delta[:, :, i]
            return res.flatten()


        x0 = P.flatten()
        val_init = function_to_optimize_P(x0)
        if verbose and iter == 0:
                print("Valeur initiale : ", val_init)

        result_optimization_P = minimize(function_to_optimize_P, x0,method='L-BFGS-B',jac = function_to_optimize_P_gradient)
        P = result_optimization_P.x.reshape((n,n))

        if verbose:
            print("Optimisation sur P, résultat : ", result_optimization_P.success , ", message : ", result_optimization_P.message)
            print(result_optimization_P.fun)


        def function_to_optimize_Q(Q_vectorized):
            Q = Q_vectorized.reshape((n,n))
            res = 0.5*alpha*( np.linalg.norm(P,'fro')**2 + np.linalg.norm(Q,'fro')**2 ) + 0.5*beta*np.linalg.norm(P@Q-np.identity(n),'fro')**2
            for i in range(M):
                res += 0.5*np.linalg.norm(L[: ,:, i] - P@delta[:, :, i]@Q,'fro')**2
            return res

        def function_to_optimize_Q_gradient(Q_vectorized):
            Q = Q_vectorized.reshape((n,n))
            res = alpha*Q + beta*P.T@(P@Q - np.identity(n))
            for i in range(M):
                res = res - delta[:, :, i] @ P.T @ (L[:, :, i] - P@delta[:, :, i]@Q)
            return res.flatten()

        x0 = Q.flatten()
        result_optimization_Q = minimize(function_to_optimize_Q, x0,method='L-BFGS-B',  jac = function_to_optimize_Q_gradient)
        Q = result_optimization_Q.x.reshape((n,n))
        val_end = result_optimization_Q.fun

        if verbose:
            print("Optimisation sur Q, résultat : ", result_optimization_Q.success , ", message : ", result_optimization_Q.message)
            print(result_optimization_Q.fun)


        if(val_init - val_end < condition):
            stopping_condition = True

        iter += 1

    U = P
    U = U[:,:k]
    return k_means_cluster(U,k) 

def SC_SR(W,k):
    "TO DO"

    return 0


#-----------------------BASELINE ALGORITHMS----------------------#
#-----------------See Section VI-B of the paper------------------#
#--------------Includes SC-SUM, K-KMeans, SC-AL------------------#

def SC_SUM(adj_matrix, k, normalized = False):
    """
    Spectral Clustering with summation of adjacency matrices
    See Eq 15 of the paper

    Parameters
    ----------
    adj_matrix : numpy array of shape (M,n,n)
    k (int): number of target clusters
    normalized (bool): whether to use normalized adjacency matrces

    Returns
    -------
    numpy array of shape (n,k) : cluster assignment matrix
    """
    n,_,M = adj_matrix.shape # M is the number of clusters, n is the number of nodes
    
    if normalized:
        W = np.zeros((n,n))
        for i in range(M):
            W_i = adj_matrix[:,:,i]
            D_i = computeDegreeMatrix(W_i)
            W += (np.sqrt(np.linalg.inv(D_i))) @ W_i @ (np.sqrt(np.linalg.inv(D_i)))

        return spectralClustering(W, k)


    else:
        W = np.sum(adj_matrix, axis=-1) #summation of the M adjacency matrices
        return spectralClustering(W, k)  

def K_KMeans(adj_matrix, d, k):
    "TO DO"
    return 0

def SC_AL(adj_matrix, k):
    """
    Spectral Clustering with average of Laplacian matrices
    See Eq 18 of the paper

    Parameters
    ----------
    adj_matrix : numpy array of shape (M,n,n)
    k (int): number of target clusters

    Returns
    -------
    numpy array of shape (n,k) : cluster assignment matrix
    """
    
    n,_,M = adj_matrix.shape # M is the number of clusters, n is the number of nodes

    #Computation of the average on the M Laplacian matrices
    L = np.zeros((n,n))
    for i in range(M):
        W_i = adj_matrix[:,:, i]
        D_i = computeDegreeMatrix(W_i)
        Lrw_i = np.linalg.inv(D_i) @ (D_i - W_i)
        L += Lrw_i
    
    L = L/M

    eigvals, eigvecs = np.linalg.eig(L)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    U = eigvecs[:,np.argsort(eigvals)[:k]] # eigenvectors corrresponding to the k smallest eigenvalues; shape (n,k)

    kmeans = KMeans(n_clusters=k).fit(U)

    return kmeans.labels_
