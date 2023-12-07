import numpy as np
import time
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
from utils.utils import *
from mvlearn.cluster import MultiviewCoRegSpectralClustering



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
    startTime = time.time()

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

    runTime = time.time() - startTime
    return k_means_cluster(U,k), runTime

def twoLayerStepSCSR(f,W, mu):
  """
  Perform one step of the SC-SR algorithm

  Parameters
  ----------
    f : array shape ((n,k))
        The vectors f
    W : array shape (n,n)
        The adjacency matrix of the layer we want to integrate
    mu : float
        The parameter mu = 1/lambda


  Returns
  -------

  newf : array shape ((k,n))
       Updated vectors f

  """
  n, k = f.shape
  newf = np.zeros((n,k))
  D_2 = computeDegreeMatrix(W)
  L_sym2 = np.linalg.inv(np.sqrt(D_2))@(D_2-W)@np.linalg.inv(np.sqrt(D_2)) #L_sym of the layer we want to integrate
  for i in range(k):
    newf[:, i] = mu*np.linalg.inv((L_sym2 + mu*np.identity(n)))@f[:,i]

  return newf


def ourSC_SR(W, k, true_clusters, mu_seq):
    """
    This function performs the multilayer spectral clustering with spectral regularization.
    It generalizes the Algorithm 3 of the paper.

    Parameters
    ----------
    W : numpy array shape (n,n,M)
        The adjacency mayrix of the multilayer network.
    
    k : int
        The target number of clusters.

    Returns
    -------
    numpy array shape (n,)
        The clustering of the multilayer network.
    """
    startTime = time.time()
    n,_,M = W.shape
    ranking, _ = rankingInformativeLayers(W, k, true_clusters)
    
    #Initialization: we take the best layer and compute the k first eigenvectors of its Laplacian
    #---------------------#
    bestLayerMatrix = W[:, :, ranking[0]] #shape (n,n,1) containing the adjacency matrices of the best layer
    D = computeDegreeMatrix(bestLayerMatrix)
    L = np.linalg.inv(D)@(D-bestLayerMatrix)
    eigvals,U = np.linalg.eig(L) #U.shape = (n,n), contains the eigenvectors of the best layer
    f = U[:,np.argsort(eigvals)[:k]] #f.shape = (n,k), contains the k first eigenvectors of the best layer
    #---------------------#

    #Integration of the other layers by updating the fs
    for i in range(1, M):
        nextLayerMatrix = W[:, :, ranking[i]]
        f = twoLayerStepSCSR(f,nextLayerMatrix, float(mu_seq[0]))
        mu_seq = mu_seq[1:]
    
    runTime = time.time() - startTime
    return k_means_cluster(f,k), runTime


def SC_SR(W, k, true_clusters, mu_seq):
    """
    This function performs the multilayer spectral clustering with spectral regularization.
    It generalizes the Algorithm 3 of the paper.

    Parameters
    ----------
    W : numpy array shape (n,n,M)
        The adjacency mayrix of the multilayer network.
    k : int
        The target number of clusters.
    mu_seq : numpy array shape (M-1,)
        The sequence of mu = 1/lambda values

    Returns
    -------
    numpy array shape (n,)
        The clustering of the multilayer network.
    """
    startTime = time.time()
    n,_,M = W.shape
    ranking, _ = rankingInformativeLayers(W, k, true_clusters)
    non_integrated_layers = list(range(M))
    #Initialization: we take the best layer and compute the k first eigenvectors of its Laplacian
    #---------------------#
    bestLayerMatrix = W[:, :, ranking[0]] #shape (n,n,1) containing the adjacency matrices of the best layer
    D = computeDegreeMatrix(bestLayerMatrix)
    L = np.linalg.inv(D)@(D-bestLayerMatrix)
    eigvals,U = np.linalg.eig(L) #U.shape = (n,n), contains the eigenvectors of the best layer
    f = U[:,np.argsort(eigvals)[:k]] #f.shape = (n,k), contains the k first eigenvectors of the best layer

    current_clustering = k_means_cluster(f,k)
    non_integrated_layers.remove(ranking[0])
    #---------------------#

    #Integration of the other layers by updating the fs
    while len(non_integrated_layers) > 0:
        nextLayer = searchNextLayer(W, k, current_clustering, non_integrated_layers)
        nextLayerMatrix = W[:, :, nextLayer]
        f = twoLayerStepSCSR(f,nextLayerMatrix, float(mu_seq[0]))

        non_integrated_layers.remove(nextLayer)
        current_clustering = k_means_cluster(f,k)
        mu_seq = mu_seq[1:]
    
    runTime = time.time() - startTime
    return k_means_cluster(f,k), runTime



#-----------------------BASELINE ALGORITHMS----------------------#
#-------------------See Section VI-B of the paper----------------#
#-------------------Includes SC-SUM, SC-AL, CoR------------------#

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
    startTime = time.time()

    n,_,M = adj_matrix.shape # M is the number of clusters, n is the number of nodes
    
    if normalized:
        W = np.zeros((n,n))
        for i in range(M):
            W_i = adj_matrix[:,:,i]
            D_i = computeDegreeMatrix(W_i)
            W += (np.sqrt(np.linalg.inv(D_i))) @ W_i @ (np.sqrt(np.linalg.inv(D_i)))

        runTime = time.time() - startTime
        return spectralClustering(W, k), runTime


    else:
        W = np.sum(adj_matrix, axis=-1) #summation of the M adjacency matrices
        runTime = time.time() - startTime
        return spectralClustering(W, k), runTime


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
    startTime = time.time()
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
    runTime = time.time() - startTime
    return kmeans.labels_, runTime


def CoR(adj_matrix,k):
    """
    Co-regularized spectral clustering
    
    Parameters
    ----------
    adj_matrix : numpy array of shape (M,n,n)
    k (int): number of target clusters

    Returns
    -------
    numpy array of shape (n,k) : cluster assignment matrix
    """
    startTime = time.time()

    data = []
    for i in range(len(adj_matrix)):
        data.append(adj_matrix[:,:,0])
    data = np.array(data)
    mv_spectral = MultiviewCoRegSpectralClustering(n_clusters=k)

    labels = mv_spectral.fit_predict(data)

    runTime = time.time() - startTime
    return labels, runTime

