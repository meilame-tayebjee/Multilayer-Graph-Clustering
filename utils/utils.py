import numpy as np
from sklearn.cluster import KMeans
from utils.Metrics import *


def computeDegreeMatrix(M):
    """
    Computes the degree matrix of a given adjacency matrix.

    Parameters:
    M (array shape (n,n)): The adjacency matrix of a graph.

    Returns:
    numpy.ndarray shape(n,n): The degree matrix of the graph.
    """
    res = np.array([0]*len(M))
    for i in range(len(M)):
        for j in range(len(M)):
            if(M[i][j] != 0):
                res[i]+=1    
    
    return np.diag(res)


def k_means_cluster(U,k):
    """
    Computes the k-means clustering of a given matrix.

    Parameters:
    U (array shape (n,k)): The matrix to cluster.
    k (int): The number of clusters to compute.

    Returns:
    numpy.ndarray shape(n,): The k-means cluster assignment of the matrix.
    """
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)
    return kmeans.labels_

def spectralClustering(W, k, normalized=False):
    """
    Computes the normalized spectral clustering of a given adjacency matrix.
    Corresponds to the Algorithm 1 of the paper.

    Parameters:
    W (array shape (n,n)): The adjacency matrix of a graph.
    k (int): The number of clusters to compute.

    Returns:
    numpy.ndarray shape(n,k): The normalized spectral clustering of the graph.
    """

    D = computeDegreeMatrix(W)

    if normalized:
        L = np.linalg.inv(D) @ (D-W) # normalized Laplacian matrix L_rw
    else:
        L = D-W # unnormalized Laplacian matrix L

    eigvals, eigvecs = np.linalg.eig(L)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    U = eigvecs[:,np.argsort(eigvals)[:k]] # eigenvectors corrresponding to the k smallest eigenvalues; shape (n,k)

    return k_means_cluster(U,k)

def rankingInformativeLayers(W, k, true_clusters):
    """
    This function ranks the layers of a multilayer network according to their informativeness.

    Parameters
    ----------
    W : numpy array shape (n,n,M)
        The adjacency mayrix of the multilayer network.
    
    k : int
        The target number of clusters.

    true_clusters : numpy array shape (n,)
        The true cluster assignment of the nodes.

    Returns
    -------
    numpy array shape (M,)
        The ranking of the layers.
    """

    n,_,M = W.shape
    NMIs = np.zeros(M)
    clusterings = []
    for i in range(M):
        W_i = W[:,:,i]
        clustering = spectralClustering(W_i, k)
        NMIs[i] = NMI(clustering, true_clusters)
        clusterings.append(clustering)
    
    return np.argsort(NMIs)[::-1][:n], NMIs, clusterings

def searchNextLayer(W, current_clustering, indexes, clusterings):

    n,_,M = W.shape
    NMIs = np.zeros(M)
    for i in indexes:
        NMIs[i] = NMI(clusterings[i], current_clustering)
    
    NMIs = NMIs[indexes]
    return indexes[np.argmax(NMIs)]