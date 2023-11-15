import numpy as np
from sklearn.cluster import KMeans

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

    kmeans = KMeans(n_clusters=k).fit(U)
    return kmeans.labels_