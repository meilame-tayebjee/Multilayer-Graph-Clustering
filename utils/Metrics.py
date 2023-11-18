import numpy as np
from utils.Algorithms import *


#--------------------------#
""" 
Metrics for clustering cited in the paper, namely:
- Purity
- NMI
- Rand Index
"""
#--------------------------#

def Purity(cluster_assignment, true_clusters):
    """
    Compute the purity metric of a clustering assignment, given the true clusters.
    See Equation 19 of the paper

    Parameters
    ----------
    cluster_assignment : list or numpy array of length n
        List of cluster assignments for each node
    true_clusters : list or numpy array of length n
        List of true cluster assignments for each node

    Returns
    -------
    purity : float
        Purity of the clustering assignment, best being 1 and worst being 0
    """
    n = len(cluster_assignment)
    k = true_clusters.max() + 1
    purity = 0
    for i in range(k):
        max_count = 0
        for j in range(k):
            count = 0
            for l in range(n):
                if cluster_assignment[l] == i and true_clusters[l] == j:
                    count += 1
            if count > max_count:
                max_count = count
        purity += max_count
    purity /= n
    return purity

def NMI(cluster_assignment, true_clusters):
    """
    Compute the normalized mutual information metric of a clustering assignment, given the true clusters.
    See Equation 20 of the paper

    Parameters
    ----------
    cluster_assignment : list or numpy array of length n
        List of cluster assignments for each node
    true_clusters : list or numpy array of length n
        List of true cluster assignments for each node

    Returns
    -------
    NMI : float
        Normalized mutual information of the clustering assignment, best being 1 and worst being 0
    """

    n = len(cluster_assignment)
    k = true_clusters.max() + 1 # number of clusters
    I = 0 # mutual information
    H_C = 0 # entropy of cluster_assignment
    H_T = 0 # entropy of true_clusters

    #We iterate over the clusters i
    for i in range(k):
        count_C = np.sum(cluster_assignment == i) # number of elements in predicted cluster i
        count_T = np.sum(true_clusters == i) # number of elements in true cluster i

        H_C += -count_C/n*np.log(count_C/n)  #add the entropy of cluster i
        H_T += -count_T/n*np.log(count_T/n)
        
        for j in range(k):
            count = np.sum(np.logical_and(cluster_assignment == i, true_clusters == j)) # number of elements in predited cluster i and true cluster j
            if count != 0:
                I += count/n*np.log(n*count/(count_C*count_T))
    return I/(np.sqrt(H_C*H_T))


def RI(cluster_assignment, true_clusters):
    """
    Compute the rand index metric of a clustering assignment, given the true clusters.
    See Equation 21 of the paper

    Parameters
    ----------
    cluster_assignment : list or numpy array of length n
        List of cluster assignments for each node
    true_clusters : list or numpy array of length n
        List of true cluster assignments for each node
    
    Returns
    -------
    RI : float
        Rand index of the clustering assignment, best being 1 and worst being 0
    """
    n = len(cluster_assignment)
    TP = 0 # true positive
    TN = 0 # true negative
    FP = 0 # false positive
    FN = 0 # false negative
    for i in range(n):
        for j in range(i+1,n):
            if cluster_assignment[i] == cluster_assignment[j] and true_clusters[i] == true_clusters[j]:
                TP += 1
            elif cluster_assignment[i] != cluster_assignment[j] and true_clusters[i] != true_clusters[j]:
                TN += 1
            elif cluster_assignment[i] == cluster_assignment[j] and true_clusters[i] != true_clusters[j]:
                FP += 1
            else:
                FN += 1
    return (TP+TN)/(TP+TN+FP+FN)