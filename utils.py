"""Fairness measurements."""

import numpy as np

def calc_balance(labels, sens):
    """Calculate balance by checking fairness in each of the clusters.
    
    Fair clustering through fairlets, 2017, Chierichetti, Flavio, et al.
    
    Args:
        labels: labels of each point. ndarray of shape (n_samples,)
        sens: The sensitive attribute of training instance. ndarray of shape (n_samples,).
    
    Returns:
        the balance of clustering.
    """
    labels_unique, n_clusters = np.unique(labels, return_counts=True)
    sens_unique = np.unique(sens)
    fair = dict([(i, [0, 0]) for i in labels_unique])
    
    for i in range(len(labels_unique)):
        fair[labels_unique[i]][1] = n_clusters[i]
        # the number of a special group in the i'th cluster
        fair[labels_unique[i]][0] = len(np.intersect1d(np.where(labels == labels_unique[i]),
                                                       np.where(sens == sens_unique[0])))
    
    curr_b = []
    for i in list(fair.keys()):
        p = fair[i][0]
        q = fair[i][1] - fair[i][0]
        if p == 0 or q == 0:
            balance = 0
        else:
            balance = min(float(p/q), float(q/p))
        curr_b.append(balance)
    
    return min(curr_b)

def cal_balance_v2(labels, sens):
    """Calculate balance by checking fairness in each of the clusters.
    
    Fair algorithms for clustering, 2019, Bera, Suman, et al.
    
    Args:
        labels: labels of each point. ndarray of shape (n_samples,)
        sens: The sensitive attribute of training instance. ndarray of shape (n_samples,).
    
    Returns:
        the balance of clustering.
    """
    n_samples = len(sens)
    labels_unique, n_clusters = np.unique(labels, return_counts=True)
    sens_unique, n_sens = np.unique(sens, return_counts=True)
    r_sens = n_sens / n_samples
    
    curr_b = []
    cl_b = []
    for i in range(len(labels_unique)):
        for j in range(len(sens_unique)):
            r = len(np.intersect1d(np.where(labels == labels_unique[i]),
                                   np.where(sens == sens_unique[j]))) / n_clusters[i]
            if r == 0:
                cl_b.append(0)
            else:
                cl_b.append(min(float(r_sens[j]/r), float(r/r_sens[j])))
        curr_b.append(min(cl_b))
        cl_b = []
    
    return min(curr_b)

def dv_ae(labels, sens):
    """Calculate average euclidean distance between distribution vectors.
    
    Fairness in clustering with multiple sensitive attributes, 2019,
    Abraham, Savitha Sam, and Sowmya S. Sundaram.
    
    Args:
        labels: labels of each point. ndarray of shape (n_samples,)
        sens: The sensitive attribute of training instance. ndarray of shape (n_samples,).
    
    Returns:
        the average euclidean distance between distribution vectors of clustering.
    """
    n_samples = len(sens)
    labels_unique, n_clusters = np.unique(labels, return_counts=True)
    sens_unique, n_sens = np.unique(sens, return_counts=True)
    r_sens = n_sens / n_samples
    
    ae = 0
    for i in range(len(labels_unique)):
        r_cl = np.zeros_like(r_sens)
        for j in range(len(sens_unique)):
            r_cl[j] = len(np.intersect1d(np.where(labels == labels_unique[i]),
                                         np.where(sens == sens_unique[j]))) / n_clusters[i]
        ae += n_clusters[i] * np.linalg.norm(r_cl-r_sens)
    ae /= n_samples
    return ae
