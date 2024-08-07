import numpy as np
import scanpy as sc
import pandas as pd
import harmonypy as hm
import matplotlib.pyplot as plt
import squidpy as sq
from scipy.spatial import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from scipy.spatial.distance import *
def res_search(adata, target_k=7, res_start=0.1, res_step=0.1, res_epochs=10):
    """
        adata: the Anndata object, a dataset.
        target_k: int, expected number of clusters.
        res_start: float, starting value of resolution. default: 0.1.
        res_step: float, step of resoution. default: 0.1.
        res_epochs: int, epoch of resolution. default: 10.
    """

    print(f"searching resolution to k={target_k}")
    res = res_start
    sc.tl.leiden(adata, resolution=res)

    old_k = len(adata.obs['leiden'].cat.categories)
    print("Res = ", res, "Num of clusters = ", old_k)

    run = 0
    while old_k != target_k:
        old_sign = 1 if (old_k < target_k) else -1
        sc.tl.leiden(adata, resolution=res + res_step * old_sign)
        new_k = len(adata.obs['leiden'].cat.categories)
        print("Res = ", res + res_step * old_sign, "Num of clusters = ", new_k)
        if new_k == target_k:
            res = res + res_step * old_sign
            print("recommended res = ", str(res))
            return res
        new_sign = 1 if (new_k < target_k) else -1
        if new_sign == old_sign:
            res = res + res_step * old_sign
            print("Res changed to", res)
            old_k = new_k
        else:
            res_step = res_step / 2
            print("Res changed to", res)
        if run > res_epochs:
            print("Exact resolution not found")
            print("Recommended res = ", str(res))
            return res
        run += 1
    print("Recommended res = ", str(res))
    return res


def _compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val) / len(clusterlabel)


def fx_1NN(i, location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    return np.min(dist_array)


def fx_kNN(i, location_in, k, cluster_in):
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)

    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    else:
        return 0


def _compute_PAS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results) / len(clusterlabel)

def compute_ARI(adata, gt_key, pred_key):
    return adjusted_rand_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_NMI(adata, gt_key, pred_key):
    return normalized_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_CHAOS(adata, pred_key, spatial_key='spatial'):
    return _compute_CHAOS(adata.obs[pred_key], adata.obsm[spatial_key])

def compute_PAS(adata, pred_key, spatial_key='spatial'):
    return _compute_PAS(adata.obs[pred_key], adata.obsm[spatial_key])

def compute_ASW(adata, pred_key, spatial_key='spatial'):
    d = squareform(pdist(adata.obsm[spatial_key]))
    return silhouette_score(X=d, labels=adata.obs[pred_key], metric='precomputed')

def compute_HOM(adata, gt_key, pred_key):
    return homogeneity_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_COM(adata, gt_key, pred_key):
    return completeness_score(adata.obs[gt_key], adata.obs[pred_key])


import ot


def refine_label(adata, radius=30, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type

def get_metrics(adata, label_key, prediction_key):
    adata = adata[~pd.isnull(adata.obs[label_key])]
    ARI = compute_ARI(adata, label_key, prediction_key)
    NMI = compute_NMI(adata, label_key, prediction_key)
    HOM = compute_HOM(adata, label_key, prediction_key)
    COM = compute_COM(adata, label_key, prediction_key)
    CHAOS = compute_CHAOS(adata, prediction_key)
    PAS = compute_PAS(adata, prediction_key)
    return ARI, (NMI + HOM + COM)/3, (CHAOS + PAS)/2