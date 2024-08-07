import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.neighbors
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import scipy.sparse as sp
from torch_geometric.data import Data

# the location of R (used for the mclust clustering)
os.environ['R_HOME'] = 'D:/software/R/R-4.3.2'
os.environ['R_USER'] = 'D:/software/anaconda/anaconda3/envs/pt20cu118/Lib/site-packages/rpy2'

def load_data(args, adata, n_clusters, pca_dims=200, radius=300):
    adata.var_names_make_unique()
    if isinstance(adata.X, np.ndarray):
        adata.layers['count'] = adata.X
    else:
        adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=args.top_genes)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
    adata_X = PCA(n_components=pca_dims, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    args.spots_input_dims = adata_X.shape[1]

    args.cluster_n = n_clusters

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    adata.uns['Spatial_Net'] = Spatial_Net

    plotnet = True
    if plotnet:
        Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
        Mean_edge = Num_edge / adata.shape[0]
        print('Number of Neighbors (Mean=%.2f)' % Mean_edge)

    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = (1 - args.lambda_I) * G + args.lambda_I * sp.eye(G.shape[0])
    edge_weight = G.data
    edgeList = np.nonzero(G)
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                x=torch.FloatTensor(adata_X), edge_attr=edge_attr)

    return data, None, adata
