from deepst import DeepST
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import pandas as pd
import numpy as np
import os
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score
from sklearn.cluster import KMeans
# the location of R (used for the mclust clustering)
os.environ['R_HOME'] = 'D:/software/R/R-4.3.2'
os.environ['R_USER'] = 'D:/software/anaconda/anaconda3/envs/pt20cu118/Lib/site-packages/rpy2'

def train_one_tissue(tissue_name, save_path='./'):
    n_domains = 5 if tissue_name in ['151669', '151670', '151671', '151672'] else 7
    dst = DeepST.run(save_path=save_path, task="Identify_Domain", pre_epochs=1000, epochs=500, use_gpu=True)
    adata = dst._get_adata(platform='Visium', data_path=Path("D:\\project\\datasets\\DLPFC_Simple\\"), data_name=tissue_name, count_file=tissue_name + "_filtered_feature_bc_matrix.h5")
    ###### Segment the Morphological Image
    adata = dst._get_image_crop(adata, data_name=tissue_name)
    ###### "use_morphological" defines whether to use morphological images.
    adata = dst._get_augment(adata, spatial_type="LinearRegress", use_morphological=True)
    ###### Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py
    graph_dict = dst._get_graph(adata.obsm["spatial"], distType="BallTree")
    ###### Enhanced data preprocessing
    data = dst._data_process(adata, pca_n_comps=200)

    ###### Training models
    deepst_embed = dst._fit(
        data=data,
        graph_dict=graph_dict, )
    ###### DeepST outputs
    adata.obsm["DeepST_embed"] = deepst_embed
    ###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
    adata = dst._get_cluster_data(adata, n_domains=n_domains, priori=True)

    truth_path = "D:\\project\\datasets\\DLPFC\\" + tissue_name + '/' + tissue_name + '_truth.txt'
    Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['DeepST_refine_domain'], obs_df['Ground Truth'])
    NMI = v_measure_score(obs_df['DeepST_refine_domain'], obs_df['Ground Truth'])
    print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")

    ###### Spatial localization map of the spatial domain
    sc.pl.spatial(adata, color='DeepST_refine_domain', frameon=False, spot_size=150)
    # plt.savefig(os.path.join(save_path, f'{tissue_name}_domains.pdf'), bbox_inches='tight', dpi=300)
    # adata.write(save_path + tissue_name + '_results.h5ad')

def train_dlpfc():
    save_path = './output/DLPFC/'
    for tissue_name in ["151672"]:
        train_one_tissue(tissue_name, save_path)


train_dlpfc()