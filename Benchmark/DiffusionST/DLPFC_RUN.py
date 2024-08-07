#%%
from Diffusionst.DenoiseST import DenoiseST
import os
import torch
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
from pathlib import Path
#%%
from Diffusionst.repair_model import main_repair
#%%
from Diffusionst.utils import clustering
import os
#%%
from sklearn.metrics import adjusted_rand_score as ari_score


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# n_clusters = 7
# file_fold = '/home/cuiyaxuan/spatialLIBD/151673'
# adata = sc.read_visium(file_fold, count_file='151673_filtered_feature_bc_matrix.h5', load_images=True)
proj_name = '151675'

for proj_name in ['151507', '151508', '151509', '151510','151669','151670','151671','151672', '151673','151674','151675']:
   n_clusters = 5 if proj_name in ['151669', '151670', '151671', '151672'] else 7

   data_root = Path('D:\\project\\datasets\\DLPFC\\')
   adata = sc.read_visium(data_root / proj_name, count_file=proj_name + "_filtered_feature_bc_matrix.h5")
   adata.var_names_make_unique()
   #%%
   model = DenoiseST(adata,device=device,n_top_genes=4096)
   adata = model.train()

   df=pd.DataFrame(adata.obsm['emb'])
   #%%
   main_repair(adata,df,device, save_name=proj_name)
   #%%
   csv_file = proj_name + "_example.csv"
   data_df = pd.read_csv(csv_file, header=None)
   data_df = data_df.values
   adata.obsm['emb'] = data_df

   os.environ['R_HOME'] = 'D:/software/R/R-4.3.2'
   os.environ['R_USER'] = 'D:/software/anaconda/anaconda3/envs/pt20cu118/Lib/site-packages/rpy2'
   radius = 50
   tool = 'mclust' # mclust, leiden, and louvain
   if tool == 'mclust':
      clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
   elif tool in ['leiden', 'louvain']:
      clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

   df=adata.obs['domain']
   df.to_csv("label_" + proj_name + ".csv")
   #%%
   ##### Load layer_guess label, if have
   truth_path = "D:\\project\\datasets\\DLPFC\\" + proj_name + '/' + proj_name + '_truth.txt'
   Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
   Ann_df.columns = ['Ground Truth']
   adata.obs['layer_guess'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
   adata = adata[~pd.isnull(adata.obs['layer_guess'])]

   sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
   ARI = ari_score(sub_adata.obs['layer_guess'], sub_adata.obs['domain'])
   print(f"total ARI:{ARI}")
   #%%
   sc.pl.spatial(adata, color=['domain'], title=[proj_name+':'+str(ARI)],show=True)
   #%%