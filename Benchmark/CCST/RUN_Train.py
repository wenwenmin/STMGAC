import shutil
import warnings

from Data_Preprocess import load_data

warnings.filterwarnings("ignore")
from pathlib import Path
import scanpy as sc
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score, normalized_mutual_info_score
import pandas as pd

# the location of R (used for the mclust clustering)
os.environ['R_HOME'] = 'D:/software/R/R-4.3.2'
os.environ['R_USER'] = 'D:/software/anaconda/anaconda3/envs/pt20cu118/Lib/site-packages/rpy2'

from CCST_Model import build_model

import argparse
import torch


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--model_name", type=str, default="ccst")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", type=str, default="../dataset/DLPFC")
    parser.add_argument("--dataset_name", type=str, default="151507")

    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument("--max_epoch", type=int, default=5000, help="number of training epochs")

    parser.add_argument("--hidden_dims", type=int, default=1024)
    parser.add_argument('--lambda_I', type=float, default=0.3)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def mclust_R(args, adata, random_seed=2023):
    num_cluster = args.cluster_n
    model_name = 'EEE'
    used_obsm = args.key_added_pred

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, model_name)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def train(args, adata, n_clusters):
    set_random_seed(args.seed)
    data, _, adata = load_data(args, adata, n_clusters, pca_dims=200, radius=args.radius)
    data = data.to(args.device)
    model = build_model(args).to(args.device)
    args.lr = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_iter = tqdm(range(args.max_epoch))
    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        epoch_iter.set_description(f"Dataset_Name:{args.dataset_name}, Ep {epoch}: train loss:{loss.item():.4f}")

    args.key_added_pred = args.model_name + '_pred'
    nps = 30
    with torch.no_grad():
        model.eval()
        pos_z, neg_z, summary = model(data)
        rep = pos_z.to('cpu').detach().numpy()
        from sklearn.decomposition import PCA
        pca = PCA(n_components=nps)
        X_PC = pca.fit_transform(rep)
        adata.obsm[args.key_added_pred] = X_PC

    adata = mclust_R(args, adata)
    # obs_df = adata.obs.dropna()
    # ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
    # NMI = normalized_mutual_info_score(obs_df['mclust'], obs_df['Ground Truth'])
    # print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
    # adata.write("./res/DLPFC/" + args.dataset_name + "_results.h5ad")
    return adata


def dlpfc(args):
    ari_list = []
    nmi_list = []
    for tissue_name in ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672",
                        "151673", "151674", "151675", "151676"]:
        n_clusters = 5 if tissue_name in ['151669', '151670', '151671', '151672'] else 7
        args.dataset_name = tissue_name
        data_root = Path("D:\\project\\datasets\\DLPFC\\")
        count_file = tissue_name + "_filtered_feature_bc_matrix.h5"
        adata = sc.read_visium(data_root / tissue_name, count_file=count_file)
        truth_path = "D:\\project\\datasets\\DLPFC\\" + tissue_name + '/' + tissue_name + '_truth.txt'
        Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
        Ann_df.columns = ['Ground Truth']
        adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
        adata = train(args, adata, n_clusters)
        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
        NMI = normalized_mutual_info_score(obs_df['mclust'], obs_df['Ground Truth'])
        print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
        ari_list.append(ARI)
        nmi_list.append(NMI)
    mid_ari = np.median(ari_list)
    mid_nmi = np.median(nmi_list)
    return mid_ari, mid_nmi


def train_dlpfc():
    args = build_args()
    args.top_genes = 2000
    args.radius = 200
    ARI, NMI = dlpfc(args)
    print(f"ari:{ARI:.4f},nmi:{NMI:.4f}")


def hmelanoma(args):
    args.dataset_name = 'HM'
    n_clusters = 4

    adata = sc.read('D:/project/datasets/Melanoma/reading_h5/Melanoma_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    ##### Load layer_guess label, if have
    truth_path = 'D:/project/datasets/Melanoma/manual_annotations.csv'
    Ann_df = pd.read_csv(truth_path, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['layer_guess'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    adata = train(args, adata, n_clusters)
    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['mclust'], obs_df['layer_guess'])
    NMI = normalized_mutual_info_score(obs_df['mclust'], obs_df['layer_guess'])
    print(f"final_ari:{ARI:.4f},final_nmi:{NMI:.4f}")
    return ARI

def train_hmela():
    args = build_args()
    args.top_genes = 3000
    args.radius = 1
    ari = hmelanoma(args)


if __name__ == '__main__':
    # train_dlpfc()
    # train_brca()
    train_hmela()
    # train_mousebrain()
