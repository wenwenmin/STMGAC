import copy
import numpy as np
import random
import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import (
    TransformerConv,
    LayerNorm,
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

import faiss

def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def full_block(in_features, out_features, p_drop, act=nn.ELU()):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        act,  # nn.ELU(),
        nn.Dropout(p=p_drop),
    )

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True, graphtype="gcn"):
        super(GraphConv, self).__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn(out_features)
        self.act = act
        self.dropout = dropout
        if graphtype == "gcn":
            self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gat": # Default heads=1
            self.conv = GATConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gin": # Default heads=1
            self.conv = TransformerConv(in_channels=self.in_features, out_channels=self.out_features)
        else:
            raise NotImplementedError(f"{graphtype} is not implemented.")

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = config['feat_hidden1']
        self.feat_hidden2 = config['feat_hidden2']
        self.gcn_hidden = config['gcn_hidden']
        self.latent_dim = config['latent_dim']

        self.p_drop = config['p_drop']
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))
        # GCN layers
        self.gc1 = GraphConv(self.feat_hidden2, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.gc2 = GraphConv(self.gcn_hidden, self.latent_dim, dropout=self.p_drop, act=lambda x: x)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)
        return x

class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['latent_dim']
        self.gcn_hidden = config['project_dim']
        self.p_drop = config['p_drop']
        self.layer1 = GraphConv(self.input_dim, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.layer2 = nn.Linear(self.gcn_hidden, self.input_dim, bias=False)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, config):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = config['latent_dim']
        self.p_drop = config['p_drop']
        self.layer1 = GraphConv(self.input_dim, self.output_dim, dropout=self.p_drop, act=nn.Identity())

    def forward(self, x, edge_index):
        return self.layer1(x, edge_index)


class Neighbor(nn.Module):
    def __init__(self, device, num_centroids, num_kmeans, clus_num_iters):
        super(Neighbor, self).__init__()
        self.device = device
        self.num_centroids = num_centroids
        self.num_kmeans = num_kmeans
        self.clus_num_iters = clus_num_iters

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei

    def forward(self, adj, student, teacher, top_k):
        n_data, d = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
        similarity += torch.eye(n_data, device=self.device) * 10

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(self.device)

        knn_neighbor = self.create_sparse(I_knn)
        locality = knn_neighbor * adj

        ncentroids = self.num_centroids
        niter = self.clus_num_iters

        pred_labels = []

        for seed in range(self.num_kmeans):
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=seed + 1234)
            kmeans.train(teacher.cpu().numpy())
            _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)

            clust_labels = I_kmeans[:, 0]

            pred_labels.append(clust_labels)

        pred_labels = np.stack(pred_labels, axis=0)
        cluster_labels = torch.from_numpy(pred_labels).long().to(self.device)

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn, I_knn.shape[1])

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)

        globality = self.create_sparse_revised(I_knn, all_close_nei_in_back)

        pos_ = locality + globality
        ind = pos_.coalesce()._indices()
        anchor = ind[0]
        positive = ind[1]
        negative = torch.tensor(random.choices(list(range(n_data)), k=len(anchor))).to(self.device)
        return anchor, positive, negative

    def create_sparse(self, I):

        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])

        assert len(similar) == len(index)
        indices = torch.tensor([index, similar]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data])

        return result

class AFRM(nn.Module):
    def __init__(self, num_clusters, input_dim, config, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.dec_in_dim = config['latent_dim']
        self.online_encoder = Encoder(input_dim, config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self._init_target()
        self.projector = Projector(config)
        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.decoder = Decoder(input_dim, config)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.rep_mask = nn.Parameter(torch.zeros(1, self.dec_in_dim))
        self.neighbor = Neighbor(device, num_clusters, config['num_kmeans'], config['clus_num_iters'])
        self.topk = config['topk']
        self.mask_rate = config['mask_rate']
        self.t = config['t']
        self.momentum_rate = config['momentum_rate']
        self.replace_rate = config['replace_rate']
        self.mask_token_rate = 1 - self.replace_rate
        self.anchor_pair = None

    def _init_target(self):
        for param_teacher in self.target_encoder.parameters():
            param_teacher.detach()
            param_teacher.requires_grad = False

    def momentum_update(self, base_momentum=0.1):
        for param_encoder, param_teacher in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + param_encoder.data * (1. - base_momentum)

    def encoding_mask_noise(self, x, edge_index, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]

        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_edge_index = edge_index.clone()

        return out_x, use_edge_index, (mask_nodes, keep_nodes)

    def mask_attr_prediction(self, x, edge_index, flag):
        use_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, edge_index, self.mask_rate)
        enc_rep = self.online_encoder(use_x, use_adj)

        with torch.no_grad():
            x_t = x.clone()
            x_t[keep_nodes] = 0.0
            x_t[keep_nodes] += self.enc_mask_token
            enc_rep_t = self.target_encoder(x_t, use_adj)
            rep_t = enc_rep_t
            self.momentum_update(self.momentum_rate)

        rep = enc_rep
        rep = self.encoder_to_decoder(rep)
        rep[mask_nodes] = 0.
        rep[mask_nodes] += self.rep_mask
        rep = self.projector(rep, use_adj)

        online = rep[mask_nodes]
        target = rep_t[mask_nodes]

        rep[keep_nodes] = 0.0
        recon = self.decoder(rep, use_adj)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        if flag:
            self.anchor_pair = self.neighbor(use_adj, F.normalize(enc_rep, dim=-1, p=2), F.normalize(enc_rep_t, dim=-1, p=2), self.topk)
        if self.anchor_pair is not None:
            anchor, positive, negative = self.anchor_pair
            tri_loss = self.triplet_loss(enc_rep, anchor, positive, negative)
        else:
            tri_loss = 0

        mean_loss = F.mse_loss(online, target)
        rec_loss = self.sce_loss(x_rec, x_init, t=self.t)

        return mean_loss, rec_loss, tri_loss


    def sce_loss(self, x, y, t=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
        loss = -torch.log(cos_m.pow_(t))
        return loss.mean()

    def triplet_loss(self, emb, anchor, positive, negative, margin=1.0):
        anchor_arr = emb[anchor]
        positive_arr = emb[positive]
        negative_arr = emb[negative]
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        return tri_output

    def forward(self, x, edge_index, flag):
        return self.mask_attr_prediction(x, edge_index, flag)

    @torch.no_grad()
    def evaluate(self, x, edge_index):
        enc_rep = self.online_encoder(x, edge_index)
        rep = self.encoder_to_decoder(enc_rep)
        rep = self.projector(rep, edge_index)
        recon = self.decoder(rep, edge_index)
        return enc_rep, recon