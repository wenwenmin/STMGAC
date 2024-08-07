import os
import torch
import torch.nn.modules.loss
from tqdm import tqdm
from torch.backends import cudnn

from .Models import *

class Mgac:
    def __init__(self, adata, graph_dict, num_clusters,  device, config, roundseed=0):
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.graph_dict = graph_dict
        self.mode = config['mode']
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters


    def _start_(self):
        if self.mode == 'clustering':
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        elif self.mode == 'imputation':
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            raise Exception
        self.adj_norm = self.graph_dict["adj_norm"].to(self.device)
        self.adj_label = self.graph_dict["adj_label"].to(self.device)
        self.norm_value = self.graph_dict["norm_value"]

        self.input_dim = self.X.shape[-1]
        self.model = AFRM(self.num_clusters, self.input_dim, self.model_config, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.train_config['lr'],
            weight_decay=self.train_config['decay'],
        )

    def _fit_(self):
        pbar = tqdm(range(self.train_config['epochs']))
        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad()
            flag = False
            if epoch % self.train_config['t_step'] == 0:
                flag = True
            mean_loss, rec_loss, tri_loss = self.model(self.X, self.adj_norm, flag)
            loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_mean'] * mean_loss + self.train_config['w_tri'] * tri_loss
            loss.backward()
            if self.train_config['gradient_clipping'] > 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.train_config['gradient_clipping'])
            self.optimizer.step()
            pbar.set_description(
                "Epoch {0} total loss={1:.3f} recon loss={2:.3f} mean loss={3:.3f} tri loss={4:.3f}".format(
                    epoch, loss, rec_loss, mean_loss, tri_loss),
                refresh=True)

    def trian(self):
        self._start_()
        self._fit_()

    def process(self):
        self.model.eval()
        enc_rep, recon = self.model.evaluate(self.X, self.adj_norm)
        return enc_rep, recon