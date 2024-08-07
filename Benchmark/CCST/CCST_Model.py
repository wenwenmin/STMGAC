import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_4 = GCNConv(hidden_channels, hidden_channels)

        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.conv_2(x, edge_index, edge_weight=edge_weight)
        x = self.conv_3(x, edge_index, edge_weight=edge_weight)
        x = self.conv_4(x, edge_index, edge_weight=edge_weight)
        x = self.prelu(x)
        return x


class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

def corruption(data): # 打乱节点数据，但不破坏图结构。也就是说节点特征矩阵中，将节点顺序重新打乱。特征值不变
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)
# torch.randperm(data.x.size(0)) 是在PyTorch中生成一个随机的排列（permutation）。
# 具体来说，这个函数会生成一个包含0到data.x.size(0)-1的随机整数序列，其中data.x.size(0)是数据集的样本数量。
# 这个随机排列可以用于多种情况，例如在数据洗牌（shuffle）或者在某些需要随机顺序的操作中。

def build_model(args):
    model = DeepGraphInfomax(
            hidden_channels=args.hidden_dims,
            encoder=Encoder(in_channels=args.spots_input_dims, hidden_channels=args.hidden_dims),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption
        )
    return model


