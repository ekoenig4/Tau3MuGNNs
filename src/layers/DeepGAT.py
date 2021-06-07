from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class DeepGATConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int):

        super(DeepGATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.att_l = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, out_channels))
        self.leakyrelu = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.att_l)
        nn.init.kaiming_normal_(self.att_r)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""

        alpha_l = (x * self.att_l).sum(dim=-1).view(-1, 1)  # N x 1
        alpha_r = (x * self.att_r).sum(dim=-1).view(-1, 1)  # N x 1

        out = self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r), edge_attr=edge_attr, size=size)  # N x C
        return out  # N x C

    def message(self, x_j, edge_attr, alpha_i, alpha_j, index):

        alpha = self.leakyrelu(alpha_i + alpha_j)  # E x 1
        alpha = softmax(alpha, index)  # E x 1
        msg = F.leaky_relu(x_j + edge_attr)
        return alpha * msg
