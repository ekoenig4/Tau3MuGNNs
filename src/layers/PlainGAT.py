# -*- coding: utf-8 -*-

"""
Created on 2021/3/27

@author: Siqi Miao
"""

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class GATConvWithEdgeAttr(MessagePassing):

    def __init__(self, in_channels, out_channels, heads):
        super(GATConvWithEdgeAttr, self).__init__(node_dim=0)

        self.out_channels = out_channels
        self.heads = heads

        self.linear_in = nn.Linear(in_channels, heads * out_channels)
        self.linear_out = nn.Linear(heads * out_channels, out_channels)

        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.leakyrelu = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.att_l)
        nn.init.xavier_normal_(self.att_r)

    def forward(self, x, edge_index, edge_emb):
        """
            E = Number of edges
            C = Hidden size
            H = Number of heads
            N = Number of nodes
        :param x: N x D
        :param edge_index: E x 2
        :param edge_attr: E x NumEdgeFeatures
        :return:
        """

        x = self.linear_in(x).view(-1, self.heads, self.out_channels)  # N x H x C
        alpha_l = (x * self.att_l).sum(dim=-1)  # N x H
        alpha_r = (x * self.att_r).sum(dim=-1)  # N x H

        out = self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r), edge_emb=edge_emb)  # N x H x C
        out = self.linear_out(out.view(-1, self.heads * self.out_channels))
        return out  # N x C

    def message(self, x_j, edge_emb, alpha_i, alpha_j, index):
        """

        :param x_j: E x H x C
        :param edge_emb: E x C
        :param alpha_i: E x H
        :param alpha_j: E x H
        :param index:
        :return:
        """
        alpha = self.leakyrelu(alpha_i + alpha_j)  # E x H
        alpha = softmax(alpha, index).unsqueeze(-1)  # E x H
        return alpha * (x_j + edge_emb.unsqueeze(1))

