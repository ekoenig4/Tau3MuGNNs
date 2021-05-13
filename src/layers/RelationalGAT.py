# -*- coding: utf-8 -*-

"""
Created on 2021/4/20

@author: Siqi Miao
"""

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class RelationalGATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads, virtual_node, do_dropout):
        super(RelationalGATConv, self).__init__(node_dim=0)

        self.out_channels = out_channels
        self.heads = heads
        self.virtual_node = virtual_node
        self.do_dropout = do_dropout

        self.linear = nn.Linear(in_channels, heads * out_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.att_l_inter = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r_inter = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_l_intra = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r_intra = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_l_virtual = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r_virtual = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.leakyrelu = nn.LeakyReLU()
        self.downsample_out = nn.Linear(heads * out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.att_l_inter)
        nn.init.zeros_(self.att_r_inter)
        nn.init.zeros_(self.att_l_intra)
        nn.init.zeros_(self.att_r_intra)
        nn.init.zeros_(self.att_l_virtual)
        nn.init.zeros_(self.att_r_virtual)

    def forward(self, x, edge_index, edge_emb):
        """
            E = Number of edges
            C = Hidden size
            H = Number of heads
            N = Number of nodes
        :param x: N x D
        :param edge_index: E x 2
        :param edge_emb: E x NumEdgeFeatures
        :return:
        """
        intra_level_edge_index, inter_level_edge_index, virtual_edge_index = edge_index
        intra_level_edge_emb, inter_level_edge_emb, virtual_edge_emb = edge_emb

        x = self.linear(x).view(-1, self.heads, self.out_channels)  # N x H x C

        alpha_l_intra = (x * self.att_l_intra).sum(dim=-1)  # N x H
        alpha_r_intra = (x * self.att_r_intra).sum(dim=-1)  # N x H
        alpha_l_inter = (x * self.att_l_inter).sum(dim=-1)  # N x H
        alpha_r_inter = (x * self.att_r_inter).sum(dim=-1)  # N x H
        alpha_l_virtual = (x * self.att_l_virtual).sum(dim=-1)  # N x H
        alpha_r_virtual = (x * self.att_r_virtual).sum(dim=-1)  # N x H

        intra_level_out = x
        if intra_level_edge_index.shape[1] != 0:
            intra_level_out = self.propagate(intra_level_edge_index, x=x,
                                             alpha=(alpha_l_intra, alpha_r_intra),
                                             edge_emb=intra_level_edge_emb)  # N x H x C

        if self.virtual_node:
            index = torch.cat((inter_level_edge_index, virtual_edge_index), dim=1)
            emb = torch.cat((inter_level_edge_emb, virtual_edge_emb), dim=0)

            out = self.propagate(index, x=intra_level_out,
                                 alpha=(alpha_l_virtual, alpha_r_virtual),
                                 edge_emb=emb)  # N x H x C
        else:
            out = intra_level_out
            if inter_level_edge_index.shape[1] != 0:
                out = self.propagate(inter_level_edge_index, x=intra_level_out,
                                     alpha=(alpha_l_inter, alpha_r_inter),
                                     edge_emb=inter_level_edge_emb)  # N x H x C

        out = self.downsample_out(out.view(-1, self.heads * self.out_channels))  # N x (HxC) -> N x C
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
        #  if self.do_dropout:
        #      alpha = self.dropout(alpha)
        return alpha * (x_j + edge_emb.unsqueeze(1))
