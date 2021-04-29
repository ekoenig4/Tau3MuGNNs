# -*- coding: utf-8 -*-

"""
Created on 2021/4/21

@author: Siqi Miao
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn.conv import TransformerConv

from layers import PlainGAT, RelationalGAT


class Model(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, virtual_node, config):
        super(Model, self).__init__()
        self.x_dim = x_dim
        self.heads = config['heads']
        self.out_channels = config['out_channels']
        self.n_layers = config['n_layers']
        self.model_name = config['model_name']
        self.do_dropout = config['dropout']
        self.virtual_node = virtual_node

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.node_encoder = nn.Linear(x_dim, self.out_channels)
        self.edge_encoder = nn.Linear(edge_attr_dim, self.out_channels)

        for _ in range(self.n_layers):
            self.convs.append(self.get_layer_by_name(self.model_name))
            self.bns.append(nn.BatchNorm1d(self.out_channels))

        if virtual_node:
            self.rnn = nn.LSTMCell(self.out_channels, self.out_channels)
        else:
            gate_nn = nn.Linear(self.out_channels, 1)
            self.pool = GlobalAttention(gate_nn)

        self.fc_out = nn.Linear(self.out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = self.node_encoder(data.x)
        v_idx = Model.get_virtual_node_idx(x, data) if self.virtual_node else None

        if self.model_name in ['PlainGAT', 'UniMP']:
            edge_index = torch.cat((data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index),
                                   dim=1)
            edge_attr = torch.cat((data.intra_level_edge_features, data.inter_level_edge_features,
                                   data.virtual_edge_features), dim=0)
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_index = [data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index]

            intra_level_edge_emb = self.edge_encoder(data.intra_level_edge_features)
            inter_level_edge_emb = self.edge_encoder(data.inter_level_edge_features) if \
                data.inter_level_edge_features.shape[0] != 0 else data.inter_level_edge_features
            virtual_edge_emb = self.edge_encoder(
                data.virtual_edge_features) if self.virtual_node else data.virtual_edge_features

            edge_emb = [intra_level_edge_emb, inter_level_edge_emb, virtual_edge_emb]

        for i in range(self.n_layers):
            identity = x
            x = self.convs[i](x, edge_index, edge_emb)
            x = self.bns[i](x)
            x = self.leakyrelu(x)

            if self.do_dropout:
                x = self.dropout(x)

            if i == 0:
                hx = identity[v_idx]
                cx = torch.zeros_like(identity[v_idx])
            if self.virtual_node:
                hx, cx = self.rnn(x[v_idx], (hx, cx))

            x += identity

        if self.virtual_node:
            pool_out = hx
        else:
            pool_out = self.pool(x, data.batch)

        out = self.fc_out(pool_out)
        return self.sigmoid(out)

    def get_layer_by_name(self, model_name):

        if model_name == 'PlainGAT':
            conv_layer = PlainGAT.GATConvWithEdgeAttr(self.out_channels, self.out_channels, self.heads)
        elif model_name == 'UniMP':
            conv_layer = TransformerConv(self.out_channels, self.out_channels, self.heads, edge_dim=self.out_channels,
                                         concat=False)
        elif model_name == 'RelationalGAT':
            conv_layer = RelationalGAT.RelationalGATConv(self.out_channels, self.out_channels, self.heads,
                                                         self.virtual_node)
        else:
            raise NotImplementedError
        return conv_layer

    @staticmethod
    def get_virtual_node_idx(x, data):
        idx = torch.tensor(data.__slices__['x'][1:]) - 1
        if x.is_cuda:
            idx = idx.cuda()
        return idx
