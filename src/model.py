# -*- coding: utf-8 -*-

"""
Created on 2021/4/21

@author: Siqi Miao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_sum
from torch_geometric.utils import softmax
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool.topk_pool import TopKPooling, filter_adj

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
        self.readout = config['readout']
        self.att_sup = config['att_sup']
        self.att_sup_beta = config['att_sup_beta']
        self.att_unsup = config['att_unsup']
        self.virtual_node = virtual_node
        self.att_sup_layer = int(2/3 * self.n_layers)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.node_encoder = nn.Linear(x_dim, self.out_channels)
        self.edge_encoder = nn.Linear(edge_attr_dim, self.out_channels)

        for _ in range(self.n_layers):
            self.convs.append(self.get_layer_by_name(self.model_name))
            self.bns.append(nn.BatchNorm1d(self.out_channels))

        if self.att_sup or self.att_unsup:
            self.top_att = AttSup(self.out_channels, min_score=config['att_sup_min_score'])

        if virtual_node:
            if self.readout == 'rnn':
                self.rnn = nn.LSTMCell(self.out_channels, self.out_channels)
            elif self.readout == 'jknet':
                self.downsample = nn.Linear(self.out_channels * self.n_layers, self.out_channels)
            else:
                raise NotImplementedError
        else:
            gate_nn = nn.Linear(self.out_channels, 1)
            self.pool = GlobalAttention(gate_nn)

        self.fc_out = nn.Linear(self.out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = self.node_encoder(data.x)
        v_idx, v_emb = (Model.get_virtual_node_idx(x, data), []) if self.virtual_node else (None, None)
        kl_loss = None

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

            if self.virtual_node:
                if self.readout == 'rnn':
                    if i == 0:
                        hx = identity[v_idx]
                        cx = torch.zeros_like(identity[v_idx])
                    hx, cx = self.rnn(x[v_idx], (hx, cx))
                elif self.readout == 'jknet':
                    v_emb.append(x[v_idx])

            if i == self.att_sup_layer and (self.att_sup or self.att_unsup):
                assert (not self.att_sup) == self.att_unsup
                x, edge_index, edge_emb, batch, perm, v_idx, score = self.top_att(x, edge_index, edge_emb,
                                                                                  data.batch, v_idx, data.num_nodes)
                if self.att_sup and data.score_gt.shape[0] != 0:
                    kl_loss = self.att_sup_beta * scatter_sum(F.kl_div(torch.log(score + 1e-14), data.score_gt,
                                                                       reduction='none'), data.batch).mean()
                identity = identity[perm]
                data.batch = batch

            x += identity

        if self.virtual_node:
            if self.readout == 'rnn':
                pool_out = hx
            elif self.readout == 'jknet':
                pool_out = self.downsample(torch.cat(v_emb, dim=1))
        else:
            pool_out = self.pool(x, data.batch)

        out = self.fc_out(pool_out)
        return self.sigmoid(out), kl_loss

    def get_layer_by_name(self, model_name):

        if model_name == 'PlainGAT':
            conv_layer = PlainGAT.GATConvWithEdgeAttr(self.out_channels, self.out_channels, self.heads)
        elif model_name == 'UniMP':
            conv_layer = TransformerConv(self.out_channels, self.out_channels, self.heads, edge_dim=self.out_channels,
                                         concat=False)
        elif model_name == 'RelationalGAT':
            conv_layer = RelationalGAT.RelationalGATConv(self.out_channels, self.out_channels, self.heads,
                                                         self.virtual_node, self.do_dropout)
        else:
            raise NotImplementedError
        return conv_layer

    @staticmethod
    def get_virtual_node_idx(x, data):
        idx = torch.tensor(data.__slices__['x'][1:]) - 1
        if x.is_cuda:
            idx = idx.cuda()
        return idx


class AttSup(TopKPooling):
    def forward(self, x, edge_index, edge_emb, batch, v_idx, num_nodes):

        score = (x * self.weight).sum(dim=-1)
        score = softmax(score, batch)

        perm, v_idx = AttSup.topk(score, batch, self.min_score, v_idx)
        x = x[perm] * score[perm].view(-1, 1)
        batch = batch[perm]

        if isinstance(edge_index, list):
            for level in range(3):
                if edge_index[level].shape[0] != 0:
                    edge_index[level], edge_emb[level] = filter_adj(edge_index[level], edge_emb[level], perm, num_nodes)
        else:
            edge_index, edge_emb = filter_adj(edge_index, edge_emb, perm, num_nodes)

        return x, edge_index, edge_emb, batch, perm, v_idx, score

    @staticmethod
    def topk(x, batch, min_score, v_idx, tol=1e-7):
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)

        # make sure that all virtual nodes are added
        perm = torch.unique(torch.cat((perm, v_idx)), sorted=True)
        # find the indices of v_idx in perm and return them as the new v_idx
        v_idx = (perm.unsqueeze(1) == v_idx).nonzero(as_tuple=False)[:, 0]
        return perm, v_idx
