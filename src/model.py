# -*- coding: utf-8 -*-

"""
Created on 2021/4/21

@author: Siqi Miao
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_max
from torch_geometric.utils import softmax
from torch_geometric.nn import GlobalAttention, InstanceNorm, LayerNorm
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool.topk_pool import TopKPooling, filter_adj

from layers import PlainGAT, RelationalGAT, DeepGCN


class Model(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, model_config, data_config):
        super(Model, self).__init__()
        self.x_dim = x_dim
        self.heads = model_config['heads']
        self.out_channels = model_config['out_channels']
        self.n_layers = model_config['n_layers']
        self.model_name = model_config['model_name']
        self.do_dropout = model_config['dropout']
        self.readout = model_config['readout']
        self.norm_type = model_config['norm_type']

        self.att_sup = model_config['att_sup']
        self.att_unsup = model_config['att_unsup']
        self.topk_pooling = model_config['topk_pooling']

        self.pred_pt = data_config['pred_pt']
        self.virtual_node = data_config['virtual_node']
        self.att_sup_layer = int(0.2 * self.n_layers)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.node_atts = nn.ModuleList()

        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.node_encoder = nn.Linear(x_dim, self.out_channels)
        self.edge_encoder = nn.Linear(edge_attr_dim, self.out_channels)

        for _ in range(self.n_layers):
            self.convs.append(self.get_layer_by_name(self.model_name))

            if self.norm_type == 'BN':
                norm = nn.BatchNorm1d(self.out_channels)
            elif self.norm_type == 'IN':
                norm = InstanceNorm(self.out_channels)
            elif self.norm_type == 'LN':
                norm = LayerNorm(self.out_channels)
            else:
                raise NotImplementedError
            self.norms.append(norm)

            if self.att_sup or self.att_unsup:
                self.node_atts.append(AttSup(self.out_channels, min_score=model_config['att_sup_min_score']))

        if self.virtual_node:
            if self.readout == 'rnn':
                self.rnn = nn.LSTMCell(self.out_channels, self.out_channels)
            elif self.readout == 'jknet':
                self.downsample = nn.Linear(self.out_channels * self.n_layers, self.out_channels)
            else:
                raise NotImplementedError
        else:
            gate_nn = nn.Linear(self.out_channels, 1)
            self.pool = GlobalAttention(gate_nn)

        if self.pred_pt:
            self.fc_out = nn.Linear(self.out_channels, 4)
        else:
            self.fc_out = nn.Linear(self.out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = self.node_encoder(data.x)
        v_idx, v_emb = (Model.get_virtual_node_idx(x, data), []) if self.virtual_node else (None, None)
        score_pair, pt_pair = None, None

        if self.model_name == 'RelationalGAT':
            edge_index = [data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index]

            intra_level_edge_emb = self.edge_encoder(data.intra_level_edge_features)
            inter_level_edge_emb = self.edge_encoder(data.inter_level_edge_features) if \
                data.inter_level_edge_features.shape[0] != 0 else data.inter_level_edge_features
            virtual_edge_emb = self.edge_encoder(
                data.virtual_edge_features) if self.virtual_node else data.virtual_edge_features

            edge_emb = [intra_level_edge_emb, inter_level_edge_emb, virtual_edge_emb]
        else:
            edge_index = torch.cat((data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index),
                                   dim=1)
            edge_attr = torch.cat((data.intra_level_edge_features, data.inter_level_edge_features,
                                   data.virtual_edge_features), dim=0)
            edge_emb = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            identity = x

            x = self.convs[i](x, edge_index, edge_emb)

            if self.virtual_node:
                if self.readout == 'rnn':
                    if i == 0:
                        hx = identity[v_idx]
                        cx = torch.zeros_like(identity[v_idx])
                    hx, cx = self.rnn(x[v_idx], (hx, cx))
                elif self.readout == 'jknet':
                    v_emb.append(x[v_idx])

            if i >= self.att_sup_layer and (self.att_sup or self.att_unsup):
                assert (not self.att_sup) == self.att_unsup
                x, edge_index, edge_emb, batch, perm, v_idx, score = self.node_atts[i](x, edge_index, edge_emb, data.batch,
                                                                                       v_idx, data.num_nodes, self.topk_pooling)
                if self.att_sup and data.score_gt.shape[0] != 0:
                    score_pair = [score, data.score_gt, data.batch]
                if self.topk_pooling:
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

        if self.pred_pt:
            pt_pair = [out[:, 1:], data.y[:, 1:]]
            out = self.sigmoid(out[:, [0]])
        else:
            out = self.sigmoid(out)
        return out, score_pair, pt_pair

    def get_layer_by_name(self, model_name):

        if model_name == 'PlainGAT':
            conv_layer = PlainGAT.GATConvWithEdgeAttr(self.out_channels, self.out_channels, self.heads)
        elif model_name == 'UniMP':
            conv_layer = TransformerConv(self.out_channels, self.out_channels, self.heads, edge_dim=self.out_channels,
                                         concat=False)
        elif model_name == 'RelationalGAT':
            conv_layer = RelationalGAT.RelationalGATConv(self.out_channels, self.out_channels, self.heads,
                                                         self.virtual_node, self.do_dropout)
        elif model_name == 'DeepGCN':
            conv_layer = DeepGCN.GENConv(self.out_channels, self.out_channels, aggr='softmax',
                                         learn_t=True, learn_p=True)
        else:
            raise NotImplementedError
        return conv_layer

    @staticmethod
    def get_virtual_node_idx(x, data):

        sl = getattr(data, '__slices__', None)
        if sl is not None:
            idx = torch.tensor(data.__slices__['x'][1:]) - 1
        else:
            idx = torch.tensor([x.shape[0] - 1])

        if x.is_cuda:
            idx = idx.cuda()
        return idx


class AttSup(TopKPooling):

    def __init__(self, in_channels, min_score):
        super().__init__(in_channels, min_score=min_score)
        self.alpha = nn.Parameter(torch.Tensor(1))
        nn.init.ones_(self.alpha)

    def forward(self, x, edge_index, edge_emb, batch, v_idx, num_nodes, topk_pooling, mask_virtual=False):
        node_mask = torch.ones_like(batch, dtype=torch.bool)
        if mask_virtual:
            node_mask[v_idx] = 0

        # calculate scores for non-virtual nodes
        score = (x[node_mask] * self.weight).sum(dim=-1)
        score = softmax(score, batch[node_mask])
        x[node_mask] *= score.view(-1, 1) * self.alpha

        perm = None
        if topk_pooling:
            # select non-virtual nodes and concat virtual nodes
            perm, v_idx = AttSup.topk(score, batch[node_mask], self.min_score, v_idx)

            # filter unimportant nodes
            batch = batch[perm]
            x = x[perm]

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
        perm = torch.sort(torch.cat((perm, v_idx)))[0]
        # find the indices of v_idx in perm and return them as the new v_idx
        v_idx = (perm.unsqueeze(1) == v_idx).nonzero(as_tuple=False)[:, 0]
        return perm, v_idx
