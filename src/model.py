# -*- coding: utf-8 -*-

"""
Created on 2021/4/21

@author: Siqi Miao
"""

import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn import GlobalAttention, InstanceNorm, LayerNorm, GraphNorm
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool.topk_pool import TopKPooling, filter_adj, topk

from layers import PlainGAT, RelationalGAT, DeepGCN, DeepGAT


class Model(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, model_config, data_config):
        super(Model, self).__init__()
        self.x_dim = x_dim
        self.heads = model_config['heads']
        self.out_channels = model_config['out_channels']
        self.n_layers = model_config['n_layers']
        self.model_name = model_config['model_name']
        self.dropout_p = model_config['dropout_p']
        self.readout = model_config['readout']
        self.norm_type = model_config['norm_type']

        self.att_sup = model_config['att_sup']
        self.att_unsup = model_config['att_unsup']
        self.topk_pooling = model_config['topk_pooling']
        self.deepgcn_aggr = model_config['deepgcn_aggr']
        self.share = model_config['share']

        self.run_type = data_config['run_type']
        self.virtual_node = data_config['virtual_node']

        self.convs = nn.ModuleList()
        self.node_atts = nn.ModuleList()
        self.leakyrelu = nn.LeakyReLU()

        if self.norm_type == 'batch':
            norm = nn.BatchNorm1d
        elif self.norm_type == 'layer':
            norm = LayerNorm
        elif self.norm_type == 'instance':
            norm = InstanceNorm
        elif self.norm_type == 'graph':
            norm = GraphNorm
        else:
            raise NotImplementedError

        self.node_encoder = nn.Linear(x_dim, self.out_channels)
        self.edge_encoder = nn.Linear(edge_attr_dim, self.out_channels)

        channels = [self.out_channels, self.out_channels*2, self.out_channels]
        if self.share:
            self.mlps = MLP(channels, norm=norm, dropout=self.dropout_p)
        else:
            self.mlps = nn.ModuleList()

        for i in range(self.n_layers):
            self.convs.append(self.get_layer_by_name(self.model_name))

            if (self.att_sup or self.att_unsup) and i % 3 == 0 and i < self.n_layers - 1:
                self.node_atts.append(AttSup(self.out_channels, ratio=0.8))

            if not self.share:
                self.mlps.append(MLP(channels, norm=norm, dropout=self.dropout_p))

        if self.virtual_node:
            if self.readout == 'lstm':
                self.lstm = nn.LSTMCell(self.out_channels, self.out_channels)
            elif self.readout == 'gru':
                self.gru = nn.GRUCell(self.out_channels, self.out_channels)
            elif self.readout == 'jknet':
                self.downsample = nn.Linear(self.out_channels * self.n_layers, self.out_channels)
            elif self.readout == 'virtual-node':
                pass
            else:
                raise NotImplementedError
        else:
            gate_nn = nn.Linear(self.out_channels, 1)
            self.pool = GlobalAttention(gate_nn)

        if self.run_type == 'regress':
            self.fc_out = nn.Linear(self.out_channels, 3)
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
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                assert hasattr(data, 'edge_attr') and data.edge_attr is not None
                edge_index = data.edge_index
                edge_attr = data.edge_attr
            else:
                edge_index = torch.cat((data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index), dim=1)
                edge_attr = torch.cat((data.intra_level_edge_features, data.inter_level_edge_features, data.virtual_edge_features), dim=0)
            edge_emb = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            identity = x

            x = self.convs[i](x, edge_index, edge_emb)
            x = self.mlps(x, data.batch) if self.share else self.mlps[i](x, data.batch)

            if self.virtual_node:
                if i == 0:
                    hx = identity[v_idx]
                    cx = torch.zeros_like(identity[v_idx])
                if self.readout == 'lstm':
                    hx, cx = self.lstm(x[v_idx], (hx, cx))
                elif self.readout == 'gru':
                    hx = self.gru(x[v_idx], hx)
                elif self.readout == 'jknet':
                    v_emb.append(x[v_idx])

            if (self.att_sup or self.att_unsup) and i % 3 == 0 and i < self.n_layers - 1:
                assert (not self.att_sup) == self.att_unsup
                x, edge_index, edge_emb, batch, perm, v_idx, score = self.node_atts[i//3](x, edge_index, edge_emb, data.batch,
                                                                                          v_idx, data.num_nodes, self.topk_pooling)
                if self.att_sup and data.score_gt.shape[0] != 0:
                    score_pair = [score, data.score_gt, data.batch]
                if self.topk_pooling:
                    identity = identity[perm]
                data.batch = batch

            x += identity

        if self.virtual_node:
            if self.readout in ['lstm', 'gru']:
                pool_out = hx
            elif self.readout == 'jknet':
                pool_out = self.downsample(torch.cat(v_emb, dim=1))
            elif self.readout == 'virtual-node':
                pool_out = x[v_idx]
        else:
            pool_out = self.pool(x, data.batch)

        out = self.fc_out(pool_out)

        if self.run_type == 'regress':
            pt_pair = [out, data.y]
            out = torch.zeros_like(out, dtype=torch.float)
        else:
            out = self.sigmoid(out)
        return out, score_pair, pt_pair

    def get_layer_by_name(self, model_name):

        if model_name == 'PlainGAT':
            conv_layer = PlainGAT.GATConvWithEdgeAttr(self.out_channels, self.out_channels, self.heads)
        elif model_name == 'UniMP':
            conv_layer = TransformerConv(self.out_channels, self.out_channels, self.heads, edge_dim=self.out_channels, concat=False)
        elif model_name == 'RelationalGAT':
            conv_layer = RelationalGAT.RelationalGATConv(self.out_channels, self.out_channels, self.heads, self.virtual_node)
        elif model_name == 'DeepGCN':
            conv_layer = DeepGCN.GENConv(self.out_channels, self.out_channels, aggr=self.deepgcn_aggr, learn_t=True, learn_p=True)
        elif model_name == 'DeepGAT':
            conv_layer = DeepGAT.DeepGATConv(self.out_channels, self.out_channels)
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


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (GraphNorm, InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, norm, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(norm(channels[i]))
                m.append(nn.LeakyReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


class AttSup(TopKPooling):

    def __init__(self, in_channels, ratio):
        super().__init__(in_channels, ratio=ratio)
        self.alpha = nn.Parameter(torch.Tensor(1))
        nn.init.ones_(self.alpha)

    def forward(self, x, edge_index, edge_emb, batch, v_idx, num_nodes, topk_pooling):

        # calculate scores for non-virtual nodes
        score = (x * self.weight).sum(dim=-1)
        score = softmax(score, batch)
        x = x * score.view(-1, 1) * self.alpha

        perm = None
        if topk_pooling:
            # select non-virtual nodes and concat virtual nodes
            perm = topk(score, self.ratio, batch)
            # make sure that all virtual nodes are added
            perm = torch.unique(torch.cat((perm, v_idx)), sorted=True)
            # find the indices of v_idx in perm and return them as the new v_idx
            v_idx = (perm.unsqueeze(1) == v_idx).nonzero(as_tuple=False)[:, 0]

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
