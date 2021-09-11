import sys
sys.path.insert(0, '..')

import yaml
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm
from main import Main
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import combinations
from torch_scatter import scatter_sum

from utils import root2df
from layers.DeepGAT import DeepGATConv
from layers.DeepGCN import GENConv

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.nn import GNNExplainer, MessagePassing
from explainer import graph_visz
import torch.nn as nn
from torch_geometric.utils import softmax


class MLP(nn.Sequential):
    def __init__(self, channels, norm, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(norm(channels[i]))
                m.append(nn.LeakyReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


class Net(torch.nn.Module):
    def __init__(self, in_node, in_edge, hz, num_classes, dropout_p, n_layers):
        super(Net, self).__init__()
        self.n_layers = n_layers

        self.node_map = nn.Linear(in_node, hz)
        self.edge_map = nn.Linear(in_edge, hz)

        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(DeepGATConv(hz, hz))
            # self.convs.append(GENConv(hz, hz, learn_t=True))
            self.mlps.append(MLP([hz, hz*2, hz], norm=nn.BatchNorm1d, dropout=dropout_p))

        self.fc_out = nn.Linear(hz, num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_map(x)
        edge_attr = self.edge_map(edge_attr)

        for i in range(self.n_layers):
            identity = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.mlps[i](x)
            x += identity

        return self.fc_out(x)


class GraphExplainer(torch.nn.Module):
    def __init__(self, model, lr, thres, lamb):
        super(GraphExplainer, self).__init__()
        self.model = model
        self.lr = lr
        self.lamb = lamb
        self.thres = thres

        self.explainer_gnn = Net(in_node=11+1, in_edge=5, hz=64, num_classes=2, dropout_p=0.2, n_layers=2).cuda()
        self.optimizer = torch.optim.AdamW(self.explainer_gnn.parameters(), lr=self.lr)
        num_params = sum(p.numel() for p in self.explainer_gnn.parameters() if p.requires_grad)
        print('# params:', num_params)

    def explain_graph(self, data, epoch):
        self.explainer_gnn.train()
        self.model.eval()

        # Get the initial prediction.
        with torch.no_grad():
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            self.model.mlps[-1].register_forward_hook(get_activation('node_emb'))
            self.model.edge_encoder.register_forward_hook(get_activation('edge_emb'))

            original_prob, _, _ = self.model(data)
            target_labels = torch.zeros_like(original_prob).float()

        edge_index = torch.cat((data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index), dim=1)
        edge_attr = torch.cat((data.intra_level_edge_features, data.inter_level_edge_features, data.virtual_edge_features), dim=0)
        # x = torch.cat((data.x, activation['node_emb']), dim=1)
        x = data.x
        v_idx = GraphExplainer.get_virtual_node_idx(x, data)
        indicator = torch.zeros((x.shape[0], 1), device=x.device)
        indicator[v_idx] = 1
        x = torch.cat((x, indicator), dim=1)

        # edge_attr = torch.cat((edge_attr, activation['edge_emb']), dim=1)
        edge_attr = edge_attr

        self.optimizer.zero_grad()
        node_mask_logit = self.explainer_gnn(x, edge_index, edge_attr)

        do_hard = True
        hard_masks = F.gumbel_softmax(node_mask_logit, tau=0.1, hard=do_hard, dim=1)[:, 1]

        src_lifted_hard_masks = hard_masks[edge_index[0]]
        dst_lifted_hard_masks = hard_masks[edge_index[1]]
        edge_mask = src_lifted_hard_masks * dst_lifted_hard_masks

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = edge_mask

        prob, _, _ = self.model(data)
        bce_loss = F.binary_cross_entropy(prob, target_labels)
        masked_prob = node_mask_logit.softmax(dim=1)[:, 0]
        reg_loss = self.lamb * torch.mean(scatter_sum(abs(masked_prob), data.batch) / scatter_sum(torch.ones_like(data.batch), data.batch))
        no_v_loss = 10 * torch.norm(node_mask_logit.softmax(dim=1)[v_idx, 0], p=1) / len(v_idx)

        loss = bce_loss + reg_loss + no_v_loss
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.explainer_gnn.eval()
            node_mask_logit = self.explainer_gnn(x, edge_index, edge_attr)
            node_mask = (node_mask_logit.softmax(1)[:, 1] > self.thres).int()
            src_lifted_hard_masks = node_mask[edge_index[0]]
            dst_lifted_hard_masks = node_mask[edge_index[1]]
            edge_mask = src_lifted_hard_masks * dst_lifted_hard_masks
            for module in self.model.modules():
                if isinstance(module, MessagePassing):
                    module.__explain__ = True
                    module.__edge_mask__ = edge_mask
            final_prob, _, _ = self.model(data)

            fine_idx = torch.where(original_prob > 0.5)[0]
            final_bce_loss = F.binary_cross_entropy(final_prob[fine_idx], target_labels[fine_idx])
            good_rate = ((final_prob[fine_idx] < 0.5).sum() / len(final_prob[fine_idx]))
            num_masked_nodes = np.mean([(node_mask[data.batch == each] == 0).sum().item() for each in fine_idx])
            num_remained_nodes = np.mean([(node_mask[data.batch == each] == 1).sum().item() for each in fine_idx])
            f_prob = torch.mean(final_prob[fine_idx])
            o_prob = torch.mean(original_prob[fine_idx])

        desc = f'[E {epoch}], bce: {bce_loss:.3f}, reg: {reg_loss:.3f}, no_v: {no_v_loss:.3f}, ' \
               f'total: {loss:.3f}, final_bce: {final_bce_loss:.3f}, ' \
               f'o_prob: {o_prob:.3f}, f_prob: {f_prob:.3f}, ' \
               f'good_rate: {good_rate:.3f}, #masked: {num_masked_nodes:.3f}, #remained: {num_remained_nodes:.3f}'

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

        return desc, bce_loss.item(), reg_loss.item(), no_v_loss.item(), loss.item(), final_bce_loss.item(), o_prob.item(), f_prob.item(), \
               good_rate.item(), num_masked_nodes.item(), num_remained_nodes.item(), node_mask.cpu().numpy()

    @torch.no_grad()
    def eval_exp(self, data):
        self.model.eval()
        self.explainer_gnn.eval()
        original_prob, _, _ = self.model(data)

        edge_index = torch.cat((data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index), dim=1)
        edge_attr = torch.cat((data.intra_level_edge_features, data.inter_level_edge_features, data.virtual_edge_features), dim=0)

        x = data.x
        v_idx = GraphExplainer.get_virtual_node_idx(x, data)
        indicator = torch.zeros((x.shape[0], 1), device=x.device)
        indicator[v_idx] = 1
        x = torch.cat((x, indicator), dim=1)

        node_mask_logit = self.explainer_gnn(x, edge_index, edge_attr)
        node_mask = (node_mask_logit.softmax(1)[:, 1] > self.thres).int()
        src_lifted_hard_masks = node_mask[edge_index[0]]
        dst_lifted_hard_masks = node_mask[edge_index[1]]
        edge_mask = src_lifted_hard_masks * dst_lifted_hard_masks
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = edge_mask
        final_prob, _, _ = self.model(data)

        fine_idx = torch.where(original_prob > 0.5)[0]
        good_rate = ((final_prob[fine_idx] < 0.5).sum() / len(final_prob[fine_idx]))
        num_masked_nodes = np.mean([(node_mask[data.batch == each] == 0).sum().item() for each in fine_idx])
        num_remained_nodes = np.mean([(node_mask[data.batch == each] == 1).sum().item() for each in fine_idx])

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

        return original_prob.cpu().numpy(), final_prob.cpu().numpy(), node_mask.cpu().numpy(), good_rate.item(), num_masked_nodes.item(), num_remained_nodes.item()

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


def train_exp(data_loader, epochs, device, explainer):
    for epoch in range(epochs):
        pbar = tqdm(data_loader)
        bce_loss_all, reg_loss_all, no_v_loss_all, total_loss_all, final_bce_loss_all, good_rate_all, num_masked_nodes_all, num_remained_nodes_all = [], [], [], [], [], [], [], []
        f_prob_all, o_prob_all = [], []

        for i, data in enumerate(pbar):

            data = data.to(device)
            desc, bce_loss, reg_loss, no_v_loss, total_loss, final_bce_loss, o_prob, f_prob, good_rate, num_masked_nodes, num_remained_nodes, _ = explainer.explain_graph(data, epoch)
            bce_loss_all.append(bce_loss), reg_loss_all.append(reg_loss), no_v_loss_all.append(no_v_loss), total_loss_all.append(total_loss), \
            num_remained_nodes_all.append(num_remained_nodes), num_masked_nodes_all.append(num_masked_nodes)
            final_bce_loss_all.append(final_bce_loss), good_rate_all.append(good_rate)
            f_prob_all.append(f_prob), o_prob_all.append(o_prob)

            pbar.set_description(desc)
            if i == len(pbar) - 1:
                desc = f'[E {epoch}], bce: {np.mean(bce_loss_all):.3f}, reg: {np.mean(reg_loss_all):.3f}, ' \
                       f'no_v: {np.mean(no_v_loss_all):.3f}, total: {np.mean(total_loss_all):.3f}, ' \
                       f'final_bce: {np.mean(final_bce_loss_all):.3f}, ' \
                       f'o_prob: {np.mean(o_prob_all):.3f}, f_prob: {np.mean(f_prob_all):.3f}, '\
                       f'good_rate: {np.mean(good_rate_all):.3f}, ' \
                       f'#masked: {np.mean(num_masked_nodes_all):.3f}, #remained: {np.mean(num_remained_nodes_all):.3f}'
                pbar.set_description(desc)


def visz_exp(data_loader, explainer, device, all_xypos, labels, all_entries, dataset):
    pbar = tqdm(data_loader)
    for i, data in enumerate(pbar):
        data = data.to(device)
        original_prob, final_prob, node_mask, good_rate, num_masked_nodes, num_remained_nodes = explainer.eval_exp(data)

        batch = data.batch.cpu().numpy()
        for each_g in range(np.max(batch) + 1):
            pd_idx = data.pd_idx[each_g].cpu().int().item()
            xypos = all_xypos[pd_idx]
            entry = all_entries.loc[pd_idx]
            g = next(iter(DataLoader(dataset[[pd_idx]], batch_size=1)))

            important_node_ids = node_mask[batch == each_g]
            important_node_ids = np.argwhere(important_node_ids == 0).reshape(-1).tolist()
            graph_visz.plot_graph(xypos, g, important_node_ids, entry, labels, original_prob[each_g].item(), final_prob[each_g].item())


def main():

    cfg = Path('./configs') / 'config0_radius.yml'
    model_path = Path('./explain') / 'reparam.pt'
    eval_exp = True
    num_samples = 20000
    epochs = 500

    cfg_dict = yaml.safe_load(cfg.open('r'))
    m = Main(cfg_dict, '_'.join(cfg.stem.split('_')[1:]))
    m.load_checkpoint()

    pos_test_sample_idx = graph_visz.get_proper_idx(m.dataset.all_entries, m.dataset.idx_split)
    np.random.seed(1)
    np.random.shuffle(pos_test_sample_idx)
    pos_test_sample_idx = pos_test_sample_idx[:num_samples]

    all_xypos = graph_visz.get_xypos(m.dataset.all_entries)
    labels = ['mu_hit_sim_theta', 'mu_hit_sim_phi', 'mu_hit_sim_eta', 'mu_hit_sim_r', 'mu_hit_sim_z']

    explainer = GraphExplainer(m.model, lr=5e-5, thres=0.5, lamb=1)

    if eval_exp:
        data_loader = DataLoader(m.dataset[pos_test_sample_idx], batch_size=256, shuffle=False)
        graph_visz.load_checkpoint(explainer.explainer_gnn, model_path)
        visz_exp(data_loader, explainer, m.device, all_xypos, labels, m.dataset.all_entries, m.dataset)
    else:
        data_loader = DataLoader(m.dataset[pos_test_sample_idx], batch_size=256, shuffle=True)
        train_exp(data_loader, epochs, m.device, explainer)
        graph_visz.save_checkpoint(explainer.explainer_gnn, model_path)


if __name__ == '__main__':
    main()
