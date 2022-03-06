import os

import yaml
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm
from train_gnn import Main
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from utils import root2df

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GNNExplainer, MessagePassing
from explainer import graph_visz


class GraphExplainer(GNNExplainer):

    def explain_graph(self, batched_data):

        self.model.eval()
        self.__clear_masks__()

        # all nodes belong to same graph
        assert batched_data.batch.sum() == 0

        # Get the initial prediction.
        with torch.no_grad():
            original_prob, _, _ = self.model(batched_data)
            target_label = (original_prob < 0.5).type(torch.float)

        x = batched_data.x
        edge_index = torch.cat((batched_data.intra_level_edge_index, batched_data.inter_level_edge_index, batched_data.virtual_edge_index), dim=1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Explain graph')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            prob, _, _ = self.model(batched_data)
            loss = self.__loss__(prob, target_label)
            loss.backward()
            optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)
        final_prob = prob.item()

        if self.log:  # pragma: no cover
            pbar.close()

        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()
        return edge_mask, original_prob.item(), final_prob

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __loss__(self, prob, target):
        loss = F.binary_cross_entropy(prob, target)
        return loss

def get_important_hit_ids(virtual_node_id, edge_index, edge_mask):
    virtual_node_connected_edges = (edge_index[0] != virtual_node_id) * (edge_index[1] == virtual_node_id)
    scores = edge_mask[virtual_node_connected_edges].argsort()[-3:]
    return edge_index[0, virtual_node_connected_edges][scores].cpu().tolist()


def main():
    # os.chdir('../')
    cfg = Path('./configs') / 'config0_radius.yml'
    to_neg = False

    cfg_dict = yaml.safe_load(cfg.open('r'))
    m = Main(cfg_dict, '_'.join(cfg.stem.split('_')[1:]))
    m.load_checkpoint()

    pos_test_sample_idx = graph_visz.get_proper_idx(m.dataset.all_entries, m.dataset.idx_split, to_neg)
    np.random.seed(1)
    np.random.shuffle(pos_test_sample_idx)
    pos_test_sample_idx = pos_test_sample_idx[:1000]

    all_xypos = graph_visz.get_xypos(m.dataset.all_entries)
    labels = ['mu_hit_sim_theta', 'mu_hit_sim_phi', 'mu_hit_sim_eta', 'mu_hit_sim_r', 'mu_hit_sim_z']

    data_loader = DataLoader(m.dataset[pos_test_sample_idx], batch_size=1, shuffle=False)
    explainer = GraphExplainer(m.model, epochs=10, return_type='prob', lr=0.1, log=False)
    pbar = tqdm(data_loader)

    pos_hit_ids = []
    o_prob = []
    f_prob = []
    good_attack = []
    for i, data in enumerate(pbar):
        xypos = all_xypos[pos_test_sample_idx[i]]
        entry = m.dataset.all_entries.loc[pos_test_sample_idx[i]]

        data = data.to(m.device)
        v_id = data.x.shape[0] - 1
        edge_index = torch.cat((data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index), dim=1)

        edge_mask, original_prob, final_prob = explainer.explain_graph(data)
        important_node_ids = get_important_hit_ids(v_id, edge_index, edge_mask)

        if to_neg:
            if final_prob > 0.5 or original_prob < 0.5:
                continue
        else:
            if final_prob < 0.5 or original_prob > 0.5:
                continue

        graph_visz.plot_graph(xypos, data, important_node_ids, entry, labels, original_prob, final_prob, to_neg)
        good_attack.append(pos_test_sample_idx[i])
        pos_hit_ids.append(important_node_ids)
        o_prob.append(original_prob)
        f_prob.append(final_prob)
        pbar.set_description(f"original: {original_prob:.2f}, final: {final_prob:.2f}")

        if i == len(data_loader) - 1:
            pbar.set_description(f"original_avg: {np.mean(o_prob):.2f}, final_avg: {np.mean(f_prob):.2f}")

    pos_df = m.dataset.all_entries.loc[good_attack].reset_index(drop=True)

    dfs = root2df.Root2Df(data_dir=Path('./../data/raw')).read_df()
    neg200 = dfs['MinBiasPU200_MTD'].sample(n=100000).reset_index(drop=True)
    pos0 = dfs['DsTau3muPU0_MTD']
    pos0 = pos0[pos0.n_gen_tau == 1].reset_index(drop=True)
    pos0 = pos0.sample(n=100000).reset_index(drop=True)

    graph_visz.plot_important_hit_dist(pos_df, pos_hit_ids, neg200, pos0)


if __name__ == '__main__':
    main()
