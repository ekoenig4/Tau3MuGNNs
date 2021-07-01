import os

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

from utils import root2df

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GNNExplainer, MessagePassing
from explain import graph_visz


class GraphExplainer(torch.nn.Module):
    def __init__(self, model, batch_size, device):
        super(GraphExplainer, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def explain_graph(self, data, to_neg):

        self.model.eval()
        self.best_mask_idx = None
        self.best_prob = float('inf') if to_neg else float('-inf')

        # Get the initial prediction.
        original_prob, _, _ = self.model(data.to(self.device))

        x = data.x
        edge_index = torch.cat((data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index), dim=1)
        edge_attr = torch.cat((data.intra_level_edge_features, data.inter_level_edge_features, data.virtual_edge_features), dim=0)

        masked_nodes, mask_sizes = GraphExplainer.get_masked_nodes(x)
        data_list = GraphExplainer.make_data_list(x, edge_index, edge_attr, masked_nodes, self.device)
        data_loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)

        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            prob, _, _ = self.model(batch)
            if to_neg:
                if prob.min() < self.best_prob:
                    self.best_prob = prob.min()
                    self.best_mask_idx = batch.idx[prob.argmin()]
            else:
                if prob.max() > self.best_prob:
                    self.best_prob = prob.max()
                    self.best_mask_idx = batch.idx[prob.argmax()]

        return list(masked_nodes[self.best_mask_idx]), original_prob.item(), self.best_prob.item()

    @staticmethod
    def make_data_list(x, edge_index, edge_attr, masked_nodes, device):
        data_list = []
        for idx, mask in enumerate(masked_nodes):
            mask = torch.tensor(mask).view(-1, 1).to(device)
            edge_remained = ((edge_index[0] == mask).sum(dim=0) == 0) * ((edge_index[1] == mask).sum(dim=0) == 0)

            data_list.append(Data(x=x, edge_index=edge_index[:, edge_remained], edge_attr=edge_attr[edge_remained], idx=idx))
        return data_list

    @staticmethod
    def get_masked_nodes(x):
        all_3_comb = list(combinations(range(x.shape[0] - 1), 3))
        all_2_comb = list(combinations(range(x.shape[0] - 1), 2))
        all_1_comb = list(combinations(range(x.shape[0] - 1), 1))
        return all_2_comb, None
        # return all_1_comb + all_2_comb + all_3_comb, [len(all_1_comb), len(all_2_comb), len(all_3_comb)]


palette = 'bright'
def concat_array_to_df(arrays, names):
    dfs = []
    for idx, array in enumerate(arrays):
        name = [names[idx]] * len(array)
        dfs.append(pd.DataFrame(data={'value': array, 'name': name}))
    return pd.concat(dfs, ignore_index=True)


def plot_sim_var(dfs, names, agg=np.mean):
    print('[INFO] Plotting sim_var...')

    def hit_filter(x, key):
        idx = (x[key] != -99)
        if idx.sum() == 0:
            return np.nan
        else:
            return agg(x[key][idx])

    def averaged_df(df, key):
        filtered_df = df.apply(lambda x: hit_filter(x, key), axis=1)
        return filtered_df[filtered_df.notna()].to_numpy()

    def v_per_muon(df, key):
        v = df[key].tolist()
        return np.array([hit for event in v for hit in event if hit != -99])

    keys = ['mu_hit_ring', 'mu_hit_quality', 'mu_hit_bend', 'mu_hit_sim_phi', 'mu_hit_sim_theta', 'mu_hit_sim_eta',
            'mu_hit_sim_r', 'mu_hit_sim_z']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for idx, key in enumerate(tqdm(keys)):
        values = [averaged_df(df, key) for df in dfs]
        # values = [v_per_muon(df, key) for df in dfs]

        df = concat_array_to_df(values, names)
        cumulative = False
        title = key
        if key == 'mu_hit_bend':
            if agg == np.mean:
                axes[idx//4, idx%4].set_xlim(-50, 50)
        elif key in ['mu_hit_ring', 'mu_hit_quality']:
            cumulative = True
            title += '-cdf'

        axes[idx // 4, idx % 4].set_title(title)

        sns.histplot(data=df, x='value', hue='name', stat="density", fill=False, common_norm=False, discrete=False,
                     element='step', cumulative=cumulative, ax=axes[idx // 4, idx % 4],
                     palette=palette)
    plt.show()


def plot_important_hit_dist(pos_df: pd.DataFrame, pos_hit_ids, neg200, pos0):

    for idx, entry in pos_df.iterrows():
        for k, v in entry.items():
            if isinstance(v, np.ndarray) and v.shape[0] == entry['n_mu_hit']:
                pos_df.at[idx, k] = v[pos_hit_ids[idx]]

    # for idx, entry in pos0.iterrows():
    #     for k, v in entry.items():
    #         if isinstance(v, np.ndarray) and v.shape[0] == entry['n_mu_hit']:
    #             pos0.at[idx, k] = v[entry['mu_hit_station'] == 1]
    #
    # for idx, entry in neg200.iterrows():
    #     for k, v in entry.items():
    #         if isinstance(v, np.ndarray) and v.shape[0] == entry['n_mu_hit']:
    #             neg200.at[idx, k] = v[entry['mu_hit_station'] == 1]

    plot_sim_var([pos_df, neg200, pos0], ['pos200_filtered', 'neg200', 'pos0'], agg=np.mean)


def main():

    cfg = Path('./configs') / 'config0_no-neib.yml'
    to_neg = True

    cfg_dict = yaml.safe_load(cfg.open('r'))
    m = Main(cfg_dict, '_'.join(cfg.stem.split('_')[1:]))
    m.load_checkpoint()

    pos_test_sample_idx = graph_visz.get_proper_idx(m.dataset.all_entries, m.dataset.idx_split, to_neg)
    np.random.seed(1)
    np.random.shuffle(pos_test_sample_idx)
    pos_test_sample_idx = pos_test_sample_idx[:500]

    all_xypos = graph_visz.get_xypos(m.dataset.all_entries)
    labels = ['mu_hit_sim_theta', 'mu_hit_sim_phi', 'mu_hit_sim_eta', 'mu_hit_sim_r', 'mu_hit_sim_z']

    m.batch_size = 1560
    explainer = GraphExplainer(m.model, m.batch_size, m.device)

    data_loader = DataLoader(m.dataset[pos_test_sample_idx], batch_size=1, shuffle=False)
    pbar = tqdm(data_loader)

    pos_hit_ids = []
    o_prob = []
    f_prob = []
    n_masked = []
    good_attack = []
    for i, data in enumerate(pbar):
        xypos = all_xypos[pos_test_sample_idx[i]]
        entry = m.dataset.all_entries.loc[pos_test_sample_idx[i]]

        node_mask, original_prob, final_prob = explainer.explain_graph(data, to_neg)

        pbar.set_description(f"original: {original_prob:.2f}, final: {final_prob:.2f}, # masked nodes: {len(node_mask):.2f}")
        if i == len(data_loader) - 1:
            pbar.set_description(f"original_avg: {np.mean(o_prob):.2f}, final_avg: {np.mean(f_prob):.2f}, # masked nodes_avg: {np.mean(n_masked):.2f}")

        if to_neg:
            if final_prob > 0.5 or original_prob < 0.5:
                continue
        else:
            if final_prob < 0.5 or original_prob > 0.5:
                continue

        graph_visz.plot_graph(xypos, data, node_mask, entry, labels, original_prob, final_prob, to_neg)
        good_attack.append(pos_test_sample_idx[i])
        n_masked.append(len(node_mask))
        pos_hit_ids.append(node_mask)
        o_prob.append(original_prob)
        f_prob.append(final_prob)

    print('good rate:', len(good_attack) / len(pos_test_sample_idx))
    pos_df = m.dataset.all_entries.loc[good_attack].reset_index(drop=True)

    dfs = root2df.Root2Df(data_dir=Path('./../data/raw')).read_df()
    neg200 = dfs['MinBiasPU200_MTD'].sample(n=100000).reset_index(drop=True)
    pos0 = dfs['DsTau3muPU0_MTD']
    pos0 = pos0[pos0.n_gen_tau == 1].reset_index(drop=True)
    pos0 = pos0.sample(n=100000).reset_index(drop=True)

    plot_important_hit_dist(pos_df, pos_hit_ids, neg200, pos0)


if __name__ == '__main__':
    main()
