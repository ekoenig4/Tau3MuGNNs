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
    def __init__(self, model, batch_size, epochs, lr):
        super(GraphExplainer, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

    def explain_graph(self, data):

        self.model.eval()
        self.node_mask_logit = None
        self.edge_mask = None

        # Get the initial prediction.
        with torch.no_grad():
            original_prob, _, _ = self.model(data)
            target_label = (original_prob < 0.5).type(torch.float)

        data.edge_index = torch.cat((data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index), dim=1)
        data.edge_attr = torch.cat((data.intra_level_edge_features, data.inter_level_edge_features, data.virtual_edge_features), dim=0)

        self.node_mask_logit = torch.nn.Parameter(torch.Tensor(1, data.x.shape[0] - 1))
        torch.nn.init.kaiming_normal_(self.node_mask_logit)
        self.to(data.x.device)

        optimizer = torch.optim.Adam([self.node_mask_logit], lr=self.lr)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            node_mask_p = self.node_mask_logit.sigmoid()
            hard_mask = F.gumbel_softmax(torch.cat((node_mask_p, 1 - node_mask_p), dim=0).log(), hard=True, dim=0)[0]
            hard_mask = torch.cat((hard_mask, torch.ones(1, device=data.x.device)))

            src_lifted_hard_mask = hard_mask[data.edge_index[0]]
            dst_lifted_hard_mask = hard_mask[data.edge_index[1]]
            self.edge_mask = src_lifted_hard_mask * dst_lifted_hard_mask

            for module in self.model.modules():
                if isinstance(module, MessagePassing):
                    module.__explain__ = True
                    module.__edge_mask__ = self.edge_mask

            prob, _, _ = self.model(data)
            loss = F.binary_cross_entropy(prob, target_label)
            loss.backward()
            optimizer.step()
        final_prob = prob.item()

        node_mask = hard_mask.data
        self.node_mask_logit = None
        self.edge_mask = None
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None

        return node_mask, original_prob.item(), final_prob

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
        # values = [averaged_df(df, key) for df in dfs]
        values = [v_per_muon(df, key) for df in dfs]

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
    pos_test_sample_idx = pos_test_sample_idx[:1000]

    # all_xypos = graph_visz.get_xypos(m.dataset.all_entries)
    # labels = ['mu_hit_sim_theta', 'mu_hit_sim_phi', 'mu_hit_sim_eta', 'mu_hit_sim_r', 'mu_hit_sim_z']

    m.batch_size = 1560
    explainer = GraphExplainer(m.model, m.batch_size, 10, 1)

    data_loader = DataLoader(m.dataset[pos_test_sample_idx], batch_size=1, shuffle=False)
    pbar = tqdm(data_loader)

    pos_hit_ids = []
    o_prob = []
    f_prob = []
    n_masked = []
    good_attack = []
    for i, data in enumerate(pbar):
        if i in (0, 1):
            continue

        # xypos = all_xypos[pos_test_sample_idx[i]]
        # entry = m.dataset.all_entries.loc[pos_test_sample_idx[i]]
        data = data.to(m.device)
        node_mask, original_prob, final_prob = explainer.explain_graph(data)
        n_masked_nodes = (node_mask == 0).sum()
        ratio = n_masked_nodes / len(node_mask)

        pbar.set_description(f"original: {original_prob:.2f}, final: {final_prob:.2f}, # masked nodes: {n_masked_nodes:.2f}, ratio: {ratio:.2f}")
        if i == len(data_loader) - 1:
            pbar.set_description(f"original_avg: {np.mean(o_prob):.2f}, final_avg: {np.mean(f_prob):.2f}, # masked nodes_avg: {np.mean(n_masked):.2f}")

        if to_neg:
            if final_prob > 0.5 or original_prob < 0.5:
                continue
        else:
            if final_prob < 0.5 or original_prob > 0.5:
                continue

        # graph_visz.plot_graph(xypos, data, node_mask, entry, labels, original_prob, final_prob, to_neg)
        good_attack.append(pos_test_sample_idx[i])
        n_masked.append(n_masked_nodes)
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
