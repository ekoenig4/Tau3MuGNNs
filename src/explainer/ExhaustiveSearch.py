import os

import yaml
import numpy as np
from tqdm import tqdm
from train_gnn import Main
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import combinations

from utils import root2df

import torch
from torch_geometric.data import DataLoader, Data
from explainer import graph_visz


class GraphExplainer(torch.nn.Module):
    def __init__(self, model, batch_size, device):
        super(GraphExplainer, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def explain_graph(self, data, show_instance, xypos, entry, labels, circle=None):

        self.model.eval()

        # Get the initial prediction.
        original_prob, _, _ = self.model(data.to(self.device))
        original_prob = original_prob.item()

        x = data.x

        if data.edge_index is None:
            edge_index = torch.cat((data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index), dim=1)
            edge_attr = torch.cat((data.intra_level_edge_features, data.inter_level_edge_features, data.virtual_edge_features), dim=0)
        else:
            edge_index = data.edge_index
            edge_attr = data.edge_attr

        masked_nodes, mask_sizes = GraphExplainer.get_masked_nodes(x)
        data_list = GraphExplainer.make_data_list(x, edge_index, edge_attr, masked_nodes, self.device)
        data_loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False)

        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            prob, _, _ = self.model(batch)

        assert i == 0
        prob = prob.cpu().data.view(-1)

        mask_one_min = prob[:mask_sizes[0]].min().item()
        mask_two_min = prob[mask_sizes[0]:mask_sizes[1]].min().item()
        mask_thr_min = prob[mask_sizes[1]:mask_sizes[2]].min().item()

        trend = [original_prob, mask_one_min, mask_two_min, mask_thr_min]
        g_2_3 = trend[3] - trend[2]
        g_1_2 = trend[2] - trend[1]
        g_0_1 = trend[1] - trend[0]

        if g_2_3 * 4 < g_1_2 and g_2_3 < -0.05:
            n = 3
            best_idx = prob[mask_sizes[1]:mask_sizes[2]].argmin().item()
            final_prob = prob[mask_sizes[1]:mask_sizes[2]][best_idx].item()
            node_mask = list(masked_nodes[mask_sizes[1]:mask_sizes[2]][best_idx])

        elif g_1_2 * 4 < g_0_1:
            n = 2
            best_idx = prob[mask_sizes[0]:mask_sizes[1]].argmin().item()
            final_prob = prob[mask_sizes[0]:mask_sizes[1]][best_idx].item()
            node_mask = list(masked_nodes[mask_sizes[0]:mask_sizes[1]][best_idx])

        else:
            n = 1
            best_idx = prob[:mask_sizes[0]].argmin().item()
            final_prob = prob[:mask_sizes[0]][best_idx].item()
            node_mask = list(masked_nodes[:mask_sizes[0]][best_idx])

        if show_instance:
            graph_visz.plot_graph(xypos, data, node_mask, entry, labels, original_prob, final_prob, circle)
            sns.lineplot(x=[0, 1, 2, 3], y=[original_prob, mask_one_min, mask_two_min, mask_thr_min])
            plt.axvline(n, 0, 1, color='red')
            plt.show()
            # stats = [[masked_nodes[idx], np.around(prob[idx].item(), decimals=3)] for idx in prob.sort()[1].tolist()]
        return node_mask, original_prob, final_prob, n

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
        return all_1_comb + all_2_comb + all_3_comb, [len(all_1_comb), len(all_1_comb)+len(all_2_comb), len(all_1_comb)+len(all_2_comb)+len(all_3_comb)]


def main():

    cfg = Path('./configs') / 'config0_radius.yml'
    show_instance = True
    num_samples = 1000

    cfg_dict = yaml.safe_load(cfg.open('r'))
    m = Main(cfg_dict, '_'.join(cfg.stem.split('_')[1:]))
    m.load_checkpoint()

    pos_test_sample_idx = graph_visz.get_proper_idx(m.dataset.all_entries, m.dataset.idx_split)
    np.random.seed(1)
    np.random.shuffle(pos_test_sample_idx)
    pos_test_sample_idx = pos_test_sample_idx[:num_samples]

    all_xypos = graph_visz.get_xypos(m.dataset.all_entries)
    labels = ['mu_hit_sim_phi', 'mu_hit_sim_eta']

    m.batch_size = 1560
    explainer = GraphExplainer(m.model, m.batch_size, m.device)

    data_loader = DataLoader(m.dataset[pos_test_sample_idx], batch_size=1, shuffle=False)
    pbar = tqdm(data_loader)

    pos_hit_ids = []
    good_o_prob, good_f_prob = [], []
    bad_o_prob, bad_f_prob = [], []
    good_attack, bad_attack, model_error = [], [], []
    n_dist = []
    for i, data in enumerate(pbar):

        xypos = all_xypos[pos_test_sample_idx[i]]
        entry = m.dataset.all_entries.loc[pos_test_sample_idx[i]]

        node_mask, original_prob, final_prob, n_selected = explainer.explain_graph(data, show_instance, xypos, entry, labels)

        graph_visz.record_attack(original_prob, final_prob, pos_test_sample_idx, i, n_selected, node_mask, pos_hit_ids,
                                 model_error, bad_attack, bad_o_prob, bad_f_prob, good_attack, good_o_prob, good_f_prob, n_dist)

        graph_visz.set_desc(pbar, data_loader, i, original_prob, final_prob, n_selected, good_o_prob, good_f_prob, bad_o_prob, bad_f_prob, n_dist)

    sns.histplot(n_dist, discrete=True, stat='probability', cumulative=False)
    plt.show()

    print('good attack rate:', len(good_attack) / len(pos_test_sample_idx))
    print('bad attack rate:', len(bad_attack) / len(pos_test_sample_idx))
    print('model error rate:', len(model_error) / len(pos_test_sample_idx))

    pos_df = m.dataset.all_entries.loc[good_attack].reset_index(drop=True)
    dfs = root2df.Root2Df(data_dir=Path('./../data/raw')).read_df()
    neg200 = dfs['MinBiasPU200_MTD'].sample(n=100000).reset_index(drop=True)
    pos0 = dfs['DsTau3muPU0_MTD']
    pos0 = pos0[pos0.n_gen_tau == 1].reset_index(drop=True)
    pos0 = pos0.sample(n=100000).reset_index(drop=True)

    graph_visz.plot_important_hit_dist(pos_df, pos_hit_ids, neg200, pos0)


if __name__ == '__main__':
    main()
