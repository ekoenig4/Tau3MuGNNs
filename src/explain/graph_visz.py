import os
os.chdir('../')

import yaml
import numpy as np
import networkx as nx
from main import Main
from pathlib import Path
import matplotlib.pyplot as plt


import torch
from torch_geometric.utils.convert import to_networkx


def get_proper_idx(df, idx_split, to_neg):
    test_samples = df.loc[idx_split['test']].reset_index(drop=True)

    if to_neg:
        pos_test_sample_idx = np.array(idx_split['test'])[np.where((test_samples.y == 1))[0]]
    else:
        pos_test_sample_idx = np.array(idx_split['test'])[np.where((test_samples.y == 0))[0]]

    filtered_idx = []
    for idx in pos_test_sample_idx:
        n_hits = (df.loc[idx].mu_hit_station == 1).sum()
        if 4 <= n_hits <= 20:
            filtered_idx.append(idx)
    return filtered_idx


def get_xypos(df):
    all_xypos = df.apply(lambda entry: np.stack([entry['mu_hit_sim_r'] * np.cos(np.deg2rad(entry['mu_hit_sim_phi'])),
                                                 entry['mu_hit_sim_r'] * np.sin(np.deg2rad(entry['mu_hit_sim_phi']))]).T, axis=1)
    all_xypos = all_xypos.apply(lambda sample: {i: each for i, each in enumerate(sample)})
    return all_xypos


def plot_graph(xypos, data, important_node_ids, entry, labels, original_prob, final_prob, to_neg):
    fig, ax = plt.subplots()
    xlabel = f'original_prob: {original_prob:.2f}, final_prob: {final_prob:.2f}'
    if to_neg:
        tau_prop_labels = ['gen_tau_pt', 'gen_tau_eta', 'gen_tau_phi']
        entry.at['gen_tau_phi'] = np.rad2deg(entry['gen_tau_phi'])

        tau_prop = entry[tau_prop_labels].to_list()
        xlabel += '\n' + ', '.join([label + ': ' + str(np.around(tau_prop[i].item(), decimals=1)) for i, label in enumerate(tau_prop_labels)])

    plt.xlabel(xlabel)

    data.edge_index = torch.cat((data.intra_level_edge_index, data.inter_level_edge_index, data.virtual_edge_index), dim=1)
    data.edge_attr = torch.cat((data.intra_level_edge_features, data.inter_level_edge_features, data.virtual_edge_features), dim=0)

    xypos[data.x.shape[0] - 1] = np.array([0, 0], dtype=np.float32)
    y = np.zeros(data.x.shape[0])
    y[-1] = 1
    if important_node_ids is not None:
        y[important_node_ids] = 2

    node_feat = entry[labels].to_list()
    node_feat = [np.around(np.concatenate((each, [0])), decimals=1) for each in node_feat]
    node_feat = list(zip(*node_feat))
    short_labels = [label.split('_')[-1] for label in labels]
    node_feat = [dict(zip(short_labels, each)) for each in node_feat]

    feat_important_nodes = {i: each for i, each in enumerate(node_feat)}

    node_feat = {i: str(i)+', eta: '+ str(each['eta']) for i, each in enumerate(node_feat)}

    G = to_networkx(data)
    nx.draw_networkx_nodes(G, xypos, cmap='cool', node_size=500, node_color=y.tolist(), ax=ax)
    nx.draw_networkx_labels(G, xypos, labels=node_feat, font_size=7, ax=ax)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    title = '\n'.join([str(each) + ': ' + str(feat_important_nodes[each]) for each in important_node_ids])
    fig.set_figheight(7)
    fig.set_figwidth(7)
    plt.title(title, fontsize=10)
    plt.show()


def main():

    cfg = Path('./configs') / 'config0_no-neib.yml'

    cfg_dict = yaml.safe_load(cfg.open('r'))
    m = Main(cfg_dict, '_'.join(cfg.stem.split('_')[1:]))
    all_xypos = get_xypos(m.dataset.all_entries)

    idx = 550000
    data = m.dataset[idx]
    plot_graph(all_xypos.loc[idx], data, important_node_ids=None, entry=m.dataset.all_entries.loc[idx], labels=['mu_hit_sim_theta', 'mu_hit_sim_eta'])


if __name__ == '__main__':
    main()
