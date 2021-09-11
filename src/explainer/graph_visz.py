import os
os.chdir('../')

import yaml
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import seaborn as sns
from main import Main
from pathlib import Path
import matplotlib.pyplot as plt


import torch
from torch_geometric.utils.convert import to_networkx


def get_proper_idx(df, idx_split):
    idx = idx_split['test'] + idx_split['valid']
    test_samples = df.loc[idx].reset_index(drop=True)
    pos_test_sample_idx = np.array(idx)[np.where((test_samples.y == 1))[0]]

    filtered_idx = []
    for idx in pos_test_sample_idx:
        n_hits = ((df.loc[idx].mu_hit_station == 1) * (df.loc[idx].mu_hit_neighbor == 0)).sum()
        if 4 <= n_hits <= 20:
            filtered_idx.append(idx)
    return filtered_idx


def get_xypos(df):
    # all_xypos = df.apply(lambda entry: np.stack([entry['mu_hit_sim_r'] * np.cos(np.deg2rad(entry['mu_hit_sim_phi'])),
    #                                              entry['mu_hit_sim_r'] * np.sin(np.deg2rad(entry['mu_hit_sim_phi']))]).T, axis=1)
    all_xypos = df.apply(lambda entry: np.stack([entry['mu_hit_sim_eta'],
                                                 np.deg2rad(entry['mu_hit_sim_phi'])]).T, axis=1)
    all_xypos = all_xypos.apply(lambda sample: {i: each for i, each in enumerate(sample)})
    return all_xypos


def plot_graph(xypos, data, important_node_ids, entry, labels, original_prob, final_prob, circle=None):
    fig, ax = plt.subplots()
    xlabel = f'Eta\noriginal_prob: {original_prob:.2f}, final_prob: {final_prob:.2f}'

    xlabel += f"\n Gen_tauuuu , pt:{entry['gen_tau_pt'].item():.3f}, eta:{entry['gen_tau_eta'].item():.3f}, phi:{entry['gen_tau_phi'].item():.3f}"
    for i in range(3):
        plt.plot(entry['gen_mu_eta'][i], entry['gen_mu_phi'][i], marker='x', color='r', ls='')

    if circle is not None:
        plt.plot(circle['eta'], circle['phi'], marker='o', color='yellow', ls='')
        circle1 = plt.Circle((circle['eta'], circle['phi']), circle['r'], color='r', fill=False)
        plt.gca().add_patch(circle1)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Phi (Rad)', fontsize=12)

    if data.edge_index is None:
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

    node_feat = {i: str(i) for i, each in enumerate(node_feat)}

    G = to_networkx(data)
    nx.draw_networkx_nodes(G, xypos, cmap='cool', node_size=1000, node_color=y.tolist(), ax=ax)
    nx.draw_networkx_labels(G, xypos, labels=node_feat, font_size=12, ax=ax)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    title = '\n'.join([str(each) + ': ' + str(feat_important_nodes[each]) for each in important_node_ids])
    fig.set_figheight(12)
    fig.set_figwidth(12)
    plt.title(title, fontsize=12)
    plt.axis('equal')
    plt.show()


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


def record_attack(original_prob, final_prob, pos_test_sample_idx, i, n_selected, node_mask, pos_hit_ids,
                  model_error, bad_attack, bad_o_prob, bad_f_prob, good_attack, good_o_prob, good_f_prob, n_dist):
    if original_prob < 0.5:
        model_error.append(pos_test_sample_idx[i])
        n_dist.append(0)
    elif final_prob > 0.5:
        bad_attack.append(pos_test_sample_idx[i])
        bad_o_prob.append(original_prob), bad_f_prob.append(final_prob)
        n_dist.append(n_selected)
    else:
        good_attack.append(pos_test_sample_idx[i])
        good_o_prob.append(original_prob), good_f_prob.append(final_prob)
        n_dist.append(n_selected)
        pos_hit_ids.append(node_mask)


def set_desc(pbar, data_loader, i, original_prob, final_prob, n_selected, good_o_prob, good_f_prob, bad_o_prob, bad_f_prob, n_dist):
    pbar.set_description(f"original: {original_prob:.2f}, final: {final_prob:.2f}, # masked nodes: {n_selected:.2f}")
    if i == len(data_loader) - 1:
        pbar.set_description(
            f"good_original_avg: {np.mean(good_o_prob):.2f}, good_final_avg: {np.mean(good_f_prob):.2f}, "
            f"bad_original_avg: {np.mean(bad_o_prob):.2f}, bad_final_avg: {np.mean(bad_f_prob):.2f}, "
            f"total_original_avg: {np.mean(good_o_prob + bad_o_prob):.2f}, total_final_avg: {np.mean(good_f_prob + bad_f_prob):.2f}, "
            f"# masked nodes_avg: {np.mean(n_dist):.2f}")


def load_checkpoint(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])


def save_checkpoint(model, model_path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, model_path)


def main():

    cfg = Path('./configs') / 'config0_radius.yml'

    cfg_dict = yaml.safe_load(cfg.open('r'))
    m = Main(cfg_dict, '_'.join(cfg.stem.split('_')[1:]))
    all_xypos = get_xypos(m.dataset.all_entries)

    idx = 550000
    data = m.dataset[idx]
    plot_graph(all_xypos.loc[idx], data, important_node_ids=None, entry=m.dataset.all_entries.loc[idx], labels=['mu_hit_sim_theta', 'mu_hit_sim_eta'])


if __name__ == '__main__':
    main()
