# -*- coding: utf-8 -*-

"""
Created on 2021/4/14
@author: Siqi Miao
"""

import yaml
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import permutations, product
try:
    from .root2df import Root2Df
except:
    from root2df import Root2Df

import torch
import torch_geometric
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader


class Tau3MuDataset(InMemoryDataset):
    def __init__(self, setting, data_config):
        self.setting = setting
        self.data_dir = Path(data_config['data_dir'])
        self.conditions = data_config['conditions']
        self.add_self_loops = data_config.get('add_self_loops', None)
        self.node_feature_names = data_config['node_feature_names']
        self.edge_feature_names = data_config.get('edge_feature_names', [])
        self.only_one_tau = data_config['only_one_tau']
        self.splits = data_config['splits']
        self.pos_neg_ratio = data_config['pos_neg_ratio']
        self.radius = data_config.get('radius', False)
        self.virtual_node = data_config.get('virtual_node', False)
        self.augdR = data_config.get('augdR', False)
        self.filter_soft_mu = data_config.get('filter_soft_mu', False)

        super(Tau3MuDataset, self).__init__(root=self.data_dir)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.shape[-1]
        self.edge_attr_dim = self.data.edge_attr.shape[-1] if self.edge_feature_names else 0
        print_splits(self)

    @property
    def raw_file_names(self):
        return ['DsTau3muPU0_MTD.pkl', 'DsTau3muPU200_MTD.pkl', 'MinBiasPU200_MTD.pkl'] if 'mix' in self.setting else ['DsTau3muPU200_MTD.pkl', 'MinBiasPU200_MTD.pkl']

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed_{self.setting}')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        print('Please put .pkl files ino $PROJECT_DIR/data/raw!')
        raise KeyboardInterrupt

    def process(self):
        dfs = Root2Df(self.data_dir / 'raw').read_df(self.setting)
        df = self.get_df(dfs)

        data_list = []
        print('[INFO] Processing entries...')
        for entry in tqdm(df.itertuples(), total=len(df)):
            masked_entry = Tau3MuDataset.mask_hits(entry, self.conditions)

            if 'half' in self.setting:
                entry_signal_endcap, entry_nontau_endcap, entry_pos_endcap, entry_neg_endcap = Tau3MuDataset.split_endcap(masked_entry)
                if entry_signal_endcap is None:  # negative samples
                    entry_pos_endcap['y'], entry_neg_endcap['y'] = 0, 0
                    data_list.append(self.process_one_entry(entry_pos_endcap))
                    data_list.append(self.process_one_entry(entry_neg_endcap))
                else:  # positive samples
                    if 'check' in self.setting:
                        entry_nontau_endcap['y'] = 1
                        data_list.append(self.process_one_entry(entry_nontau_endcap))
                    else:
                        entry_signal_endcap['y'] = 1
                        data_list.append(self.process_one_entry(entry_signal_endcap))
            elif 'DT' in self.setting:
                data = self.process_one_entry(masked_entry)
                data_list.append(data)
            else:
                assert 'GNN-full' in self.setting
                data = self.process_one_entry(masked_entry)
                data_list.append(data)

        idx_split = Tau3MuDataset.get_idx_split(data_list, self.splits, self.pos_neg_ratio)
        data, slices = self.collate(data_list)

        print('[INFO] Saving data.pt...')
        torch.save((data, slices, idx_split), self.processed_paths[0])

    def process_one_entry(self, entry):
        if 'GNN' in self.setting:
            edge_index = Tau3MuDataset.build_graph(entry, self.add_self_loops, self.radius, self.virtual_node)
            edge_attr = Tau3MuDataset.get_edge_features(entry, edge_index, self.edge_feature_names, self.virtual_node, self.augdR)
            x = Tau3MuDataset.get_node_features(entry, self.node_feature_names, self.virtual_node)
            y = torch.tensor(entry['y']).float().view(-1, 1)
            return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        else:
            assert 'DT' in self.setting
            x = Tau3MuDataset.get_node_features(entry, self.node_feature_names, self.virtual_node)
            y = torch.tensor(entry['y']).float().view(-1, 1)
            return Data(x=x, y=y)

    def get_df(self, dfs):
        for k, v in dfs.items():
            for key in v.keys():
                if 'gen' not in key and key not in self.node_feature_names and key not in self.edge_feature_names and key not in ['n_mu_hit', 'mu_hit_station', 'mu_hit_neighbor', 'mu_hit_endcap']:
                    v.drop(key, inplace=True, axis=1)

        neg200 = dfs['MinBiasPU200_MTD']
        pos200 = dfs['DsTau3muPU200_MTD']
        pos0 = dfs.get('DsTau3muPU0_MTD', None)

        if self.only_one_tau:
            pos200 = pos200[pos200.n_gen_tau == 1].reset_index(drop=True)
            if pos0 is not None:
                pos0 = pos0[pos0.n_gen_tau == 1].reset_index(drop=True)

        if self.filter_soft_mu:
            pos200 = pos200[pos200.apply(lambda x: Tau3MuDataset.filter_mu_by_pt_eta(x), axis=1)].reset_index(drop=True)
            if pos0 is not None:
                pos0 = pos0[pos0.apply(lambda x: Tau3MuDataset.filter_mu_by_pt_eta(x), axis=1)].reset_index(drop=True)

        if pos0 is not None and len(pos0) > 100000:
            print('[INFO] Sampling from pos0 to fasten processing & training...')
            pos0 = pos0.sample(100000).reset_index(drop=True)

        if 'mix' in self.setting:
            if 'check' in self.setting:
                pos, _ = self.mix(pos0, neg200)
                neg = pos200
            else:
                pos, neg = self.mix(pos0, neg200)
        else:
            pos, neg = pos200, neg200

        if 'half' in self.setting:
            min_pos_neg_ratio = len(pos) / (len(neg) * 2)
        else:
            min_pos_neg_ratio = len(pos) / len(neg)
        print(f'[INFO] min_pos_neg_ratio: {min_pos_neg_ratio}')

        pos['y'], neg['y'] = 1, 0
        assert self.pos_neg_ratio >= min_pos_neg_ratio, f'min_pos_neg_ratio = {min_pos_neg_ratio}! Now pos_neg_ratio = {self.pos_neg_ratio}!'
        return pd.concat((pos, neg), join='outer', ignore_index=True)

    @staticmethod
    def filter_mu_by_pt_eta(x):
        return ((x['mu_hit_station'] <= 4).sum() >= 1)
        # return ((x['mu_hit_station'] <= 4).sum() >= 1)  # good-4
        # return ((x['mu_hit_station'] == 1).sum() >= 3)  # bad-1
        # return ((x['mu_hit_station'] <= 1).sum() >= 1)  # bad-1
        # return ((x['mu_hit_station'] <= 2).sum() >= 0)  # good-2 bad-1
        # return ((x['mu_hit_station'] <= 2).sum() >= 1)  # good
        # return ((x['mu_hit_station'] <= 2).sum() >= 3)  # good
        # return ((x['gen_mu_pt'] > 0.5).sum() == 3)
        # return ((x['gen_mu_pt'] > 0.5).sum() == 3) and ((abs(x['gen_mu_eta']) < 2.8).sum() == 3) and ((abs(x['gen_mu_eta']) > 1.2).sum() == 3)  # good

    @staticmethod
    def get_intra_station_edges(entry, hit_id, radius):
        if radius:
            coors = Tau3MuDataset.get_coors_for_hits(entry, hit_id)
            if coors.shape[0] == 0:
                return torch.tensor([]).reshape(2, -1)
            return radius_graph(coors, r=radius, loop=False)
        else:
            return torch.tensor(list(permutations(hit_id, 2))).T

    @staticmethod
    def get_inter_station_edges(hit_ids):
        edge_index = list(product(hit_ids[0], hit_ids[1])) + list(product(hit_ids[1], hit_ids[0]))
        return torch.tensor(edge_index).T

    @staticmethod
    def get_coors_for_hits(entry, hit_id):
        eta, phi = entry['mu_hit_sim_eta'][hit_id], np.deg2rad(entry['mu_hit_sim_phi'])[hit_id]
        coors = torch.tensor(np.stack((eta, phi)).T)
        return coors

    @staticmethod
    def build_graph(entry, add_self_loops, radius, virtual_node):
        station2hitids = Tau3MuDataset.groupby_station(entry['mu_hit_station'])

        intra_station_edges = []
        for hit_id in station2hitids.values():
            intra_station_edges.append(Tau3MuDataset.get_intra_station_edges(entry, hit_id, radius))
        intra_station_edges = torch.cat(intra_station_edges, dim=1) if len(intra_station_edges) != 0 else torch.tensor([]).reshape(2, -1)

        # We cannot simply iterate four stations since many samples do not hit all the four stations.
        # Some samples may hit station [1, 2, 3], some may hit [1], and some may hit [1, 2, 4].
        inter_station_edges = []
        ordered_station_ids = sorted(station2hitids.keys())
        for i in range(len(ordered_station_ids) - 1):
            station_0, station_1 = ordered_station_ids[i], ordered_station_ids[i + 1]
            inter_station_edges.append(Tau3MuDataset.get_inter_station_edges((station2hitids[station_0], station2hitids[station_1])))
        inter_station_edges = torch.cat(inter_station_edges, dim=1) if len(inter_station_edges) != 0 else torch.tensor([]).reshape(2, -1)

        virtual_node_id = entry['n_mu_hit']
        real_node_ids = [i for i in range(virtual_node_id)]
        virtual_edges = Tau3MuDataset.get_virtual_edges(virtual_node_id, real_node_ids)
        virtual_edges = torch.tensor(virtual_edges).T if len(virtual_edges) != 0 else torch.tensor([]).reshape(2, -1)

        if virtual_node:
            edge_index = torch.cat((intra_station_edges, inter_station_edges, virtual_edges), dim=1).long()
        else:
            edge_index = torch.cat((intra_station_edges, inter_station_edges), dim=1).long()

        if add_self_loops and edge_index.shape != (0,):
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)
        return edge_index

    @staticmethod
    def get_virtual_edges(virtual_node_id, real_node_ids):
        return list(product([virtual_node_id], real_node_ids)) + list(product(real_node_ids, [virtual_node_id]))

    @staticmethod
    def groupby_station(stations: np.ndarray) -> dict:
        station2hitids = {}
        for hit_id, station_id in enumerate(stations):
            if station2hitids.get(station_id) is None:
                station2hitids[station_id] = []
            station2hitids[station_id].append(hit_id)
        return station2hitids

    @staticmethod
    def get_node_features(entry, feature_names, virtual_node):
        if entry['n_mu_hit'] == 0:
            virtual_node = True

        features = np.stack([entry[feature] for feature in feature_names], axis=1)
        if virtual_node:
            features = np.concatenate((features, np.zeros((1, features.shape[1]))))
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def get_edge_features(entry, edge_index, feature_names, virtual_node, augdR):
        if edge_index.shape == (2, 0):
            return torch.tensor([])

        # Directly index the entry using features = entry[feature_names] is extremely slow!
        features = np.stack([entry[feature] for feature in feature_names], axis=1)

        if virtual_node:
            # Initialize the feature of the virtual node with all zeros.
            features = np.concatenate((features, np.zeros((1, features.shape[1]))), axis=0)

        edge_features = features[edge_index[0]] - features[edge_index[1]]
        if augdR:
            eta, phi = entry['mu_hit_sim_eta'], np.deg2rad(entry['mu_hit_sim_phi'])
            if virtual_node:
                eta, phi = np.append(eta, 0), np.append(phi, 0)
            dR = (eta[edge_index[0]] - eta[edge_index[1]])**2 + (phi[edge_index[0]] - phi[edge_index[1]])**2
            dR = dR**0.5
            edge_features = np.concatenate((edge_features, dR.reshape(-1, 1)), axis=1)
        return torch.tensor(edge_features, dtype=torch.float)

    @staticmethod
    def get_idx_split(data_list, splits, pos2neg):
        assert sum(splits.values()) == 1.0
        y_dist = np.array([data.y.item() for data in data_list])

        pos_idx = np.argwhere(y_dist == 1).reshape(-1)
        neg_idx = np.argwhere(y_dist == 0).reshape(-1)
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        assert len(pos_idx) <= len(neg_idx) * pos2neg, 'The number of negative samples is not enough given the pos_neg_ratio!'
        n_train_pos, n_valid_pos = int(splits['train'] * len(pos_idx)), int(splits['valid'] * len(pos_idx))
        n_train_neg, n_valid_neg = int(n_train_pos / pos2neg), int(n_valid_pos / pos2neg)

        pos_train_idx = pos_idx[:n_train_pos]
        pos_valid_idx = pos_idx[n_train_pos:n_train_pos + n_valid_pos]
        pos_test_idx = pos_idx[n_train_pos + n_valid_pos:]

        neg_train_idx = neg_idx[:n_train_neg]
        neg_valid_idx = neg_idx[n_train_neg:n_train_neg + n_valid_neg]
        neg_test_idx = neg_idx[n_train_neg + n_valid_neg:]

        return {'train': np.concatenate((pos_train_idx, neg_train_idx)).tolist(),
                'valid': np.concatenate((pos_valid_idx, neg_valid_idx)).tolist(),
                'test': np.concatenate((pos_test_idx, neg_test_idx)).tolist()}

    @staticmethod
    def mix(pos0, neg200):
        neg_idx = np.arange(len(neg200))
        np.random.shuffle(neg_idx)

        # first len(pos0) neg data will be used as noise in pos0
        noise_in_pos0 = neg200.loc[neg_idx[:len(pos0)]].reset_index(drop=True)
        # rest neg data will remain negative
        neg200 = neg200.loc[neg_idx[len(pos0):]].reset_index(drop=True)

        print('[INFO] Mixing data...')
        mixed_pos = []
        for idx, entry in tqdm(pos0.iterrows(), total=len(pos0)):
            for k, v in entry.items():
                if 'gen' in k:  # directly keep gen variables
                    continue
                elif isinstance(v, int):  # accumulate n_mu_hit
                    assert k == 'n_mu_hit'
                    entry[k] += noise_in_pos0.iloc[idx][k]
                else:  # concat hit features
                    assert isinstance(v, np.ndarray)
                    mixed_hits = np.concatenate((noise_in_pos0.iloc[idx][k], v))
                    entry[k] = mixed_hits
            mixed_pos.append(entry.values)
        mixed_pos = pd.DataFrame(data=mixed_pos, columns=entry.index)
        return mixed_pos, neg200

    @staticmethod
    def split_endcap(masked_entry):
        entry_pos_endcap, entry_neg_endcap = {}, {}
        pos_endcap_idx = masked_entry['mu_hit_endcap'] == 1
        neg_endcap_idx = masked_entry['mu_hit_endcap'] == -1

        for k, v in masked_entry.items():
            if isinstance(v, np.ndarray) and 'gen' not in k and k != 'y' and 'L1' not in k:
                assert v.shape[0] == masked_entry['n_mu_hit']
                entry_pos_endcap[k] = v[pos_endcap_idx]
                entry_neg_endcap[k] = v[neg_endcap_idx]
            else:
                entry_pos_endcap[k] = v
                entry_neg_endcap[k] = v
        entry_pos_endcap['n_mu_hit'] = pos_endcap_idx.sum().item()
        entry_neg_endcap['n_mu_hit'] = neg_endcap_idx.sum().item()

        entry_signal_endcap, entry_nontau_endcap = None, None
        if masked_entry['y'] == 1:
            if ((masked_entry['gen_tau_eta'] * entry_pos_endcap['mu_hit_sim_eta']) > 0).sum() == entry_pos_endcap['n_mu_hit']:
                entry_signal_endcap = entry_pos_endcap
                entry_nontau_endcap = entry_neg_endcap
            else:
                assert ((masked_entry['gen_tau_eta'] * entry_neg_endcap['mu_hit_sim_eta']) > 0).sum() == entry_neg_endcap['n_mu_hit']
                entry_signal_endcap = entry_neg_endcap
                entry_nontau_endcap = entry_pos_endcap

        return entry_signal_endcap, entry_nontau_endcap, entry_pos_endcap, entry_neg_endcap

    @staticmethod
    def mask_hits(entry, conditions):
        mask = np.ones(entry.n_mu_hit, dtype=bool)
        for k, v in conditions.items():
            k = k.split('-')[1]
            assert isinstance(getattr(entry, k), np.ndarray)
            mask *= eval('entry.' + k + v)

        masked_entry = {'n_mu_hit': mask.sum()}
        for k in entry._fields:
            value = getattr(entry, k)
            if isinstance(value, np.ndarray) and 'gen' not in k and k != 'y':
                assert value.shape[0] == entry.n_mu_hit
                masked_entry[k] = value[mask].reshape(-1)
            else:
                if k != 'n_mu_hit':
                    masked_entry[k] = value
        return masked_entry


def get_data_loaders(setting, data_config, batch_size):
    dataset = Tau3MuDataset(setting, data_config)
    train_loader = DataLoader(dataset[dataset.idx_split['train']], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[dataset.idx_split['valid']], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[dataset.idx_split['test']], batch_size=batch_size, shuffle=False)
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, dataset.x_dim, dataset.edge_attr_dim, dataset


def print_splits(dataset):
    def get_pos_neg_count(y):
        pos_y = (y == 1).sum()
        neg_y = (y == 0).sum()
        pos2neg_ratio = pos_y / neg_y
        return pos_y, neg_y, pos2neg_ratio

    print('[Splits]')
    for k, v in dataset.idx_split.items():
        y = dataset.data.y[v]
        pos_y, neg_y, pos2neg_ratio = get_pos_neg_count(y)
        print(f'    {k}: {len(v)}. # pos: {pos_y}, # neg: {neg_y}. Pos:Neg: {pos2neg_ratio:.3f}')
