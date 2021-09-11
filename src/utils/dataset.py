# -*- coding: utf-8 -*-

"""
Created on 2021/4/14
@author: Siqi Miao
"""

import yaml
import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.root2df import Root2Df
from itertools import combinations, permutations, product

import torch
import torch_geometric
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data, InMemoryDataset

from torch import Tensor
from pathlib import Path
from typing import List, Tuple, Any, Union


class Tau3MuDataset(InMemoryDataset):
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.conditions = config['conditions']
        self.add_self_loops = config['add_self_loops']
        self.node_feature_names = config['node_feature_names']
        self.edge_feature_names = config['edge_feature_names']
        self.virtual_node = config['virtual_node']
        self.only_one_tau = config['only_one_tau']
        self.splits = config['splits']
        self.pos2neg = config['pos2neg']
        self.random_state = config['random_state']
        self.hit_constrain = config['hit_constrain']
        self.visz = config['visz']
        self.filter_soft_mu = config['filter_soft_mu']
        self.normalize = config['normalize']
        self.one_hot = config['one_hot']
        self.radius = config['radius']
        self.endcap = config['endcap']

        self.run_type = config['run_type']
        print('[INFO] run_type:', self.run_type)
        assert self.run_type in ['reg', 'mix', 'check', 'da_eval', 'regress']

        self.node_feature_stat = None

        super(Tau3MuDataset, self).__init__(root=self.data_dir)
        self.data, self.slices, self.idx_split, self.node_feature_stat = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.shape[-1]

        if self.data.edge_attr is None:
            self.edge_attr_dim = max(self.data.intra_level_edge_features.shape[-1],
                                     self.data.inter_level_edge_features.shape[-1])
        else:
            self.edge_attr_dim = self.data.edge_attr.shape[-1]

        self.data_list_endcap = torch.load(Path(self.processed_dir) / 'data_list_endcap.pt') if self.endcap else None
        self.all_entries = torch.load(Path(self.processed_dir) / 'all_entries.pt') if self.visz else None

    @property
    def raw_file_names(self):
        return ['DsTau3muPU0_MTD.pkl', 'DsTau3muPU200_MTD.pkl',
                'MinBiasPU200_MTD.pkl', 'MinBiasPU250_MTD.pkl', 'MinBiasPU140_MTD.pkl']

    @property
    def processed_file_names(self):
        processed_dir = Path(self.processed_dir)
        prev_data_config = processed_dir / 'data_config.yml'
        if prev_data_config.exists() and self.config != yaml.safe_load((processed_dir / 'data_config.yml').open('r')):
            print('[INFO] Data config has been changed. Deleting data.pt...')
            (processed_dir / 'data_config.yml').unlink(missing_ok=True)
            (processed_dir / 'data_md5.txt').unlink(missing_ok=True)
            (processed_dir / 'data.pt').unlink(missing_ok=True)
        return ['data.pt']

    def download(self):
        print('Please put .pkl files ino $PROJECT_DIR/data/raw!')
        raise KeyboardInterrupt

    def process(self):
        dfs = Root2Df(self.data_dir / 'raw').read_df()
        df = self.get_df(dfs)
        if self.hit_constrain:
            df = Tau3MuDataset.df_filter(df, self.conditions, self.hit_constrain)

        print('[INFO] Splitting datasets...')
        idx_split = Tau3MuDataset.get_idx_split(df, self.splits, self.pos2neg, self.run_type, self.random_state)
        del dfs

        data_list, data_list_endcap = [], []
        print('[INFO] Processing entries...')
        np.random.seed(self.random_state)

        idx_for_training = set(idx_split['train'])
        for pd_idx, entry in enumerate(tqdm(df.itertuples(), total=len(df))):
            masked_entry = Tau3MuDataset.mask_hits(df, entry, self.conditions)
            if not self.endcap:
                data_list.append(self.process_one_entry(masked_entry, pd_idx))
            else:
                entry_pos_endcap, entry_neg_endcap = {}, {}
                pos_endcap_idx = masked_entry['mu_hit_endcap'] == 1
                neg_endcap_idx = masked_entry['mu_hit_endcap'] == -1

                for k, v in masked_entry.items():
                    if isinstance(v, np.ndarray) and 'gen' not in k and k != 'y' and 'L1' not in k:
                        entry_pos_endcap[k] = v[pos_endcap_idx]
                        entry_neg_endcap[k] = v[neg_endcap_idx]
                    else:
                        entry_pos_endcap[k] = v
                        entry_neg_endcap[k] = v
                    entry_pos_endcap['n_mu_hit'] = pos_endcap_idx.sum().item()
                    entry_neg_endcap['n_mu_hit'] = neg_endcap_idx.sum().item()

                if masked_entry['y'] == 1:
                    if ((masked_entry['gen_tau_eta'] * entry_pos_endcap['mu_hit_sim_eta']) > 0).sum() == entry_pos_endcap['n_mu_hit']:
                        entry_pos_endcap['y'] = 1
                        entry_neg_endcap['y'] = 0
                    else:
                        assert ((masked_entry['gen_tau_eta'] * entry_neg_endcap['mu_hit_sim_eta']) > 0).sum() == entry_neg_endcap['n_mu_hit']
                        entry_pos_endcap['y'] = 0
                        entry_neg_endcap['y'] = 1

                for k in ['mu_hit_sim_theta', 'mu_hit_sim_eta', 'mu_hit_sim_z']:
                    if k == 'mu_hit_sim_theta':
                        if k in entry_pos_endcap.keys():
                            entry_pos_endcap[k] = abs(entry_pos_endcap[k] - 90)
                            entry_neg_endcap[k] = abs(entry_neg_endcap[k] - 90)
                    elif k in ['mu_hit_sim_eta', 'mu_hit_sim_z']:
                        if k in entry_pos_endcap.keys():
                            entry_pos_endcap[k] = abs(entry_pos_endcap[k])
                            entry_neg_endcap[k] = abs(entry_neg_endcap[k])

                if pd_idx in idx_for_training:
                    if np.random.random() > 0.5:
                        data_list.append(self.process_one_entry(entry_pos_endcap, pd_idx))
                        data_list_endcap.append(self.process_one_entry(entry_neg_endcap, pd_idx))
                    else:
                        data_list.append(self.process_one_entry(entry_neg_endcap, pd_idx))
                        data_list_endcap.append(self.process_one_entry(entry_pos_endcap, pd_idx))
                else:
                    data_list.append(self.process_one_entry(entry_pos_endcap, pd_idx))
                    data_list_endcap.append(self.process_one_entry(entry_neg_endcap, pd_idx))

        data, slices = self.collate(data_list)
        del data_list

        if self.endcap:
            torch.save(data_list_endcap, Path(self.processed_dir) / 'data_list_endcap.pt')
            del data_list_endcap

        print('[INFO] Saving data.pt...')
        torch.save((data, slices, idx_split, self.node_feature_stat), self.processed_paths[0])
        yaml.dump(self.config, open(Path(self.processed_dir) / 'data_config.yml', 'w'))
        (Path(self.processed_dir) / 'data_md5.txt').open('w').write(Tau3MuDataset.md5sum(data, slices, idx_split, True))

        if self.visz:
            del data
            print('[INFO] Saving all_entries.pt...')
            torch.save(df, Path(self.processed_dir) / 'all_entries.pt')

    def process_one_entry(self, entry, pd_idx=-1):
        if self.radius:
            eta, phi = entry['mu_hit_sim_eta'], np.deg2rad(entry['mu_hit_sim_phi'])
        if self.normalize:
            entry = Tau3MuDataset.z_score_with_missing_values(entry, self.node_feature_stat)

        x, score_gt = Tau3MuDataset.get_node_features(entry, self.node_feature_names, self.virtual_node, self.one_hot, self.run_type)

        if self.run_type == 'regress':
            y = torch.tensor(entry['y'], dtype=torch.float).view(-1, 9)
        else:
            y = torch.tensor(0 if entry['y'] <= 0 else 1, dtype=torch.float).view(-1, 1)

        intra_level_edges, inter_level_edges, virtual_edges = Tau3MuDataset.build_graph(entry, self.add_self_loops)
        if self.radius:
            coors = torch.tensor(np.stack((eta, phi)).T)
            if coors.shape[0] == 0:
                edge_index = intra_level_edges
            else:
                edge_index = radius_graph(coors, r=self.radius, loop=True)
            edge_features = Tau3MuDataset.get_edge_features(entry, edge_index, self.edge_feature_names,
                                                            for_virtual_edges=False)
            virtual_edge_features = Tau3MuDataset.get_edge_features(entry, virtual_edges, self.edge_feature_names,
                                                                    for_virtual_edges=True)

            edge_index = torch.cat((edge_index, virtual_edges), dim=1)
            edge_attr = torch.cat((edge_features, virtual_edge_features), dim=0)

            return Data(x=x, y=y, num_nodes=x.shape[0], score_gt=score_gt,
                        pd_idx=torch.tensor(pd_idx, dtype=torch.float).view(-1, 1),
                        edge_index=edge_index, edge_attr=edge_attr)
        else:
            intra_level_edge_features = Tau3MuDataset.get_edge_features(entry, intra_level_edges, self.edge_feature_names, for_virtual_edges=False)
            inter_level_edge_features = Tau3MuDataset.get_edge_features(entry, inter_level_edges, self.edge_feature_names, for_virtual_edges=False)
            virtual_edge_features = Tau3MuDataset.get_edge_features(entry, virtual_edges, self.edge_feature_names, for_virtual_edges=True)
            if not self.virtual_node:
                virtual_edge_features = virtual_edges = torch.tensor([], dtype=torch.long)

            return Data(x=x, y=y, num_nodes=x.shape[0], score_gt=score_gt,
                        pd_idx=torch.tensor(pd_idx, dtype=torch.float).view(-1, 1),
                        intra_level_edge_index=intra_level_edges,
                        inter_level_edge_index=inter_level_edges,
                        virtual_edge_index=virtual_edges,
                        intra_level_edge_features=intra_level_edge_features,
                        inter_level_edge_features=inter_level_edge_features,
                        virtual_edge_features=virtual_edge_features)

    @staticmethod
    def mask_hits(df, entry: pd.Series, conditions: dict) -> dict:
        mask = np.ones(entry.n_mu_hit, dtype=bool)
        for k, v in conditions.items():
            k = k.split('-')[1]
            assert isinstance(getattr(entry, k), np.ndarray)
            mask *= eval('entry.' + k + v)

        masked_entry = {'n_mu_hit': mask.sum()}
        df.at[entry.Index, 'n_mu_hit'] = masked_entry['n_mu_hit']
        new_hit_order = np.arange(masked_entry['n_mu_hit'])
        np.random.shuffle(new_hit_order)

        for k in entry._fields:
            value = getattr(entry, k)
            if isinstance(value, np.ndarray) and 'gen' not in k and k != 'y' and 'L1' not in k:
                masked_entry[k] = value[mask].reshape(-1)  #[new_hit_order]
                df.at[entry.Index, k] = masked_entry[k]
            else:
                if k != 'n_mu_hit':
                    masked_entry[k] = value
        return masked_entry

    @staticmethod
    def groupby_station(stations: np.ndarray) -> dict:
        station2hitids = {}
        for hit_id, station_id in enumerate(stations):
            if station2hitids.get(station_id) is None:
                station2hitids[station_id] = []
            station2hitids[station_id].append(hit_id)
        return station2hitids

    @staticmethod
    def get_intra_level_edges(node_list: List[int], directed: bool) -> List[Tuple[int]]:
        if directed:
            return list(combinations(node_list, 2))
        else:
            return list(permutations(node_list, 2))

    @staticmethod
    def get_inter_level_edges(nodes_lists: Tuple[List[int], List[int]], flow: str) -> List[Tuple[Any]]:
        if flow == 'undirected':
            return list(product(nodes_lists[0], nodes_lists[1])) + list(product(nodes_lists[1], nodes_lists[0]))
        elif flow == '0 -> 1':
            return list(product(nodes_lists[0], nodes_lists[1]))
        elif flow == '1 -> 0':
            return list(product(nodes_lists[1], nodes_lists[0]))
        else:
            raise NotImplementedError

    @staticmethod
    def get_virtual_edges(virtual_node_id: int, real_node_ids: List[int]) -> List[Tuple[Any]]:
        return list(product([virtual_node_id], real_node_ids)) + list(product(real_node_ids, [virtual_node_id]))

    @staticmethod
    def get_node_features(entry: dict, feature_names, virtual_node, one_hot, run_type) -> Tuple:
        feature_names = feature_names[:]
        one_hot_vec = None
        if one_hot:
            if 'mu_hit_ring' in feature_names:
                one_hot_vec = np.zeros((len(entry['mu_hit_ring']), 4))
                for idx, ring in enumerate(entry['mu_hit_ring']):
                    one_hot_vec[idx][ring - 1] = 1
                feature_names.remove('mu_hit_ring')

        features = np.stack([entry[feature] for feature in feature_names], axis=1)
        features = features if one_hot_vec is None else np.concatenate((one_hot_vec, features), axis=1)

        if virtual_node:
            features = np.concatenate((features, np.zeros((1, features.shape[1]))))

        score_gt = torch.tensor([], dtype=torch.float)
        if run_type == 'mix' and entry['score_gt'] is not None:
            if virtual_node:
                entry['score_gt'] = np.concatenate((entry['score_gt'], [1]))
            # TODO: when there is no virtual node, the denominator will be zero.
            assert virtual_node
            score_gt = torch.tensor(entry['score_gt'] / entry['score_gt'].sum(), dtype=torch.float)

        return torch.tensor(features, dtype=torch.float), score_gt

    @staticmethod
    def get_edge_features(entry: dict, edges: Tensor, feature_names: List[str], for_virtual_edges: bool) -> Tensor:
        if edges.shape == (0,):
            return torch.tensor([])

        # Directly index the entry using entry[feature_names] is extremely slow!
        features = np.stack([entry[feature] for feature in feature_names], axis=1)
        if for_virtual_edges:
            # Initialize the feature of the virtual node with all zeros.
            features = np.concatenate((features, np.zeros((1, len(feature_names)))))
        features = torch.tensor(features, dtype=torch.float)
        return features[edges[0]] - features[edges[1]]

    @staticmethod
    def build_graph(entry: dict, add_self_loops: bool) -> List[Tensor]:
        dim = entry['n_mu_hit']
        virtual_node_id = dim
        real_node_ids = [i for i in range(dim)]

        station2hitids = Tau3MuDataset.groupby_station(entry['mu_hit_station'])

        intra_level_edges = []
        for value in station2hitids.values():
            intra_level_edges.extend(Tau3MuDataset.get_intra_level_edges(value, directed=False))

        inter_level_edges = []
        # We cannot simply iterate four stations since many samples do not hit all the four stations.
        # Some samples may hit station [1, 2, 3], some may hit [1], and some may hit [1, 2, 4].
        ordered_station_ids = sorted(station2hitids.keys())
        for i in range(len(ordered_station_ids) - 1):
            station_0, station_1 = ordered_station_ids[i], ordered_station_ids[i + 1]
            inter_level_edges.extend(Tau3MuDataset.get_inter_level_edges((station2hitids[station_0],
                                                                          station2hitids[station_1]),
                                                                         flow='undirected'))

        virtual_edges = Tau3MuDataset.get_virtual_edges(virtual_node_id, real_node_ids)

        all_level_edges = [intra_level_edges, inter_level_edges, virtual_edges]
        for idx, one_level_edge in enumerate(all_level_edges):
            one_level_edge = torch.tensor(one_level_edge, dtype=torch.long).T
            if add_self_loops and one_level_edge.shape != (0,):
                one_level_edge, _ = torch_geometric.utils.add_self_loops(one_level_edge)
            all_level_edges[idx] = one_level_edge

        return all_level_edges

    @staticmethod
    def filter_mu_by_pt_eta(x):
        return ((x['gen_mu_pt'] > 0.5).sum() == 3) and ((abs(x['gen_mu_eta']) < 2.8).sum() == 3) and ((abs(x['gen_mu_eta']) > 1.2).sum() == 3)

    @staticmethod
    def md5sum(data, slices, idx_split, print_md5):
        m = hashlib.md5()
        for key in data.keys:
            assert isinstance(data[key], Tensor)
            m.update(key.encode('utf-8'))
            m.update(str(data[key]).encode('utf-8'))
        for key in slices.keys():
            assert isinstance(slices[key], Tensor)
            m.update(key.encode('utf-8'))
            m.update(str(slices[key]).encode('utf-8'))
        for key in idx_split.keys():
            assert isinstance(idx_split[key], List)
            m.update(key.encode('utf-8'))
            m.update(str(idx_split[key]).encode('utf-8'))
        md5 = m.hexdigest()
        if print_md5:
            print(f'[INFO] Data md5: {md5}')
        return md5

    def get_df(self, dfs):
        for k, v in dfs.items():
            for key in v.keys():
                if 'gen' not in key and key not in self.node_feature_names and key not in ['n_mu_hit', 'mu_hit_station', 'mu_hit_neighbor', 'mu_hit_endcap']:
                    v.drop(key, inplace=True, axis=1)

        neg250 = dfs['MinBiasPU250_MTD']
        neg200 = dfs['MinBiasPU200_MTD']
        neg140 = dfs['MinBiasPU140_MTD']

        pos0 = dfs['DsTau3muPU0_MTD']
        pos200 = dfs['DsTau3muPU200_MTD']

        # pos0 = pos0.sample(len(pos200) * 2).reset_index(drop=True)

        if self.filter_soft_mu:
            pos0 = pos0[pos0.apply(lambda x: Tau3MuDataset.filter_mu_by_pt_eta(x), axis=1)].reset_index(drop=True)
            pos200 = pos200[pos200.apply(lambda x: Tau3MuDataset.filter_mu_by_pt_eta(x), axis=1)].reset_index(drop=True)
        if self.only_one_tau:
            pos0 = pos0[pos0.n_gen_tau == 1].reset_index(drop=True)
            pos200 = pos200[pos200.n_gen_tau == 1].reset_index(drop=True)
        if self.normalize:
            print('[INFO] Normalizing features...')
            assert self.conditions['1-mu_hit_station'] == '==1'
            # assert len(self.node_feature_names) == 8
            self.node_feature_stat = Tau3MuDataset.z_score(neg200, self.node_feature_names)

        join = 'outer' if self.visz else 'inner'
        if self.run_type == 'mix':
            neg200, pos200, mixed_pos = Tau3MuDataset.mix(neg200, pos200, pos0, self.random_state)
            neg200['y'], pos200['y'], mixed_pos['y'] = 0, 1, 3
            return pd.concat((neg200, pos200, mixed_pos), join=join, ignore_index=True)
        elif self.run_type == 'check':
            neg200, pos200, mixed_pos = Tau3MuDataset.mix(neg200, pos200, pos0, self.random_state)
            pos200['y'], mixed_pos['y'] = 1, 0
            return pd.concat((pos200, mixed_pos), join=join, ignore_index=True)
        elif self.run_type == 'reg':
            neg200['y'], pos200['y'] = 0, 1
            return pd.concat((neg200, pos200), join=join, ignore_index=True)
        elif self.run_type == 'da_eval':
            neg200['y'], pos200['y'] = 0, 1
            df = pd.concat((neg200, pos200), join=join, ignore_index=True)
            idx_split = Tau3MuDataset.get_idx_split(df, self.splits, self.pos2neg, 'reg', self.random_state)
            test_neg200_pos200_df = df.loc[idx_split['test']]
            neg200 = test_neg200_pos200_df[test_neg200_pos200_df['y'] == 0].reset_index(drop=True)
            pos200 = test_neg200_pos200_df[test_neg200_pos200_df['y'] == 1].reset_index(drop=True)

            neg140['y'], neg250['y'], pos0['y'] = -2, -1, 2
            return pd.concat((neg140, neg250, neg200, pos200, pos0), join=join, ignore_index=True)
        elif self.run_type == 'regress':
            y0, y200 = [], []
            for key in ['gen_tau_e', 'gen_tau_pt', 'gen_tau_eta']:
                value0 = np.array([each.item() for each in pos0[key]])
                value200 = np.array([each.item() for each in pos200[key]])

                mu, std = np.mean(value0), np.std(value0)

                value0 = (value0 - mu) / std
                value200 = (value200 - mu) / std

                mu0, std0 = np.full_like(value0, mu.item()), np.full_like(value0, std.item())
                mu200, std200 = np.full_like(value200, mu.item()), np.full_like(value200, std.item())

                y0.append(np.stack((value0, mu0, std0)).T)
                y200.append(np.stack((value200, mu200, std200)).T)

            y0 = np.concatenate(y0, axis=1)
            y200 = np.concatenate(y200, axis=1)

            pos0['y'] = [each for each in y0]
            pos200['y'] = [each for each in y200]

            pos0['type'] = 'pos0'
            pos200['type'] = 'pos200'

            return pd.concat((pos0, pos200), join=join, ignore_index=True)
        else:
            neg140['y'], neg250['y'], neg200['y'], pos200['y'], pos0['y'] = -2, -1, 0, 1, 2
            return pd.concat((neg140, neg250, neg200, pos200, pos0), join=join, ignore_index=True)

    @staticmethod
    def df_filter(df, conditions, hit_constrain):
        condi = []
        for k, v in conditions.items():
            k = k.split('-')[1]
            condi.append('(' + 'x.' + k + v + ')')
        condi = ' * '.join(condi)
        condi = '(' + condi + ')' + '.sum()' + hit_constrain

        # use eval() to construct condition expression used in lambda()
        # df = df[df.apply(lambda x: ((x['mu_hit_station'] == 1) * (x['mu_hit_neighbor'] == 0)).sum() >= 3, axis=1)].reset_index(drop=True)

        df = df[df.apply(lambda x: eval(condi), axis=1)].reset_index(drop=True)
        return df

    @staticmethod
    def mix(neg200, pos200, pos0, random_state):
        pos200['score_gt'] = None
        neg200['score_gt'] = neg200.apply(lambda x: np.zeros(x['n_mu_hit']), axis=1)
        pos0['score_gt'] = pos0.apply(lambda x: np.ones(x['n_mu_hit']), axis=1)

        neg_idx = np.arange(len(neg200))
        np.random.seed(random_state)
        np.random.shuffle(neg_idx)

        # first len(pos) neg data will be used as noise in pos0
        noise_in_pos = neg200.loc[neg_idx[:len(pos0)]].reset_index(drop=True)
        # rest neg data will remain negative
        neg200 = neg200.loc[neg_idx[len(pos0):]].reset_index(drop=True)

        print('[INFO] Mixing data...')
        mixed_pos = []
        for idx, entry in tqdm(pos0.iterrows(), total=len(pos0)):
            hit_idx = None
            for k, v in entry.items():
                if 'gen' in k:  # directly keep gen variables
                    continue
                elif isinstance(v, int):  # accumulate n_mu_hit
                    assert k == 'n_mu_hit'
                    entry[k] += noise_in_pos.iloc[idx][k]
                    hit_idx = np.arange(0, entry['n_mu_hit'])
                    np.random.shuffle(hit_idx)
                else:  # concat hit features
                    assert isinstance(v, np.ndarray)
                    assert hit_idx is not None
                    mixed_hits = np.concatenate((noise_in_pos.iloc[idx][k], v))
                    entry[k] = mixed_hits[hit_idx]  # shuffle hit order
            mixed_pos.append(entry.values)
        mixed_pos = pd.DataFrame(data=mixed_pos, columns=entry.index)
        return neg200, pos200, mixed_pos

    @staticmethod
    def get_idx_split(df, splits, pos2neg, run_type, random_state):
        assert sum(splits.values()) == 1.0

        y_dist = df['y'].to_numpy()

        neg140_idx = np.argwhere(y_dist == -2).reshape(-1)
        neg250_idx = np.argwhere(y_dist == -1).reshape(-1)
        neg200_idx = np.argwhere(y_dist == 0).reshape(-1)
        pos200_idx = np.argwhere(y_dist == 1).reshape(-1)
        pos0_idx = np.argwhere(y_dist == 2).reshape(-1)
        mixed_pos_idx = np.argwhere(y_dist == 3).reshape(-1)

        np.random.seed(random_state)
        np.random.shuffle(neg140_idx)
        np.random.shuffle(neg250_idx)
        np.random.shuffle(neg200_idx)
        np.random.shuffle(pos200_idx)
        np.random.shuffle(pos0_idx)
        np.random.shuffle(mixed_pos_idx)

        if run_type == 'reg':
            # train/val/test on pos200 & neg200
            assert len(pos200_idx) <= len(neg200_idx) * pos2neg
            assert splits['test'] != 0

            n_train_pos, n_valid_pos = int(splits['train'] * len(pos200_idx)), int(splits['valid'] * len(pos200_idx))
            n_train_neg, n_valid_neg = int(n_train_pos / pos2neg), int(n_valid_pos / pos2neg)

            pos_train_idx = pos200_idx[0:n_train_pos]
            pos_valid_idx = pos200_idx[n_train_pos:n_train_pos + n_valid_pos]
            pos_test_idx = pos200_idx[n_train_pos + n_valid_pos:]

            neg_train_idx = neg200_idx[0:n_train_neg]
            neg_valid_idx = neg200_idx[n_train_neg:n_train_neg + n_valid_neg]
            neg_test_idx = neg200_idx[n_train_neg + n_valid_neg:]

        elif run_type == 'mix':
            # train/val on mixed_pos & neg200, test on pos200 & neg200
            assert len(mixed_pos_idx) <= len(neg200_idx) * pos2neg
            assert splits['test'] == 0

            n_train_pos, n_valid_pos = int(splits['train'] * len(mixed_pos_idx)), int(splits['valid'] * len(mixed_pos_idx))
            n_train_neg, n_valid_neg = int(n_train_pos / pos2neg), int(n_valid_pos / pos2neg)

            pos_train_idx = mixed_pos_idx[0:n_train_pos]
            pos_valid_idx = mixed_pos_idx[n_train_pos:]
            pos_test_idx = pos200_idx

            neg_train_idx = neg200_idx[0:n_train_neg]
            neg_valid_idx = neg200_idx[n_train_neg:n_train_neg + n_valid_neg]
            neg_test_idx = neg200_idx[n_train_neg + n_valid_neg:]

        elif run_type == 'check':
            # train/test/validate on mixed_pos & pos200
            assert splits['test'] != 0
            mixed_pos_idx = neg200_idx

            n_train_pos, n_valid_pos = int(splits['train'] * len(pos200_idx)), int(splits['valid'] * len(pos200_idx))
            n_train_neg, n_valid_neg = int(splits['train'] * len(mixed_pos_idx)), int(splits['valid'] * len(mixed_pos_idx))

            pos_train_idx = pos200_idx[0:n_train_pos]
            pos_valid_idx = pos200_idx[n_train_pos:n_train_pos + n_valid_pos]
            pos_test_idx = pos200_idx[n_train_pos + n_valid_pos:]

            neg_train_idx = mixed_pos_idx[0:n_train_neg]
            neg_valid_idx = mixed_pos_idx[n_train_neg:n_train_neg + n_valid_neg]
            neg_test_idx = mixed_pos_idx[n_train_neg + n_valid_neg:]
        elif run_type == 'da_eval':
            # train_loader: test_pos200 & test_neg200
            # valid_loader: pos0 & test_neg200
            # test_loader:  test_pos200 & neg140

            pos_train_idx = pos200_idx
            pos_valid_idx = pos0_idx
            pos_test_idx = pos200_idx

            neg_train_idx = neg200_idx
            neg_valid_idx = neg200_idx
            neg_test_idx = neg140_idx
        elif run_type == 'regress':
            pos0_idx = np.argwhere(df['type'].to_numpy() == 'pos0').reshape(-1)
            np.random.shuffle(pos0_idx)

            n_train_pos, n_valid_pos = int(splits['train'] * len(pos0_idx)), int(splits['valid'] * len(pos0_idx))

            pos_train_idx = pos0_idx[0:n_train_pos-1]
            pos_valid_idx = pos0_idx[n_train_pos:n_train_pos + n_valid_pos-1]
            pos_test_idx = pos0_idx[n_train_pos + n_valid_pos:-1]

            neg_train_idx = pos0_idx[[n_train_pos]]
            neg_valid_idx = pos0_idx[[n_train_pos + n_valid_pos]]
            neg_test_idx = pos0_idx[[-1]]

        else:
            raise NotImplementedError

        return {'train': np.concatenate((pos_train_idx, neg_train_idx)).tolist(),
                'valid': np.concatenate((pos_valid_idx, neg_valid_idx)).tolist(),
                'test': np.concatenate((pos_test_idx, neg_test_idx)).tolist()}

    @staticmethod
    def z_score(neg200, node_feature_names):

        def hit_filter(x, fn):
            idx = (x[fn] != -99) * (x['mu_hit_station'] == 1)
            if idx.sum() == 0:
                return np.nan
            else:
                return x[fn][idx]

        def f(fn):
            v = neg200.apply(lambda x: hit_filter(x, fn), axis=1)
            v = v[v.notna()]
            values = [each_hit for each_graph in v for each_hit in each_graph]
            mu, sigma = np.mean(values), np.std(values)

            node_feature_stat[fn]['mu'] = mu
            node_feature_stat[fn]['sigma'] = sigma

        node_feature_stat = {fn: {'mu': None, 'sigma': None} for fn in node_feature_names if 'ring' not in fn}
        list(map(f, tqdm(node_feature_stat)))
        return node_feature_stat

    @staticmethod
    def z_score_with_missing_values(entry: dict, node_feature_stat: dict):
        for fn, stat in node_feature_stat.items():
            missing_value_idx = entry[fn] == -99
            entry[fn] = (entry[fn] - stat['mu']) / stat['sigma']
            entry[fn][missing_value_idx] = 0
        return entry


if __name__ == '__main__':
    import os
    os.chdir('../')

    configs = Path('./configs')
    for cfg in configs.iterdir():
        cfg_dict = yaml.safe_load(cfg.open('r'))
        dataset = Tau3MuDataset(cfg_dict['data'])
