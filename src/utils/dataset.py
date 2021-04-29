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
from torch_geometric.data import Data, InMemoryDataset

from torch import Tensor
from pathlib import Path
from typing import List, Tuple, Any, Union


class Tau3MuDataset(InMemoryDataset):
    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.conditions = config['conditions']
        self.train_with_noise = config['train_with_noise']
        self.add_self_loops = config['add_self_loops']
        self.node_feature_names = config['node_feature_names']
        self.edge_feature_names = config['edge_feature_names']
        self.virtual_node = config['virtual_node']
        self.only_one_tau = config['only_one_tau']
        self.splits = config['splits']
        self.random_state = config['random_state']

        super(Tau3MuDataset, self).__init__(root=self.data_dir)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.shape[-1]
        self.edge_attr_dim = max(self.data.intra_level_edge_features.shape[-1],
                                 self.data.inter_level_edge_features.shape[-1])

    @property
    def raw_file_names(self):
        return ['DsTau3muPU0_Private.pkl', 'DsTau3muPU200_MTD.pkl', 'MinBiasPU200_MTD.pkl']

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
        df = self.add_labels_and_merge_dfs(dfs, self.train_with_noise, self.only_one_tau)

        data_list = []
        for idx, entry in tqdm(df.iterrows(), total=len(df)):

            entry = Tau3MuDataset.filter_hits(entry, self.conditions)
            if entry is None:
                continue

            x = self.get_node_features(entry, self.node_feature_names, self.virtual_node)
            y = torch.tensor(entry.y, dtype=torch.float).view(-1, 1)

            intra_level_edges, inter_level_edges, virtual_edges = self.build_graph(entry, self.add_self_loops)

            intra_level_edge_features = self.get_edge_features(entry, intra_level_edges, self.edge_feature_names,
                                                               for_virtual_edges=False)
            inter_level_edge_features = self.get_edge_features(entry, inter_level_edges, self.edge_feature_names,
                                                               for_virtual_edges=False)
            virtual_edge_features = self.get_edge_features(entry, virtual_edges, self.edge_feature_names,
                                                           for_virtual_edges=True)
            if not self.virtual_node:
                virtual_edge_features = virtual_edges = torch.tensor([], dtype=torch.long)

            data_list.append(Data(x=x, y=y, num_nodes=x.shape[0],
                                  intra_level_edge_index=intra_level_edges,
                                  inter_level_edge_index=inter_level_edges,
                                  virtual_edge_index=virtual_edges,
                                  intra_level_edge_features=intra_level_edge_features,
                                  inter_level_edge_features=inter_level_edge_features,
                                  virtual_edge_features=virtual_edge_features))

        data, slices = self.collate(data_list)
        idx_split = Tau3MuDataset.get_idx_split(data, self.splits, self.random_state)

        print('[INFO] Saving data.pt...')
        torch.save((data, slices, idx_split), self.processed_paths[0])
        yaml.dump(self.config, open(Path(self.processed_dir) / 'data_config.yml', 'w'))
        (Path(self.processed_dir) / 'data_md5.txt').open('w').write(Tau3MuDataset.md5sum(data, slices, idx_split))

    @staticmethod
    def filter_hits(entry: pd.Series, conditions: dict) -> Union[pd.Series, None]:
        for k, v in conditions.items():
            if isinstance(entry[k], np.ndarray):
                mask = np.argwhere(eval('entry.' + k + v))
                entry.n_mu_hit = mask.shape[0]
                if entry.n_mu_hit == 0:
                    return None
                for idx, each in enumerate(entry):
                    if isinstance(each, np.ndarray):
                        entry[idx] = each[mask].reshape(-1)
            else:
                if not eval('entry.' + k + v):
                    return None
        return entry

    @staticmethod
    def get_idx_split(data, splits: dict, random_state) -> dict:
        assert sum(splits.values()) == 1.0

        pos_idx = np.argwhere(data.y.reshape(-1).numpy() == 1).reshape(-1)
        neg_idx = np.argwhere(data.y.reshape(-1).numpy() == 0).reshape(-1)

        np.random.seed(random_state)
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        i = int(splits['train'] * len(pos_idx))
        j = int(splits['valid'] * len(pos_idx))

        # len(pos_train_idx) : len(neg_train_idx) = len(pos_valid_idx) : len(neg_valid_idx) = 1 : 1
        # len(pos_test_idx) << len(neg_test_idx)
        pos_train_idx, pos_valid_idx, pos_test_idx = pos_idx[0:i], pos_idx[i:i + j], pos_idx[i + j:-1]
        neg_train_idx, neg_valid_idx, neg_test_idx = neg_idx[0:i], neg_idx[i:i + j], neg_idx[i + j:-1]

        return {'train': np.concatenate((pos_train_idx, neg_train_idx)).tolist(),
                'valid': np.concatenate((pos_valid_idx, neg_valid_idx)).tolist(),
                'test': np.concatenate((pos_test_idx, neg_test_idx)).tolist()}

    @staticmethod
    def add_labels_and_merge_dfs(dfs: dict, train_with_noise: bool, only_one_tau: bool) -> pd.DataFrame:
        neg = dfs['MinBiasPU200_MTD']
        pos0 = dfs['DsTau3muPU0_Private']
        pos200 = dfs['DsTau3muPU200_MTD']

        pos = pos200 if train_with_noise else pos0

        neg['y'] = 0
        pos['y'] = 1
        if only_one_tau:
            pos = pos[pos.n_gen_tau == 1]

        return pd.concat((pos, neg), join='inner', ignore_index=True)

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
    def get_node_features(entry: pd.Series, feature_names: List[str], virtual_node: bool) -> Tensor:
        # one-hot encoding for station ids? Perhaps the order of the station matters.
        features = np.stack([entry[feature] for feature in feature_names], axis=1)
        if virtual_node:
            features = np.concatenate((features, np.zeros((1, len(feature_names)))))
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def get_edge_features(entry: pd.Series, edges: Tensor, feature_names: List[str], for_virtual_edges: bool) -> Tensor:
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
    def build_graph(entry: pd.Series, add_self_loops: bool) -> List[Tensor]:
        dim = entry.n_mu_hit
        virtual_node_id = dim
        real_node_ids = [i for i in range(dim)]

        station2hitids = Tau3MuDataset.groupby_station(entry.mu_hit_station)

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
    def md5sum(data, slices, idx_split):
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
        return m.hexdigest()
