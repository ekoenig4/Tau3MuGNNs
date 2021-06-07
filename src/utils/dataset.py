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
        self.pos2neg = config['pos2neg']
        self.random_state = config['random_state']
        self.mix_samples = config['mix_samples']
        self.mix_and_check = config['mix_and_check']
        self.pred_pt = config['pred_pt']
        self.visz = config['visz']
        self.filter_soft_mu = config['filter_soft_mu']
        self.da_test = config['da_test']
        self.normalize = config['normalize']
        self.one_hot = config['one_hot']

        if (self.mix_samples and not self.mix_and_check) or self.da_test:
            assert self.splits['test'] == 0

        super(Tau3MuDataset, self).__init__(root=self.data_dir)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.shape[-1]
        self.edge_attr_dim = max(self.data.intra_level_edge_features.shape[-1],
                                 self.data.inter_level_edge_features.shape[-1])
        self.all_entries = torch.load(Path(self.processed_dir) / 'all_entries.pt') if self.visz else None

    @property
    def raw_file_names(self):
        return ['DsTau3muPU0_Private.pkl', 'DsTau3muPU200_MTD.pkl', 'MinBiasPU200_MTD.pkl', 'MinBiasPU250_MTD.pkl']

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
        del dfs

        data_list, y_dist = [], []
        all_entries = []
        entry_index = None
        print('[INFO] Processing entries...')
        for idx, entry in tqdm(df.iterrows(), total=len(df)):

            entry = Tau3MuDataset.filter_hits(entry, self.conditions)
            if entry is None:
                continue
            elif self.visz:
                all_entries.append(entry.values)
                if entry_index is None:
                    entry_index = entry.index

            x, score_gt = Tau3MuDataset.get_node_features(entry, self.node_feature_names, self.virtual_node, self.mix_samples, self.one_hot)
            y_dist.append(entry.y)

            if self.pred_pt:
                y = torch.tensor(np.concatenate([[0 if entry.y <= 0 else 1], entry.gen_mu_pt]), dtype=torch.float).view(-1, 4)
            elif self.mix_and_check:
                y = torch.tensor(1 if entry.y == 1 else 0, dtype=torch.float).view(-1, 1)
            else:
                y = torch.tensor(0 if entry.y <= 0 else 1, dtype=torch.float).view(-1, 1)

            intra_level_edges, inter_level_edges, virtual_edges = Tau3MuDataset.build_graph(entry, self.add_self_loops)

            intra_level_edge_features = Tau3MuDataset.get_edge_features(entry, intra_level_edges, self.edge_feature_names,
                                                                        for_virtual_edges=False)
            inter_level_edge_features = Tau3MuDataset.get_edge_features(entry, inter_level_edges, self.edge_feature_names,
                                                                        for_virtual_edges=False)
            virtual_edge_features = Tau3MuDataset.get_edge_features(entry, virtual_edges, self.edge_feature_names,
                                                                    for_virtual_edges=True)
            if not self.virtual_node:
                virtual_edge_features = virtual_edges = torch.tensor([], dtype=torch.long)

            data_list.append(Data(x=x, y=y, num_nodes=x.shape[0], score_gt=score_gt,
                                  intra_level_edge_index=intra_level_edges,
                                  inter_level_edge_index=inter_level_edges,
                                  virtual_edge_index=virtual_edges,
                                  intra_level_edge_features=intra_level_edge_features,
                                  inter_level_edge_features=inter_level_edge_features,
                                  virtual_edge_features=virtual_edge_features))
        data, slices = self.collate(data_list)
        del data_list, df

        idx_split = self.get_idx_split(np.array(y_dist))

        print('[INFO] Saving data.pt...')
        torch.save((data, slices, idx_split), self.processed_paths[0])
        yaml.dump(self.config, open(Path(self.processed_dir) / 'data_config.yml', 'w'))
        (Path(self.processed_dir) / 'data_md5.txt').open('w').write(Tau3MuDataset.md5sum(data, slices, idx_split, True))

        if self.visz:
            del data
            print('[INFO] Saving all_entries.pt...')
            all_entries = pd.DataFrame(data=all_entries, columns=entry_index)
            torch.save(all_entries, Path(self.processed_dir) / 'all_entries.pt')

    @staticmethod
    def filter_hits(entry: pd.Series, conditions: dict) -> Union[pd.Series, None]:
        for k, v in conditions.items():
            k = k.split('-')[1]
            if isinstance(entry[k], np.ndarray):
                mask = np.argwhere(eval('entry.' + k + v))
                entry.n_mu_hit = mask.shape[0]
                if entry.n_mu_hit == 0:
                    return None
                for key, value in entry.items():
                    if isinstance(value, np.ndarray) and 'gen' not in key:
                        entry[key] = value[mask].reshape(-1)
            else:
                if not eval('entry.' + k + v):
                    return None
        return entry

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
    def get_node_features(entry, feature_names, virtual_node, mix_samples, one_hot) -> Tuple:
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
        if mix_samples and entry['score_gt'] is not None:
            if virtual_node:
                entry['score_gt'] = np.concatenate((entry['score_gt'], [1]))
            # TODO: when there is no virtual node, the denominator will be zero.
            assert virtual_node
            score_gt = torch.tensor(entry['score_gt'] / entry['score_gt'].sum(), dtype=torch.float)

        return torch.tensor(features, dtype=torch.float), score_gt

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
        neg250 = dfs['MinBiasPU250_MTD']
        neg200 = dfs['MinBiasPU200_MTD']
        pos0 = dfs['DsTau3muPU0_Private']
        pos200 = dfs['DsTau3muPU200_MTD']

        if self.filter_soft_mu:
            pos0 = pos0[pos0.apply(lambda x: Tau3MuDataset.filter_mu_by_pt_eta(x), axis=1)].reset_index(drop=True)
            pos200 = pos200[pos200.apply(lambda x: Tau3MuDataset.filter_mu_by_pt_eta(x), axis=1)].reset_index(drop=True)
        if self.only_one_tau:
            pos0 = pos0[pos0.n_gen_tau == 1].reset_index(drop=True)
            pos200 = pos200[pos200.n_gen_tau == 1].reset_index(drop=True)

        if self.normalize:
            print('[INFO] Normalizing features...')
            assert self.conditions['1-mu_hit_station'] == '==1'
            assert len(self.node_feature_names) == 8
            pos0, pos200, neg200, neg250 = Tau3MuDataset.z_score(pos0, pos200, neg200, neg250, self.node_feature_names)

        if self.mix_samples:
            return Tau3MuDataset.get_mixed_df(pos0, pos200, neg200, self.random_state, self.visz)
        else:
            return Tau3MuDataset.add_labels_and_merge_dfs(pos0, pos200, neg200, neg250, self.train_with_noise,
                                                          self.pred_pt, self.visz, self.da_test)

    @staticmethod
    def add_labels_and_merge_dfs(pos0, pos200, neg200, neg250, train_with_noise, pred_pt, visz, da_test) -> pd.DataFrame:
        pos = pos200 if train_with_noise else pos0

        neg250['y'], neg200['y'], pos['y'] = -1, 0, 1

        if pred_pt:
            pos['gen_mu_pt'] = pos.apply(lambda x: np.sort(x['gen_mu_pt']), axis=1)
            neg200['gen_mu_pt'] = neg200.apply(lambda x: np.full(3, -2), axis=1)
            neg250['gen_mu_pt'] = neg250.apply(lambda x: np.full(3, -2), axis=1)

        if visz:
            assert train_with_noise
            pos0['y'] = 2
            return pd.concat((pos, neg200, pos0, neg250), join='outer', ignore_index=True)
        elif da_test:
            assert train_with_noise
            pos0['y'] = 2
            return pd.concat((pos, neg200, pos0, neg250), join='inner', ignore_index=True)
        else:
            return pd.concat((pos, neg200), join='inner', ignore_index=True)

    @staticmethod
    def get_mixed_df(pos0, pos200, neg200, random_state, visz):
        pos200['score_gt'] = None

        neg_idx = np.arange(len(neg200))
        np.random.seed(random_state)
        np.random.shuffle(neg_idx)

        noise_in_mixed_pos = neg200.iloc[neg_idx[:len(pos0)]].reset_index(drop=True)
        pure_neg = neg200.iloc[neg_idx[len(pos0):]].reset_index(drop=True)
        pure_neg['score_gt'] = pure_neg.apply(lambda x: np.zeros(x['n_mu_hit']), axis=1)

        print('[INFO] Mixing data...')
        mixed_pos = []
        for idx, entry in tqdm(pos0.iterrows(), total=len(pos0)):
            hit_idx = None
            for k, v in entry.items():
                if 'gen' in k:
                    continue
                elif isinstance(v, int):
                    assert k == 'n_mu_hit'
                    entry[k] += noise_in_mixed_pos.iloc[idx][k]
                    entry['score_gt'] = np.ones(entry[k])
                    entry['score_gt'][:noise_in_mixed_pos.iloc[idx][k]] = 0
                    hit_idx = np.arange(0, entry['n_mu_hit'])
                    np.random.shuffle(hit_idx)
                else:
                    assert isinstance(v, np.ndarray)
                    assert hit_idx is not None
                    mixed_hits = np.concatenate((noise_in_mixed_pos.iloc[idx][k], v))
                    entry[k] = mixed_hits[hit_idx]

            mixed_pos.append(entry.values)
        mixed_pos = pd.DataFrame(data=mixed_pos, columns=entry.index)

        pure_neg['y'], pos200['y'], mixed_pos['y'] = 0, 1, 2

        if visz:
            return pd.concat((mixed_pos, pure_neg, pos200), join='outer', ignore_index=True)
        else:
            return pd.concat((mixed_pos, pure_neg, pos200), join='inner', ignore_index=True)

    def get_idx_split(self, y_dist):
        if self.mix_samples:
            return Tau3MuDataset.get_mixed_idx_split(y_dist, self.splits, self.pos2neg, self.random_state, self.mix_and_check)
        elif self.da_test:
            return Tau3MuDataset.get_da_idx_split(y_dist, self.splits, self.pos2neg, self.random_state)
        else:
            return Tau3MuDataset.get_pu200_idx_split(y_dist, self.splits, self.pos2neg, self.random_state)

    @staticmethod
    def get_pu200_idx_split(y_dist, splits: dict, pos2neg: float, random_state: int) -> dict:
        assert sum(splits.values()) == 1.0

        pos_idx = np.argwhere(y_dist == 1).reshape(-1)
        neg_idx = np.argwhere(y_dist == 0).reshape(-1)
        assert len(pos_idx) < len(neg_idx)
        assert len(pos_idx) <= len(neg_idx) * pos2neg

        np.random.seed(random_state)
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        n_train_pos, n_valid_pos = int(splits['train'] * len(pos_idx)), int(splits['valid'] * len(pos_idx))
        n_train_neg, n_valid_neg = int(n_train_pos / pos2neg), int(n_valid_pos / pos2neg)

        # len(pos_train_idx) : len(neg_train_idx) = len(pos_valid_idx) : len(neg_valid_idx) = pos2neg
        # len(pos_test_idx) << len(neg_test_idx)
        pos_train_idx = pos_idx[0:n_train_pos]
        pos_valid_idx = pos_idx[n_train_pos:n_train_pos + n_valid_pos]
        pos_test_idx = pos_idx[n_train_pos + n_valid_pos:]

        neg_train_idx = neg_idx[0:n_train_neg]
        neg_valid_idx = neg_idx[n_train_neg:n_train_neg + n_valid_neg]
        neg_test_idx = neg_idx[n_train_neg + n_valid_neg:]

        return {'train': np.concatenate((pos_train_idx, neg_train_idx)).tolist(),
                'valid': np.concatenate((pos_valid_idx, neg_valid_idx)).tolist(),
                'test': np.concatenate((pos_test_idx, neg_test_idx)).tolist()}

    @staticmethod
    def get_mixed_idx_split(y_dist, splits: dict, pos2neg: float, random_state: int, mix_and_check: bool):
        assert sum(splits.values()) == 1.0

        pure_neg_idx = np.argwhere(y_dist == 0).reshape(-1)
        pos200_idx = np.argwhere(y_dist == 1).reshape(-1)
        mixed_pos_idx = np.argwhere(y_dist == 2).reshape(-1)

        np.random.seed(random_state)
        np.random.shuffle(mixed_pos_idx)
        np.random.shuffle(pure_neg_idx)
        np.random.shuffle(pos200_idx)

        if mix_and_check:
            # train/validate/test on mixed_samples & pu200
            n_train_pos, n_valid_pos = int(splits['train'] * len(pos200_idx)), int(splits['valid'] * len(pos200_idx))
            n_train_neg, n_valid_neg = int(splits['train'] * len(mixed_pos_idx)), int(splits['valid'] * len(mixed_pos_idx))

            pos_train_idx = pos200_idx[0:n_train_pos]
            pos_valid_idx = pos200_idx[n_train_pos:n_train_pos + n_valid_pos]
            pos_test_idx = pos200_idx[n_train_pos+n_valid_pos:]

            neg_train_idx = mixed_pos_idx[0:n_train_neg]
            neg_valid_idx = mixed_pos_idx[n_train_neg:n_train_neg + n_valid_neg]
            neg_test_idx = mixed_pos_idx[n_train_neg + n_valid_neg:]

        else:
            # train/validate on mixed_samples & neg200, test on pu200 & neg200
            assert len(mixed_pos_idx) <= len(pure_neg_idx) * pos2neg
            n_train_pos, n_valid_pos = int(splits['train'] * len(mixed_pos_idx)), int( splits['valid'] * len(mixed_pos_idx))
            n_train_neg, n_valid_neg = int(n_train_pos / pos2neg), int(n_valid_pos / pos2neg)

            pos_train_idx = mixed_pos_idx[0:n_train_pos]
            pos_valid_idx = mixed_pos_idx[n_train_pos:]
            pos_test_idx = pos200_idx

            neg_train_idx = pure_neg_idx[0:n_train_neg]
            neg_valid_idx = pure_neg_idx[n_train_neg:n_train_neg + n_valid_neg]
            neg_test_idx = pure_neg_idx[n_train_neg + n_valid_neg:]

        return {'train': np.concatenate((pos_train_idx, neg_train_idx)).tolist(),
                'valid': np.concatenate((pos_valid_idx, neg_valid_idx)).tolist(),
                'test': np.concatenate((pos_test_idx, neg_test_idx)).tolist()}

    @staticmethod
    def get_da_idx_split(y_dist, splits: dict, pos2neg: float, random_state: int):
        assert sum(splits.values()) == 1.0

        neg250_idx = np.argwhere(y_dist == -1).reshape(-1)
        neg200_idx = np.argwhere(y_dist == 0).reshape(-1)
        pos200_idx = np.argwhere(y_dist == 1).reshape(-1)
        pos0_idx = np.argwhere(y_dist == 2).reshape(-1)

        np.random.seed(random_state)
        np.random.shuffle(neg250_idx)
        np.random.shuffle(neg200_idx)
        np.random.shuffle(pos200_idx)
        np.random.shuffle(pos0_idx)

        # train/validate on pos200 & neg200, test on pu0 & neg250
        n_train_pos, n_valid_pos = int(splits['train'] * len(pos200_idx)), int(splits['valid'] * len(pos200_idx))
        n_train_neg, n_valid_neg = int(n_train_pos / pos2neg), int(n_valid_pos / pos2neg)

        # pos_train_idx = pos200_idx[0:n_train_pos]
        # pos_valid_idx = pos200_idx[n_train_pos:]
        # pos_test_idx = pos0_idx
        #
        # neg_train_idx = neg200_idx[0:n_train_neg]
        # neg_valid_idx = neg200_idx[n_train_neg:]
        # neg_test_idx = neg250_idx

        pos_train_idx = pos200_idx[0:n_train_pos]
        pos_valid_idx = pos200_idx[n_train_pos:]
        pos_test_idx = pos0_idx

        neg_train_idx = neg200_idx[0:n_train_neg]
        neg_valid_idx = neg250_idx
        neg_test_idx = neg200_idx[n_train_neg:]

        return {'train': np.concatenate((pos_train_idx, neg_train_idx)).tolist(),
                'valid': np.concatenate((pos_valid_idx, neg_valid_idx)).tolist(),
                'test': np.concatenate((pos_test_idx, neg_test_idx)).tolist()}

    @staticmethod
    def z_score(pos0, pos200, neg200, neg250, node_feature_names):

        def hit_filter(x, fn):
            idx = (x[fn] != -99) * (x['mu_hit_station'] == 1)
            if idx.sum() == 0:
                return np.nan
            else:
                return x[fn][idx]

        def z_score_with_missing_values(x, fn, mu, sigma):
            missing_value_idx = x[fn] == -99
            normalized_data = (x[fn] - mu) / sigma
            normalized_data[missing_value_idx] = 0
            return normalized_data

        def f(fn):
            v = neg200.apply(lambda x: hit_filter(x, fn), axis=1)
            v = v[v.notna()]
            values = [each_hit for each_graph in v for each_hit in each_graph]
            mu, sigma = np.mean(values), np.std(values)

            neg200[fn] = neg200.apply(lambda x: z_score_with_missing_values(x, fn, mu, sigma), axis=1)
            pos0[fn] = pos0.apply(lambda x: z_score_with_missing_values(x, fn, mu, sigma), axis=1)
            pos200[fn] = pos200.apply(lambda x: z_score_with_missing_values(x, fn, mu, sigma), axis=1)
            neg250[fn] = neg250.apply(lambda x: z_score_with_missing_values(x, fn, mu, sigma), axis=1)

        node_feature_names = [fn for fn in node_feature_names if 'ring' not in fn]
        list(map(f, tqdm(node_feature_names)))
        return pos0, pos200, neg200, neg250


if __name__ == '__main__':
    import os
    os.chdir('../')

    configs = Path('./configs')
    for cfg in configs.iterdir():
        cfg_dict = yaml.safe_load(cfg.open('r'))
        dataset = Tau3MuDataset(cfg_dict['data'])
