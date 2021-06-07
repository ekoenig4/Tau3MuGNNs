# -*- coding: utf-8 -*-

"""
Created on 2021/6/6

@author: Siqi Miao
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn import metrics
from datetime import datetime
from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, InMemoryDataset, DataLoader


class Tau3MuDataset(InMemoryDataset):

    def __init__(self, data_dir, df_names, run_type, node_feature_names, edge_feature_names, filter_soft_mu):
        self.data_dir = data_dir
        self.df_names = df_names
        self.run_type = run_type
        self.node_feature_names = node_feature_names
        self.edge_feature_names = edge_feature_names
        self.filter_soft_mu = filter_soft_mu

        super(Tau3MuDataset, self).__init__(root=self.data_dir)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])

        self.x_dim = self.data.x.shape[-1]
        self.edge_attr_dim = self.data.edge_attr.shape[-1]

    @property
    def raw_file_names(self):
        return ['DsTau3muPU0_Private.pkl', 'DsTau3muPU200_MTD.pkl', 'MinBiasPU200_MTD.pkl', 'MinBiasPU250_MTD.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        print('Please put .pkl files ino $PROJECT_DIR/data/raw!')
        raise KeyboardInterrupt

    def process(self):
        df_dict = Tau3MuDataset.read_df(self.data_dir, self.df_names)
        df = self.get_df(df_dict, self.run_type, self.node_feature_names, self.filter_soft_mu)
        del df_dict

        data_list, y_dist,  = [], []
        all_entries, entry_index = [], None
        print('[INFO] Processing entries...')
        for idx, entry in tqdm(df.iterrows(), total=len(df)):

            entry = Tau3MuDataset.filter_hits(entry)
            if entry is None:
                continue
            all_entries.append(entry.values)
            if entry_index is None:
                entry_index = entry.index

            x = Tau3MuDataset.get_node_features(entry, self.node_feature_names)
            y_dist.append(entry.y)
            y = torch.tensor(0 if entry.y <= 0 else 1, dtype=torch.float).view(-1, 1)

            edge_index = Tau3MuDataset.build_graph(entry)
            edge_features = Tau3MuDataset.get_edge_features(entry, edge_index, self.edge_feature_names)
            data_list.append(Data(x=x, y=y, num_nodes=x.shape[0], edge_index=edge_index, edge_attr=edge_features))

        data, slices = self.collate(data_list)
        idx_split = Tau3MuDataset.get_idx_split(np.array(y_dist), self.run_type)
        print('[INFO] Saving data.pt...')
        torch.save((data, slices, idx_split), self.processed_paths[0])

        del data, df
        print('[INFO] Saving all_entries.pt...')
        all_entries = pd.DataFrame(data=all_entries, columns=entry_index)  # may consume lots of memory
        torch.save(all_entries, Path(self.processed_dir) / 'all_entries.pt')

    @staticmethod
    def read_df(data_dir, df_names):
        df_dict = {}
        print('[INFO] Loading .pkl...')
        for df_name in tqdm(df_names):
            df_path = data_dir / 'raw' / df_name
            df_dict[df_path.stem] = pd.read_pickle(df_path)
        return df_dict

    @staticmethod
    def mix(pos: pd.DataFrame, neg: pd.DataFrame):
        np.random.seed(1)
        neg_idx = np.arange(0, len(neg))
        np.random.shuffle(neg_idx)

        # first len(pos) neg data will be used as noise in pos0
        noise_in_pos = neg.loc[neg_idx[:len(pos)]].reset_index(drop=True)
        # rest neg data will remain negative
        neg = neg.loc[neg_idx[len(pos):]].reset_index(drop=True)

        mixed_pos = []
        print('[INFO] Mixing...')
        for idx, entry in tqdm(pos.iterrows(), total=len(pos)):
            hit_idx = None
            for k, v in entry.items():
                if 'gen' in k:  # directly keep gen variables
                    continue
                elif isinstance(v, int):  # accumulate n_mu_hit
                    assert k == 'n_mu_hit'
                    entry['n_mu_hit'] += noise_in_pos.loc[idx]['n_mu_hit']
                    hit_idx = np.arange(0, entry['n_mu_hit'])
                    np.random.shuffle(hit_idx)
                else:  # concat hit features
                    assert isinstance(v, np.ndarray)
                    assert hit_idx is not None
                    mixed_hits = np.concatenate((noise_in_pos.loc[idx][k], v))
                    entry[k] = mixed_hits[hit_idx]  # shuffle hit order
            mixed_pos.append(entry.values)
        mixed_pos = pd.DataFrame(data=mixed_pos, columns=entry.index)
        return mixed_pos, neg

    @staticmethod
    def get_df(df_dict, run_type, node_feature_names, filter_soft_mu) -> pd.DataFrame:
        # If memory is limited, can only use a part of the data.
        # neg250 = df_dict['MinBiasPU250_MTD'].loc[:100000].reset_index(drop=True)
        # neg200 = df_dict['MinBiasPU200_MTD'].loc[:100000].reset_index(drop=True)
        # pos0 = df_dict['DsTau3muPU0_Private'].loc[:20000].reset_index(drop=True)
        # pos200 = df_dict['DsTau3muPU200_MTD'].loc[:20000].reset_index(drop=True)

        for k, v in df_dict.items():
            for key in v.keys():
                if 'gen' not in key and key not in node_feature_names and key not in ['n_mu_hit', 'mu_hit_station']:
                    v.drop(key, inplace=True, axis=1)

        neg250 = df_dict['MinBiasPU250_MTD']
        neg200 = df_dict['MinBiasPU200_MTD']
        pos0 = df_dict['DsTau3muPU0_Private']
        pos200 = df_dict['DsTau3muPU200_MTD']

        if filter_soft_mu:
            def filter_mu_by_pt_eta(x):
                return ((x['gen_mu_pt'] > 0.5).sum() == 3) and ((abs(x['gen_mu_eta']) < 2.8).sum() == 3) and ((abs(x['gen_mu_eta']) > 1.2).sum() == 3)
            pos0 = pos0[pos0.apply(lambda x: filter_mu_by_pt_eta(x), axis=1)].reset_index(drop=True)
            pos200 = pos200[pos200.apply(lambda x: filter_mu_by_pt_eta(x), axis=1)].reset_index(drop=True)


        if run_type == 'normal':
            # train/val on pos200 & neg200, test on pos200 & neg250
            neg250['y'], neg200['y'], pos200['y'] = -1, 0, 1
            return pd.concat((neg250, neg200, pos200), join='outer', ignore_index=True)

        elif run_type == 'mixed':
            # train/val on mixed_pos & neg200, test on pos200 & neg200
            mixed_pos, neg200 = Tau3MuDataset.mix(pos0, neg200)
            neg200['y'], pos200['y'], mixed_pos['y'] = 0, 1, 2
            return pd.concat((neg200, pos200, mixed_pos), join='outer', ignore_index=True)

        elif run_type == 'check':
            # train/val/test on mixed_pos & pos200
            mixed_pos, neg200 = Tau3MuDataset.mix(pos0, neg200)
            mixed_pos['y'], pos200['y'] = 0, 1
            return pd.concat((mixed_pos, pos200), join='outer', ignore_index=True)
        else:
            raise NotImplementedError

    @staticmethod
    def get_idx_split(y_dist, run_type):
        neg250_idx = np.argwhere(y_dist == -1).reshape(-1)
        neg200_idx = np.argwhere(y_dist == 0).reshape(-1)
        pos200_idx = np.argwhere(y_dist == 1).reshape(-1)
        mixed_pos_idx = np.argwhere(y_dist == 2).reshape(-1)

        np.random.seed(1)
        np.random.shuffle(neg250_idx)
        np.random.shuffle(neg200_idx)
        np.random.shuffle(pos200_idx)
        np.random.shuffle(mixed_pos_idx)

        if run_type == 'normal':
            # train/val on pos200 & neg200, test on pos200 & neg250

            # positive data: 70% pos200 to train, 15% to validate, 15% to test
            n_train_pos, n_valid_pos = int(0.7 * len(pos200_idx)), int(0.15 * len(pos200_idx))
            # negative data: three times more than positive data
            n_train_neg, n_valid_neg = int(n_train_pos / 0.3), int(n_valid_pos / 0.3)

            pos_train_idx = pos200_idx[0:n_train_pos]
            pos_valid_idx = pos200_idx[n_train_pos:n_train_pos + n_valid_pos]
            pos_test_idx = pos200_idx[n_train_pos + n_valid_pos:]

            neg_train_idx = neg200_idx[0:n_train_neg]
            neg_valid_idx = neg200_idx[n_train_neg:]
            neg_test_idx = neg250_idx

        elif run_type == 'mixed':
            # train/val on mixed_pos & neg200, test on pos200 & neg200

            # positive data: 80% mixed_pos to train, 20% to validate, and pos200 is used to test
            n_train_pos, n_valid_pos = int(0.8 * len(mixed_pos_idx)), int(0.2 * len(mixed_pos_idx))
            # negative data: three times more than positive data
            n_train_neg, n_valid_neg = int(n_train_pos / 0.3), int(n_valid_pos / 0.3)

            pos_train_idx = mixed_pos_idx[0:n_train_pos]
            pos_valid_idx = mixed_pos_idx[n_train_pos:]
            pos_test_idx = pos200_idx

            neg_train_idx = neg200_idx[0:n_train_neg]
            neg_valid_idx = neg200_idx[n_train_neg:n_train_neg + n_valid_neg]
            neg_test_idx = neg200_idx[n_train_neg + n_valid_neg:]

        elif run_type == 'check':
            # train/test/validate on mixed_pos & pos200
            mixed_pos_idx = np.argwhere(y_dist == 0).reshape(-1)
            np.random.shuffle(mixed_pos_idx)

            # positive data: 70% pos200 to train, 15% to validate, 15% to test
            n_train_pos, n_valid_pos = int(0.7 * len(pos200_idx)), int(0.15 * len(pos200_idx))
            # negative data: 70% mixed_pos_idx to train, 15% to validate, 15% to test
            n_train_neg, n_valid_neg = int(0.7 * len(mixed_pos_idx)), int(0.15 * len(mixed_pos_idx))

            pos_train_idx = pos200_idx[0:n_train_pos]
            pos_valid_idx = pos200_idx[n_train_pos:n_train_pos + n_valid_pos]
            pos_test_idx = pos200_idx[n_train_pos + n_valid_pos:]

            neg_train_idx = mixed_pos_idx[0:n_train_neg]
            neg_valid_idx = mixed_pos_idx[n_train_neg:n_train_neg + n_valid_neg]
            neg_test_idx = mixed_pos_idx[n_train_neg + n_valid_neg:]
        else:
            raise NotImplementedError

        print('[Splits]:')
        print('    pos', pos_train_idx.shape[0], pos_valid_idx.shape[0], pos_test_idx.shape[0])
        print('    neg', neg_train_idx.shape[0], neg_valid_idx.shape[0], neg_test_idx.shape[0])

        return {'train': np.concatenate((pos_train_idx, neg_train_idx)).tolist(),
                'valid': np.concatenate((pos_valid_idx, neg_valid_idx)).tolist(),
                'test': np.concatenate((pos_test_idx, neg_test_idx)).tolist()}

    @staticmethod
    def build_graph(entry: pd.Series):
        dim = entry.n_mu_hit + 1
        node_ids = [i for i in range(dim)]
        edge_index = list(permutations(node_ids, 2))

        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)

        return edge_index

    @staticmethod
    def filter_hits(entry: pd.Series):
        if not entry['n_mu_hit'] >= 1:
            return None

        mask = np.argwhere(entry['mu_hit_station'] == 1)
        entry.n_mu_hit = mask.shape[0]
        if entry.n_mu_hit < 1:
            return None
        for key, value in entry.items():
            if isinstance(value, np.ndarray) and 'gen' not in key:
                entry[key] = value[mask].reshape(-1)  # mask out hits that are not on station 1
        return entry

    @staticmethod
    def get_node_features(entry, feature_names):
        features = np.stack([entry[feature] for feature in feature_names], axis=1)
        features = np.concatenate((features, np.zeros((1, features.shape[1]))))  # for virtual node
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def get_edge_features(entry: pd.Series, edges, feature_names):
        features = np.stack([entry[feature] for feature in feature_names], axis=1)
        features = np.concatenate((features, np.zeros((1, len(feature_names)))))  # for virtual node
        features = torch.tensor(features, dtype=torch.float)
        return features[edges[0]] - features[edges[1]]


class DeepGATConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int):

        super(DeepGATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.att_l = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, out_channels))
        self.leakyrelu = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.att_l)
        nn.init.kaiming_normal_(self.att_r)

    def forward(self, x, edge_index, edge_attr):

        alpha_l = (x * self.att_l).sum(dim=-1).view(-1, 1)  # N x 1
        alpha_r = (x * self.att_r).sum(dim=-1).view(-1, 1)  # N x 1

        out = self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r), edge_attr=edge_attr)  # N x C
        return out  # N x C

    def message(self, x_j, edge_attr, alpha_i, alpha_j, index):

        alpha = self.leakyrelu(alpha_i + alpha_j)  # E x 1
        alpha = softmax(alpha, index)  # E x 1
        msg = F.leaky_relu(x_j + edge_attr)
        return alpha * msg


class Model(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, n_layers, hidden_size, dropout_p):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.out_channels = hidden_size
        self.dropout_p = dropout_p

        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.leakyrelu = nn.LeakyReLU()

        self.node_encoder = nn.Linear(x_dim, self.out_channels)
        self.edge_encoder = nn.Linear(edge_attr_dim, self.out_channels)

        for i in range(self.n_layers):
            self.convs.append(DeepGATConv(self.out_channels, self.out_channels))
            self.mlps.append(MLP([self.out_channels, self.out_channels*2, self.out_channels], norm=nn.BatchNorm1d, dropout=self.dropout_p))

        self.lstm = nn.LSTMCell(self.out_channels, self.out_channels)
        self.fc_out = nn.Linear(self.out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = self.node_encoder(data.x)
        v_idx = Model.get_virtual_node_idx(x, data)
        edge_emb = self.edge_encoder(data.edge_attr)

        for i in range(self.n_layers):
            identity = x

            x = self.convs[i](x, data.edge_index, edge_emb)
            x = self.mlps[i](x)

            if i == 0:
                hx = identity[v_idx]
                cx = torch.zeros_like(identity[v_idx])
            hx, cx = self.lstm(x[v_idx], (hx, cx))

            x += identity

        out = self.fc_out(hx)
        out = self.sigmoid(out)
        return out

    @staticmethod
    def get_virtual_node_idx(x, data):
        idx = torch.tensor(data.__slices__['x'][1:]) - 1
        if x.is_cuda:
            idx = idx.cuda()
        return idx


class MLP(nn.Sequential):
    def __init__(self, channels, norm, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(norm(channels[i]))
                m.append(nn.LeakyReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


@torch.no_grad()
def eval_one_batch(data):
    model.eval()
    probs = model(data)
    loss = criterion(probs, data.y)
    return loss.item(), probs.data.cpu(), data.y.data.cpu()


def train_one_batch(data):
    model.train()
    optimizer.zero_grad()

    probs = model(data)
    loss = criterion(probs, data.y)

    loss.backward()
    optimizer.step()

    return loss.item(), probs.data.cpu(), data.y.data.cpu()


def run_one_epoch(data_loader, epoch, phase):
    loader_len = len(data_loader)
    all_probs, all_targets, all_batch_losses = [], [], []

    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
    phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

    pbar = tqdm(data_loader)
    for idx, data in enumerate(pbar):
        loss, probs, targets = run_one_batch(data.to(device))

        acc = ((probs > 0.5) == targets).sum() / len(targets)
        desc = f'[Epoch: {epoch}]: {phase}........., loss: {loss:.3f}, acc: {acc:.3f}..................................'

        all_probs.append(probs)
        all_targets.append(targets)
        all_batch_losses.append(loss)

        if idx == loader_len - 1:
            all_probs, all_targets = np.concatenate(all_probs), np.concatenate(all_targets)

            auroc = metrics.roc_auc_score(all_targets, all_probs)
            partial_auroc = metrics.roc_auc_score(all_targets, all_probs, max_fpr=0.001)
            acc = ((all_probs > 0.5) == all_targets).sum() / len(all_targets)
            desc = f'[Epoch: {epoch}]: {phase}........., loss: {np.mean(all_batch_losses):.3f}, acc: {acc:.3f}, ' \
                   f'auroc: {auroc:.3f}, partial_roc: {partial_auroc:.3f}'

        pbar.set_description(desc)


def save_checkpoint(epoch):
    torch.save({
        'epoch': epoch,
        'n_iters': (epoch + 1) * len(train_loader),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, log_path / 'model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='mixed')
    parser.add_argument('--filter_soft_mu', type=bool, default=False)

    args = parser.parse_args()
    print('[run_type]:', args.run_type)
    if args.run_type == 'normal':
        print('    train/val on pos200 & neg200, test on pos200 & neg250')
    elif args.run_type == 'mixed':
        print('    train/val on mixed_pos & neg200, test on pos200 & neg200')
    elif args.run_type == 'check':
        print('    train/test/validate on mixed_pos & pos200')
    else:
        assert NotImplementedError

    data_dir = Path('../data')
    log_path = Path('../logs') / (args.run_type + '-' + datetime.now().strftime("%m_%d_%Y-%H_%M_%S"))
    log_path.mkdir(exist_ok=False)

    df_names = ['DsTau3muPU0_Private.pkl', 'DsTau3muPU200_MTD.pkl', 'MinBiasPU200_MTD.pkl', 'MinBiasPU250_MTD.pkl']
    node_feature_names = ['mu_hit_ring', 'mu_hit_quality', 'mu_hit_bend', 'mu_hit_sim_phi', 'mu_hit_sim_theta', 'mu_hit_sim_eta', 'mu_hit_sim_r', 'mu_hit_sim_z']
    edge_feature_names = ['mu_hit_sim_phi', 'mu_hit_sim_theta', 'mu_hit_sim_eta', 'mu_hit_sim_r', 'mu_hit_sim_z']
    dataset = Tau3MuDataset(data_dir, df_names, args.run_type, node_feature_names, edge_feature_names, args.filter_soft_mu)

    batch_size = 256
    train_loader = DataLoader(dataset[dataset.idx_split['train']], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[dataset.idx_split['valid']], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[dataset.idx_split['test']], batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(dataset.x_dim, dataset.edge_attr_dim, n_layers=16, hidden_size=128, dropout_p=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    criterion = FocalLoss(alpha=0.5, gamma=3)

    print('[Splits]:')
    [print(f'    {k}: {len(v)}') for k, v in dataset.idx_split.items()]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] Number of trainable parameters: {num_params}')

    test_interval = 5
    for epoch in range(100):
        run_one_epoch(train_loader, epoch, 'train')
        run_one_epoch(valid_loader, epoch, 'valid')

        if epoch % test_interval == 0:
            run_one_epoch(test_loader, epoch, 'test')
            save_checkpoint(epoch)
        print('====================================')
