# -*- coding: utf-8 -*-

"""
Created on 2021/4/15
@author: Siqi Miao
"""

import yaml
from tqdm import tqdm
from pathlib import Path

import torch
from torch_geometric.data import DataLoader

from model import Model
from utils import logger
from utils.dataset import Tau3MuDataset
from loss import Criterion


class Main(object):

    def __init__(self, config, log_name):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = Tau3MuDataset(config['data'])
        self.run_type = config['data']['run_type']
        self.endcap = config['data']['endcap']

        logger.print_splits_and_acc_lower_bound(self.dataset, self.run_type)
        md5 = self.dataset.md5sum(self.dataset.data, self.dataset.slices, self.dataset.idx_split, print_md5=True)

        if __name__ == '__main__':
            self.log_path = logger.get_log_path_and_save_metadata(config, self.dataset.processed_dir, log_name)
            self.writer = logger.Writer(self.log_path)

        self.auroc_max_fpr, self.test_interval = config['eval']['auroc_max_fpr'], config['eval']['test_interval']
        self.batch_size, self.only_eval = config['model']['batch_size'], config['model']['only_eval']
        self.start_epoch, self.resume, self.epochs = 0, config['model']['resume'], config['model']['epochs']

        self.train_loader = DataLoader(self.dataset[self.dataset.idx_split['train']], batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.dataset[self.dataset.idx_split['valid']], batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.dataset[self.dataset.idx_split['test']], batch_size=self.batch_size, shuffle=False)
        self.num_train_batch = len(self.train_loader)

        if self.endcap:
            self.train_loader = [self.train_loader, DataLoader([self.dataset.data_list_endcap[i] for i in self.dataset.idx_split['train']],
                                                               batch_size=self.batch_size, shuffle=True)]
            self.valid_loader = [self.valid_loader, DataLoader([self.dataset.data_list_endcap[i] for i in self.dataset.idx_split['valid']],
                                                               batch_size=self.batch_size, shuffle=False)]
            self.test_loader = [self.test_loader, DataLoader([self.dataset.data_list_endcap[i] for i in self.dataset.idx_split['test']],
                                                             batch_size=self.batch_size,  shuffle=False)]

        self.model = Model(self.dataset.x_dim, self.dataset.edge_attr_dim, config['model'], config['data']).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.hparams = {**config['model'], **logger.log_data_config(config['data']), 'md5': md5, 'num_params': num_params}
        print(f'[INFO] Number of trainable parameters: {num_params}')

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=eval(config['model']['lr']))
        self.criterion = Criterion(config['model'], config['data'])
        self.scheduler = None
        if config['model']['scheduler']:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=eval(config['model']['lr']),
                                                                 steps_per_epoch=self.num_train_batch, epochs=self.epochs)

    def run_one_epoch(self, data_loader, epoch, phase):
        loader_len = len(data_loader) if not self.endcap else len(data_loader[0])
        all_probs, all_targets, all_batch_losses = [], [], []

        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        pbar = tqdm(data_loader, total=loader_len) if not self.endcap else tqdm(zip(*data_loader), total=loader_len)
        for idx, data in enumerate(pbar):
            if self.endcap:
                data1, data2 = data
                loss_dict_1, probs_1, targets_1, _ = run_one_batch(data1.to(self.device))
                loss_dict_2, probs_2, targets_2, lr = run_one_batch(data2.to(self.device))
                loss_dict = {}
                for key in loss_dict_1.keys():
                    loss_dict[key] = (loss_dict_1[key] + loss_dict_2[key]) / 2

                if phase == 'train':
                    probs = torch.cat((probs_1, probs_2), dim=0)
                    targets = torch.cat((targets_1, targets_2), dim=0)
                else:
                    targets = torch.cat((targets_1, targets_2), dim=1)
                    targets = torch.max(targets, dim=1)[0].reshape(-1, 1)

                    probs = torch.cat((probs_1, probs_2), dim=1)
                    probs = torch.max(probs, dim=1)[0].reshape(-1, 1)

            else:
                loss_dict, probs, targets, lr = run_one_batch(data.to(self.device))

            desc = logger.log_batch(loss_dict, phase, epoch, epoch * loader_len + idx, self.writer, lr)

            all_probs.append(probs)
            all_targets.append(targets)
            all_batch_losses.append(loss_dict)

            if idx == loader_len - 1:
                desc, total_loss, auroc, recall = logger.log_epoch(torch.cat(all_probs), torch.cat(all_targets),
                                                                   all_batch_losses, self.auroc_max_fpr,
                                                                   self.writer, phase, epoch, self.run_type)
            pbar.set_description(desc)

        return total_loss, auroc, recall

    @torch.no_grad()
    def eval_one_batch(self, data):
        self.model.eval()

        probs, score_pair, pt_pair = self.model(data)
        loss, loss_dict = self.criterion(probs, data.y, kl_pairs=score_pair, pt_pairs=pt_pair)
        return loss_dict, probs.data.cpu(), data.y.data.cpu(), None

    def train_one_batch(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        probs, score_pair, pt_pair = self.model(data)
        loss, loss_dict = self.criterion(probs, data.y, kl_pairs=score_pair, pt_pairs=pt_pair)

        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.param_groups[0]['lr']
        return loss_dict, probs.data.cpu(), data.y.data.cpu(), lr

    def load_checkpoint(self):
        print(f'[INFO] Loading checkpoint from {self.resume}')
        checkpoint = torch.load(Path(self.config['data']['log_dir']) / self.resume / 'model.pt',
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        if self.only_eval:
            self.epochs = self.start_epoch
        assert self.start_epoch < self.epochs or self.only_eval

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'n_iters': (epoch + 1) * self.num_train_batch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.log_path / 'model.pt')
        print('====================================')

    def main(self):

        if self.resume:
            self.load_checkpoint()

        res = {'train_res': [], 'valid_res': [], 'test_res': []}
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.only_eval:
                print('[INFO] Only evaluate data!')
                train_res = self.run_one_epoch(self.train_loader, epoch, 'evtrn')
            else:
                train_res = self.run_one_epoch(self.train_loader, epoch, 'train')

            valid_res = self.run_one_epoch(self.valid_loader, epoch, 'valid')
            if epoch % self.test_interval == 0 or epoch == self.start_epoch:
                test_res = self.run_one_epoch(self.test_loader, epoch, 'test')

            for phase in res.keys():
                res[phase].append(eval(phase))

            self.writer.add_hparams(self.hparams, logger.get_best_res(res))
            self.save_checkpoint(epoch)

        self.writer.close()


if __name__ == '__main__':
    configs = Path('./configs')

    for cfg in configs.iterdir():
        cfg_dict = yaml.safe_load(cfg.open('r'))
        Main(cfg_dict, '_'.join(cfg.stem.split('_')[1:])).main()
