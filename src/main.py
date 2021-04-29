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
from pytorch_model_summary import summary

from model import Model
from utils import utils
from utils.dataset import Tau3MuDataset


def run_one_epoch(model, criterion, data_loader, writer, thres, device, epoch, phase, optimizer, scheduler):
    loader_len = len(data_loader)
    all_logits, all_targets = [], []

    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
    phase = 'test ' if phase == 'test' else phase

    pbar = tqdm(data_loader)
    for idx, data in enumerate(pbar):
        loss, logits, targets, lr = run_one_batch(model, criterion, data.to(device), optimizer, scheduler)
        desc = utils.log_batch(logits, targets, thres, loss, phase, epoch, epoch * loader_len + idx, writer, lr)

        all_logits.append(logits)
        all_targets.append(targets)

        if idx == loader_len - 1:
            desc, loss, acc, auprc, auroc = utils.log_epoch(torch.cat(all_logits).cpu(), torch.cat(all_targets).cpu(),
                                                            thres, criterion, writer, phase, epoch)
        pbar.set_description(desc)

    return loss, acc, auprc, auroc


@torch.no_grad()
def eval_one_batch(model, criterion, data, optimizer, scheduler):
    assert optimizer is None
    assert scheduler is None

    model.eval()

    logits = model(data)
    loss = criterion(logits, data.y)
    return loss.item(), logits.data, data.y.data, None


def train_one_batch(model, criterion, data, optimizer, scheduler):
    model.train()
    optimizer.zero_grad()

    logits = model(data)
    loss = criterion(logits, data.y)

    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    lr = optimizer.param_groups[0]['lr']
    return loss.item(), logits.data, data.y.data, lr


def main(config, feature_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Tau3MuDataset(config['data'])
    utils.print_splits_and_acc_lower_bound(dataset)

    print('[INFO] Checking data md5...')
    md5 = dataset.md5sum(dataset.data, dataset.slices, dataset.idx_split)
    print(f'       md5: {md5}')

    model = Model(dataset.x_dim, dataset.edge_attr_dim, config['data']['virtual_node'], config['model']).to(device)
    print(summary(model, next(iter(DataLoader(dataset[[0]]))).to(device), show_input=True))

    log_path = utils.get_log_path_and_save_metadata(config, dataset.processed_dir, feature_name)
    batch_size = config['model']['batch_size']
    epochs, thres, test_interval = config['model']['epochs'], config['eval']['thres'], config['eval']['test_interval']
    hparams = {**config['model'], **utils.get_data_config_of_interest(config['data']), 'md5': md5}

    train_loader = DataLoader(dataset[dataset.idx_split['train']], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[dataset.idx_split['valid']], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[dataset.idx_split['test']], batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=eval(config['model']['lr']))
    if config['model']['scheduler']:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=eval(config['model']['lr']),
                                                        steps_per_epoch=len(train_loader), epochs=epochs)
    else:
        scheduler = None

    criterion = torch.nn.BCELoss()

    writer = utils.Writer(log_path)
    res = {'train_res': [], 'valid_res': [], 'test_res': []}
    for epoch in range(epochs + 1):
        train_res = run_one_epoch(model, criterion, train_loader, writer, thres, device, epoch, 'train', optimizer, scheduler)
        valid_res = run_one_epoch(model, criterion, valid_loader, writer, thres, device, epoch, 'valid', None, None)

        if epoch % test_interval == 0:
            test_res = run_one_epoch(model, criterion, test_loader, writer, thres, device, epoch, 'test', None, None)

        for phase in res.keys():
            res[phase].append(eval(phase))

        writer.add_hparams(hparams, utils.get_best_res(res))
        torch.save({
            'epoch': epoch,
            'n_iters': (epoch + 1) * len(train_loader),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, log_path / 'model.pt')
        print('====================================')

    writer.close()


if __name__ == '__main__':
    configs = Path('./configs')

    for cfg in configs.iterdir():
        cfg_dict = yaml.safe_load(cfg.open('r'))
        main(cfg_dict, '_'.join(cfg.stem.split('_')[1:]))
