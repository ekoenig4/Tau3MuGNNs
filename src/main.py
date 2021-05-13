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
from utils import utils
from utils.dataset import Tau3MuDataset


def run_one_epoch(model, criterion, data_loader, writer, thres, device, epoch, phase, optimizer, scheduler):
    loader_len = len(data_loader)
    all_logits, all_targets, all_kl_loss = [], [], []

    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
    phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

    pbar = tqdm(data_loader)
    for idx, data in enumerate(pbar):
        loss, logits, targets, ce_loss, kl_loss, lr = run_one_batch(model, criterion, data.to(device), optimizer, scheduler)
        desc = utils.log_batch(logits, targets, thres, loss, ce_loss, kl_loss, phase, epoch, epoch * loader_len + idx, writer, lr)

        all_logits.append(logits)
        all_targets.append(targets)
        all_kl_loss.append(kl_loss)

        if idx == loader_len - 1:
            all_kl_loss = torch.tensor(all_kl_loss).cpu() if kl_loss is not None else None
            desc, loss, acc, auprc, auroc = utils.log_epoch(torch.cat(all_logits).cpu(), torch.cat(all_targets).cpu(),
                                                            all_kl_loss, thres, criterion, writer, phase, epoch)
        pbar.set_description(desc)

    return loss, acc, auprc, auroc


@torch.no_grad()
def eval_one_batch(model, criterion, data, optimizer, scheduler):
    assert optimizer is None
    assert scheduler is None

    model.eval()

    logits, kl_loss = model(data)
    ce_loss = criterion(logits, data.y)
    loss = ce_loss + kl_loss if kl_loss is not None else ce_loss

    kl_loss = kl_loss.item() if kl_loss is not None else None
    return loss.item(), logits.data, data.y.data, ce_loss.item(), kl_loss, None


def train_one_batch(model, criterion, data, optimizer, scheduler):
    model.train()
    optimizer.zero_grad()

    logits, kl_loss = model(data)
    ce_loss = criterion(logits, data.y)
    loss = ce_loss + kl_loss if kl_loss is not None else ce_loss

    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    lr = optimizer.param_groups[0]['lr']
    kl_loss = kl_loss.item() if kl_loss is not None else None
    return loss.item(), logits.data, data.y.data, ce_loss.item(), kl_loss, lr


def main(config, feature_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Tau3MuDataset(config['data'])
    utils.print_splits_and_acc_lower_bound(dataset)

    print('[INFO] Checking data md5...')
    md5 = dataset.md5sum(dataset.data, dataset.slices, dataset.idx_split)
    print(f'       md5: {md5}')

    log_path = utils.get_log_path_and_save_metadata(config, dataset.processed_dir, feature_name)
    batch_size, only_eval = config['model']['batch_size'], config['model']['only_eval']
    epochs, thres, test_interval = config['model']['epochs'], config['eval']['thres'], config['eval']['test_interval']
    start_epoch, resume = 0, config['model']['resume']

    train_loader = DataLoader(dataset[dataset.idx_split['train']], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[dataset.idx_split['valid']], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[dataset.idx_split['test']], batch_size=batch_size, shuffle=False)

    model = Model(dataset.x_dim, dataset.edge_attr_dim, config['data']['virtual_node'], config['model']).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams = {**config['model'], **utils.get_data_config_of_interest(config['data']), 'md5': md5, 'num_params': num_params}
    print(f'[INFO] Number of parameters: {num_params}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=eval(config['model']['lr']))
    scheduler = None
    if config['model']['scheduler']:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=eval(config['model']['lr']),
                                                        steps_per_epoch=len(train_loader), epochs=epochs)

    if resume:
        print(f'[INFO] Loading checkpoint from {resume}')
        checkpoint = torch.load(Path(config['data']['log_dir']) / resume / 'model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if only_eval:
            epochs = start_epoch
        assert start_epoch < epochs or only_eval

    criterion = utils.focal_loss if config['model']['focal_loss'] else torch.nn.BCELoss()

    writer = utils.Writer(log_path)
    res = {'train_res': [], 'valid_res': [], 'test_res': []}
    for epoch in range(start_epoch, epochs + 1):
        if only_eval:
            print('[INFO] Only evaluate data!')
            train_res = run_one_epoch(model, criterion, train_loader, writer, thres, device, epoch, 'evtrn', None, None)
        else:
            train_res = run_one_epoch(model, criterion, train_loader, writer, thres, device, epoch, 'train', optimizer, scheduler)

        valid_res = run_one_epoch(model, criterion, valid_loader, writer, thres, device, epoch, 'valid', None, None)
        if epoch % test_interval == 0 or epoch == start_epoch:
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
