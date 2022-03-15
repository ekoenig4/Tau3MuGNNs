import torch
import yaml
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from models import Model
from utils import Criterion, Writer, log_epoch, set_seed, get_data_loaders
from explainer import GSAT, ExtractorMLP


class Tau3MuGNNs:

    def __init__(self, config, device, log_path, setting):
        self.config = config
        self.device = device
        self.log_path = log_path
        self.writer = Writer(log_path)

        self.data_loaders, x_dim, edge_attr_dim, _ = get_data_loaders(setting, config['data'], config['optimizer']['batch_size'])

        clf = Model(x_dim, edge_attr_dim, config['data']['virtual_node'], config['model']).to(self.device)
        extractor = ExtractorMLP(config['model']['out_channels'], learn_edge_att=False).to(device)
        optimizer = torch.optim.AdamW(list(extractor.parameters()) + list(clf.parameters()), lr=config['optimizer']['lr'])
        criterion = Criterion(config['optimizer'])

        self.gsat = GSAT(clf, extractor, criterion, optimizer, learn_edge_att=False, final_r=0.3, decay_interval=10, init_r=0.8)
        print(f'[INFO] Number of trainable parameters: {sum(p.numel() for p in self.gsat.parameters())}')

    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.gsat.extractor.eval()
        self.gsat.clf.eval()

        edge_att, loss, loss_dict, clf_logits, raw_att = self.gsat.forward_pass(data, epoch, training=False)
        return loss_dict, clf_logits.data.cpu(), edge_att.data.cpu().reshape(-1), raw_att.data.cpu().reshape(-1)

    def train_one_batch(self, data, epoch):
        self.gsat.extractor.train()
        self.gsat.clf.train()

        edge_att, loss, loss_dict, clf_logits, raw_att = self.gsat.forward_pass(data, epoch, training=True)
        self.gsat.optimizer.zero_grad()
        loss.backward()
        self.gsat.optimizer.step()
        return loss_dict, clf_logits.data.cpu(), edge_att.data.cpu().reshape(-1), raw_att.data.cpu().reshape(-1)

    def run_one_epoch(self, data_loader, epoch, phase):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict, all_clf_logits, all_clf_labels, all_exp_labels, all_exp_probs = {}, [], [], [], []
        pbar = tqdm(data_loader, total=loader_len)
        for idx, data in enumerate(pbar):
            loss_dict, clf_logits, _, att = run_one_batch(data.to(self.device), epoch)  # node-level att

            if self.config['data']['virtual_node']:
                mask = torch.ones_like(att).bool()
                mask[data.ptr[1:] - 1] = False
                att = att[mask]
            exp_labels = data.node_label.data.cpu().reshape(-1) if data.get('node_label', None) is not None else torch.full_like(att, -1).long()

            desc = log_epoch(epoch, phase, loss_dict, clf_logits, data.y.data.cpu(), batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v
            all_clf_logits.append(clf_logits), all_clf_labels.append(data.y.data.cpu())

            pos_data_node_ids = (data.y[data.batch.reshape(-1)] == 1).reshape(-1)
            all_exp_labels.append(exp_labels[pos_data_node_ids]), all_exp_probs.append(att[pos_data_node_ids])

            if idx == loader_len - 1:
                all_clf_logits, all_clf_labels = torch.cat(all_clf_logits), torch.cat(all_clf_labels)
                all_exp_labels, all_exp_probs = torch.cat(all_exp_labels), torch.cat(all_exp_probs)
                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, auroc, recall, avg_loss = log_epoch(epoch, phase, all_loss_dict, all_clf_logits, all_clf_labels, False, self.writer, all_exp_probs, all_exp_labels)
            pbar.set_description(desc)

        return avg_loss, auroc, recall

    def train(self):
        start_epoch = 0
        best_val_recall = 0
        for epoch in range(start_epoch, self.config['optimizer']['epochs'] + 1):
            self.run_one_epoch(self.data_loaders['train'], epoch, 'train')

            if epoch % self.config['eval']['test_interval'] == 0:
                valid_res = self.run_one_epoch(self.data_loaders['valid'], epoch, 'valid')
                test_res = self.run_one_epoch(self.data_loaders['test'], epoch, 'test')
                if valid_res[-1] >= best_val_recall:
                    best_val_recall, best_test_recall, best_epoch = valid_res[-1], test_res[-1], epoch

            self.writer.add_scalar('best/best_epoch', best_epoch, epoch)
            self.writer.add_scalar('best/best_val_recall', best_val_recall, epoch)
            self.writer.add_scalar('best/best_test_recall', best_test_recall, epoch)
            print('-' * 50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Tau3MuGNNs')
    parser.add_argument('--setting', type=str, help='experiment settings', default='GNN-full-dR-2-mix-debug35')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=5)
    args = parser.parse_args()
    setting = args.setting
    cuda_id = args.cuda
    print(f'[INFO] Running {setting} on cuda {cuda_id}')

    torch.set_num_threads(5)
    set_seed(42)
    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    config = yaml.safe_load(Path(f'./configs/{setting}.yml').open('r'))
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    log_name = f'{time}-{setting}' if not config['optimizer']['resume'] else config['optimizer']['resume']

    log_path = Path(config['data']['log_dir']) / log_name
    Tau3MuGNNs(config, device, log_path, setting).train()


if __name__ == '__main__':
    # import os
    # os.chdir('./src')
    main()
