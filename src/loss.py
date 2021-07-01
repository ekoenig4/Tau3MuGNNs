# -*- coding: utf-8 -*-

"""
Created on 2021/5/17

@author: Siqi Miao
"""

from math import sqrt
import torch
import torch.nn.functional as F
from torch_scatter import scatter_sum


class Criterion(torch.nn.Module):

    def __init__(self, model_config, data_config):
        super(Criterion, self).__init__()
        self.focal_loss = model_config['focal_loss']
        self.alpha = model_config['focal_alpha']
        self.gamma = model_config['focal_gamma']
        self.rmse_beta = model_config['rmse_beta']
        self.kl_beta = model_config['kl_beta']
        self.att_sup = model_config['att_sup']

        self.run_type = data_config['run_type']

    def forward(self, inputs, targets, pt_pairs, kl_pairs):

        loss_dict = {}
        if self.run_type == 'regress':
            idx_dict = {'e': 0, 'pt': 3, 'eta': 6}
            pt_inputs, pt_targets = pt_pairs
            assert pt_targets.shape[1] == 9
            sl1_loss = F.smooth_l1_loss(pt_inputs, pt_targets[:, list(idx_dict.values())])
            loss = sl1_loss

            with torch.no_grad():
                pred_v = {key: None for key in idx_dict.keys()}
                for i, (k, j) in enumerate(idx_dict.items()):
                    pred = (pt_inputs[:, i] * pt_targets[:, j+2]) + pt_targets[:, j+1]
                    pred_v[k] = pred
                    with torch.no_grad():
                        ground_truth = (pt_targets[:, j] * pt_targets[:, j + 2]) + pt_targets[:, j + 1]
                        loss_dict['MAE_' + k] = F.l1_loss(pred, ground_truth).item()
                sq_mass = pred_v['e'] ** 2 - (pred_v['pt'] * torch.cosh(pred_v['eta'])) ** 2
                mass = torch.sqrt(F.relu(sq_mass)) - 1.77682
                target_mass = torch.zeros_like(mass)
                # loss += F.smooth_l1_loss(mass, target_mass)/3 * 0.1

            with torch.no_grad():
                loss_dict['MAE_mass'] = F.l1_loss(mass, target_mass).item()

        elif self.focal_loss:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
            p_t = inputs * targets + (1 - inputs) * (1 - targets)
            loss = bce_loss * ((1 - p_t) ** self.gamma)

            if self.alpha >= 0:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                loss = alpha_t * loss
            loss = loss.mean()
            loss_dict['focal'] = loss.item()
        else:
            loss = F.binary_cross_entropy(inputs, targets)
            loss_dict['bce'] = loss.item()

        if self.att_sup:
            kl_inputs, kl_targets, batch = kl_pairs
            kl_loss = scatter_sum(F.kl_div(torch.log(kl_inputs + 1e-14), kl_targets, reduction='none'), batch).mean()
            loss += self.kl_beta * kl_loss
            loss_dict['raw_kl'] = kl_loss.item()

        loss_dict['total'] = loss.item()
        return loss, loss_dict
