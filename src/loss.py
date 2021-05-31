# -*- coding: utf-8 -*-

"""
Created on 2021/5/17

@author: Siqi Miao
"""

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

        self.pred_pt = data_config['pred_pt']

    def forward(self, inputs, targets, pt_pairs, kl_pairs):

        loss_dict = {}
        if self.focal_loss:
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

        if self.pred_pt:
            pos_idx = (targets == 1).reshape(-1)
            if pos_idx.sum() != 0:
                pt_inputs, pt_targets = pt_pairs
                pt_inputs = pt_inputs[pos_idx]
                pt_targets = pt_targets[pos_idx]

                rmse_loss = torch.sqrt(F.mse_loss(pt_inputs, pt_targets))
                loss += self.rmse_beta * rmse_loss
                loss_dict['raw_rmse'] = rmse_loss.item()
            else:
                loss_dict['raw_rmse'] = 0.0

        loss_dict['total'] = loss.item()
        return loss, loss_dict
