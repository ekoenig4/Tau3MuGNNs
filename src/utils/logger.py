# -*- coding: utf-8 -*-

"""
Created on 2021/4/24

@author: Siqi Miao
"""

import yaml
import shutil
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from itertools import product
from matplotlib import figure
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import check_matplotlib_support

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard._convert_np import make_np
from tensorboard.plugins.pr_curve.plugin_data_pb2 import PrCurvePluginData
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.summary_pb2 import Summary


def get_log_path_and_save_metadata(config, processed_data_dir, log_name):
    log_path = config['model']['model_name'] + '-' + datetime.now().strftime("%m_%d_%Y-%H_%M_%S") + '-' + log_name
    log_path = Path(config['data']['log_dir']) / log_path
    log_path.mkdir(exist_ok=False)

    src_dir = Path(config['data']['src_dir'])
    yaml.dump(config, open(Path(log_path) / 'config.yml', 'w'))
    shutil.copy(src_dir / 'model.py', log_path / 'model.py')
    shutil.copytree(src_dir / 'layers', log_path / 'layers')

    shutil.copy(Path(processed_data_dir) / 'data_md5.txt', log_path / 'data_md5.txt')
    return log_path


def print_splits_and_acc_lower_bound(dataset):
    print('[Splits]')
    [print(f'    {k}: {len(v)}') for k, v in dataset.idx_split.items()]

    _, (n0, n1) = np.unique(dataset.data.y[dataset.idx_split['train']], return_counts=True)
    lb = n0 / (n0 + n1)
    print(f'[INFO] Training accuracy lower bound: {max(1 - lb, lb): .3f}')


def get_best_res(res):
    best_res = {}

    max_recall_epoch = np.argmax(np.array(res['valid_res'])[:, 2])
    for phase, scores in res.items():
        best_res[f'Hparams/{phase}/loss'] = np.array(scores)[:, 0][max_recall_epoch]
        best_res[f'Hparams/{phase}/auroc'] = np.array(scores)[:, 1][max_recall_epoch]
        best_res[f'Hparams/{phase}/recall_on_max_fpr'] = np.array(scores)[:, 2][max_recall_epoch]
    return best_res


def log_data_config(config):
    interested_config = {}
    for k, v in config.items():
        if isinstance(v, bool):
            interested_config[k] = v

        elif k == 'conditions':
            for feature, condition in v.items():
                interested_config[feature + ' ' + condition] = True
    return interested_config


def log_batch(loss_dict, phase, epoch, step, writer, lr):
    loss_desc = ''
    for loss_name, loss_value in loss_dict.items():
        writer.add_scalar(f'Batch/{phase}/{loss_name}', loss_value, step)
        loss_desc += f'{loss_name}: {loss_value: .3f}, '

    if lr is not None and phase == 'train':
        writer.add_scalar(f'Stats/lr_schedule', lr, step)

    desc = f'[Epoch: {epoch}]: {phase}........., ' + loss_desc + f'....................................'
    return desc


# TODO: add_pr_curve will only update after testing the model
def log_epoch(probs, targets, all_batch_losses, auroc_max_fpr, writer, phase, epoch):

    loss_desc = ''
    for loss_name in all_batch_losses[0].keys():
        current_loss = 0
        for i in range(len(all_batch_losses)):
            current_loss += all_batch_losses[i][loss_name]
        current_loss /= len(all_batch_losses)

        writer.add_scalar(f'{phase}/{loss_name}', current_loss, epoch)
        loss_desc += f'{loss_name}: {current_loss: .3f}, '

        if 'total' in loss_name:
            total_loss = current_loss

    auroc = metrics.roc_auc_score(targets, probs)
    partial_auroc = metrics.roc_auc_score(targets, probs, max_fpr=auroc_max_fpr)
    fpr, recall, thres = metrics.roc_curve(targets, probs)
    indices = get_idx_for_interested_fpr(fpr, [auroc_max_fpr, auroc_max_fpr/10])

    writer.add_scalar(f'{phase}/AUROC/', auroc, epoch)
    writer.add_scalar(f'{phase}/Partial_AUROC/', partial_auroc, epoch)
    writer.add_scalar(f'{phase}/recall_on_max_fpr/', recall[indices[0]], epoch)
    writer.add_scalar(f'{phase}/recall_on_max_fpr_over_10/', recall[indices[1]], epoch)
    writer.add_roc_curve(f'ROC_Curve/{phase}', targets, probs, epoch)

    cm = metrics.confusion_matrix(targets, y_pred=probs > thres[indices[0]], normalize=None)
    fig = PlotCM(confusion_matrix=cm, display_labels=['Neg', 'Pos']).plot(cmap=plt.cm.Blues).figure_
    writer.add_figure(f'Confusion Matrix - max_fpr/{phase}', fig, epoch)

    cm = metrics.confusion_matrix(targets, y_pred=probs > thres[indices[1]], normalize=None)
    fig = PlotCM(confusion_matrix=cm, display_labels=['Neg', 'Pos']).plot(cmap=plt.cm.Blues).figure_
    writer.add_figure(f'Confusion Matrix - max_fpr_over_10/{phase}', fig, epoch)
    desc = f'[Epoch: {epoch}]: {phase} finished, ' + loss_desc + f'auroc: {auroc: .3f}, recall@maxfpr: {recall[indices[0]]: .3f}'
    return desc, total_loss, auroc, recall[indices[0]]


def get_idx_for_interested_fpr(fpr, interested_fpr):
    res = []
    for each in interested_fpr:
        for i in range(1, fpr.shape[0]):
            if fpr[i] > each:
                res.append(i-1)
                break
    assert len(res) == len(interested_fpr)
    return res


class Writer(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)

    def add_roc_curve(self, tag, labels, predictions, global_step=None,
                      num_thresholds=127, weights=None, walltime=None):

        torch._C._log_api_usage_once("tensorboard.logging.add_pr_curve")
        labels, predictions = make_np(labels), make_np(predictions)
        self._get_file_writer().add_summary(
            Writer.roc_curve(tag, labels, predictions, num_thresholds, weights),
            global_step, walltime)

    @staticmethod
    def roc_curve(tag, labels, predictions, num_thresholds=127, weights=None):
        # weird, value > 127 breaks protobuf
        num_thresholds = min(num_thresholds, 127)
        data = Writer.compute_roc_curve(labels, predictions,
                                        num_thresholds=num_thresholds, weights=weights)
        pr_curve_plugin_data = PrCurvePluginData(
            version=0, num_thresholds=num_thresholds).SerializeToString()
        plugin_data = SummaryMetadata.PluginData(
            plugin_name='pr_curves', content=pr_curve_plugin_data)
        smd = SummaryMetadata(plugin_data=plugin_data)
        tensor = TensorProto(dtype='DT_FLOAT',
                             float_val=data.reshape(-1).tolist(),
                             tensor_shape=TensorShapeProto(
                                 dim=[TensorShapeProto.Dim(size=data.shape[0]),
                                      TensorShapeProto.Dim(size=data.shape[1])]))
        return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])

    @staticmethod
    def compute_roc_curve(labels, predictions, num_thresholds=None, weights=None):
        _MINIMUM_COUNT = 1e-7

        if weights is None:
            weights = 1.0

        # Compute bins of true positives and false positives.
        bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
        float_labels = labels.astype(np.float)
        histogram_range = (0, num_thresholds - 1)
        tp_buckets, _ = np.histogram(
            bucket_indices,
            bins=num_thresholds,
            range=histogram_range,
            weights=float_labels * weights)
        fp_buckets, _ = np.histogram(
            bucket_indices,
            bins=num_thresholds,
            range=histogram_range,
            weights=(1.0 - float_labels) * weights)

        # Obtain the reverse cumulative sum.
        tp = np.cumsum(tp_buckets[::-1])[::-1]
        fp = np.cumsum(fp_buckets[::-1])[::-1]
        tn = fp[0] - fp
        fn = tp[0] - tp
        precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
        recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
        fpr = fp / np.maximum(_MINIMUM_COUNT, fp + tn)
        return np.stack((tp, fp, tn, fn, recall, fpr))


class PlotROC(metrics.RocCurveDisplay):
    def plot(self, ax=None, *, name=None, **kwargs):
        check_matplotlib_support('RocCurveDisplay.plot')

        name = self.estimator_name if name is None else name

        line_kwargs = {}
        if self.roc_auc is not None and name is not None:
            line_kwargs["label"] = f"{name} (AUC = {self.roc_auc:0.3f})"
        elif self.roc_auc is not None:
            line_kwargs["label"] = f"AUC = {self.roc_auc:0.3f}"
        elif name is not None:
            line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        if ax is None:
            fig = figure.Figure()
            ax = fig.subplots()

        self.line_, = ax.plot(self.fpr, self.tpr, **line_kwargs)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        info_pos_label = (f" (Positive label: {self.pos_label})"
                          if self.pos_label is not None else "")

        xlabel = "False Positive Rate" + info_pos_label
        ylabel = "True Positive Rate" + info_pos_label
        ax.set(xlabel=xlabel, ylabel=ylabel)

        if "label" in line_kwargs:
            ax.legend(loc="lower right")

        self.ax_ = ax
        self.figure_ = ax.figure
        return self


class PlotCM(metrics.ConfusionMatrixDisplay):
    def plot(self, *, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None,
             ax=None, colorbar=True):
        check_matplotlib_support("ConfusionMatrixDisplay.plot")

        if ax is None:
            fig = figure.Figure()
            ax = fig.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], '.2g')
                    if cm.dtype.kind != 'f':
                        text_d = format(cm[i, j], 'd')
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                self.text_[i, j] = ax.text(
                    j, i, text_cm,
                    ha="center", va="center",
                    color=color)

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels
        if colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=display_labels,
               yticklabels=display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self
