import json
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

idx2class = {0: 'fire', 1: 'no_fire'}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output : batch_size, num_classes
    maxk = max(topk)
    batch_size = target.size(0)  #target: batch_size,

    _, pred = output.topk(maxk, 1, True, True)
    # pred: batch_size, topk
    pred = pred.t()
    # pred: topk, batch_size
    correct = pred.eq(target.view(1, -1).expand_as(pred))  #correct: topk, batch_size

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def plot_epochs_stats(stats, save_path):
    df_acc = pd.DataFrame.from_dict(stats['acc']).reset_index(). \
        melt(id_vars=['index']).rename(columns={"index": "epochs"})
    df_losses = pd.DataFrame.from_dict(stats['losses']).reset_index(). \
        melt(id_vars=['index']).rename(columns={"index": "epochs"})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    sns.lineplot(data=df_acc, x="epochs", y="value",
                 hue="variable", ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=df_losses, x="epochs", y="value",
                 hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
    plt.savefig(save_path)
    plt.close()


def plot_heatmap(stats, save_path):
    plt.figure(figsize=(16, 9), dpi=100)
    df = pd.DataFrame(confusion_matrix(stats['y_test'], stats['y_pred']))
    confusion_matrix_df = df.rename(columns=idx2class, index=idx2class)
    heatmap = sns.heatmap(confusion_matrix_df, annot=True, fmt='d')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()


def plot_report(stats, save_path):
    report = classification_report(stats['y_test'], stats['y_pred'],
                                   labels=[k for k, v in idx2class.items()],
                                   target_names=[v for _, v in idx2class.items()],
                                   output_dict=True)
    with open(save_path, 'w') as fh:
        json.dump(report, fh, ensure_ascii=False)

def write_val_result(stats, save_path):
    with open(save_path, 'w') as fh:
        fh.write('path,true_label,pred_label,correct,value\n')
        for path, true_label, pred_label, pred_list_value in zip(stats['path'], stats['y_test'], stats['y_pred'],
                                                                stats['y_value']):
            true_label = idx2class[true_label]
            pred_label = idx2class[pred_label]
            pred_value = pred_list_value
            correct = 1 if true_label == pred_label else 0
            fh.write('{},{},{},{},{}\n'.format(path, true_label, pred_label, correct, pred_value))


def gen_index(inputs_torch, thre):
    np_arr = inputs_torch.numpy()
    index = np.argwhere(np_arr >= thre).reshape(-1).tolist()
    return index

