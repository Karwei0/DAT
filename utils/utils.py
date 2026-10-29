# -*- coding = utf-8 -*-
# @Time: 2025/2/13 17:12
# @Author: wisehone
# @File: utils.py
# @SoftWare: PyCharm
import os

import torch
from torch import nn
import copy

from metrics.get_all_evaluation_score import get_all_evaluation_score
from utils.augments import *
from utils.eval_utils import score_to_label


def clone_encoder(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def weak_augment(x, jittering_w=0.5, scaling=0.5):
    if jittering_w > 0:
        Jit = jitter()
        jittering_x = Jit(x)
    if scaling > 0:
        Sca = scale()
        scaling_x = Sca(x)

    return (jittering_w * jittering_x + scaling * scaling_x) / 2.0

def strong_augment(x, win_size, jittering_s=0.3, time_wrapping=0.5, permutation=0.2):
    if jittering_s > 0:
        Jit = jitter()
        jittering_x = Jit(x)
    if time_wrapping > 0:
        TiW = time_wrap()
        time_wrapping_x = TiW(x)
    if permutation > 0:
        PF = permutation_f(win_size)
        permutation_x = PF(x)
    return (jittering_s * jittering_x + time_wrapping * time_wrapping_x + permutation * permutation_x) / 3.0

def sum_peak(a, b):
    res = a.clone()
    res[a != b] = torch.rand(a.shape)[a != b] > 0.5
    return res

def f(x):
    print(f"{x['precision_ori']}\t{x['recall_ori']}\t{x['f1_score_ori']}\t{x['precision_pa']}\t{x['recall_pa']}\t{x['f_score_pa']}\t{x['point_auc']}\t{x['range_auc']}\t{x['Affiliation precision']}\t{x['Affiliation recall']}\t{x['R_AUC_ROC']}\t{x['R_AUC_PR']}\t{x['VUS_ROC']}\t{x['VUS_PR']}")


def evaluation(combined_anomaly_score, test_anomaly_score, test_true_label, dff=4):
    thresh = np.percentile(combined_anomaly_score, 100 - dff)
    # print('threshold: {}'.format(thresh))
    pred_label = score_to_label(test_anomaly_score, thresh)

    res = get_all_evaluation_score(pred_label, test_true_label)

    ans = {
        key: round(value, 5) if isinstance(value, float) else value
        for key, value in res.items()
    }

    return ans