# -*- coding = utf-8 -*-
# @Time: 2025/2/15 15:50
# @Author: wisehone
# @File: loss_func.py
# @SoftWare: PyCharm
import torch
import torch.nn.functional as F

def KL_loss(x, y):
    """
    y 的分布指导 x 的分布
    :param x:
    :param y:
    :return:
    """
    logp_x = F.log_softmax(x, dim=-1)
    p_y = F.softmax(y, dim=-1)
    return F.kl_div(logp_x, p_y, reduction='batchmean')

def MSE_loss(x, y):
    return F.mse_loss(x, y, reduction='mean')

def MAE_loss(x, y):
    return F.l1_loss(x, y, reduction='mean')

def CE_loss(x, y):
    return F.cross_entropy(x, y, reduction='mean')

def BCE_loss(x, y):
    return F.binary_cross_entropy(x, y, reduction='mean')

