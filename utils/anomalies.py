# -*- coding = utf-8 -*-
# @Time: 2025/2/14 11:18
# @Author: wisehone
# @File: anomalies.py
# @SoftWare: PyCharm
import torch
from data import data_loaders
import numpy as np
from utils.utils import sum_peak

class Anomalies_injection(object):
    def __init__(self, ori_data_loader, anomaly_rate, blended_substitution,
                 static_override, peak_implantation, flip_replacing_interval, replacing_weight):
        self.ori_data_loader = ori_data_loader
        self.anomaly_rate = anomaly_rate
        self.repalcing_weight = replacing_weight

        self.blended_substitution_prob = 1 - blended_substitution
        self.static_override_prob = self.blended_substitution_prob - static_override
        self.peak_implantation_prob = self.static_override_prob - peak_implantation

        if flip_replacing_interval == 'all':
            self.vertical_flop = True
            self.horizon_flop = True
        elif flip_replacing_interval == 'vertical':
            self.vertical_flop = True
            self.horizon_flop = False
        elif flip_replacing_interval == 'horizon':
            self.vertical_flop = False
            self.horizon_flop = True
        else:
            self.vertical_flop = False
            self.horizon_flop = False

    def anomaly_injection(self, s_x, w_x):
        n_batch, win_size, num_features = s_x.shape[0], s_x.shape[1], s_x.shape[2]
        anomaly_len = np.array(np.random.randint(1., 2., size=(n_batch, )) * win_size * self.anomaly_rate, dtype=np.int32) # 每个batch 的异常个数（根据注入的异常比例略有波动）

        replacing_type = np.random.uniform(0., 1., size=(n_batch, )) # 确定每个 batch 的异常注入方式
        replacing_columns = np.random.uniform(0., 1., size=(n_batch, num_features)) # 确定那些通道要进行异常注入
        replacing_columns = replacing_columns - np.maximum(replacing_columns.min(axis=1, keepdims=True), 0.3) < 0.01 # 通过概率确定

        true_anomaly_labels = torch.zeros(n_batch, win_size, dtype=torch.float32)

        s_x = s_x.view(n_batch, num_features, win_size)
        w_x = w_x.view(n_batch, num_features, win_size)

        ori_datas, labels = next(iter(self.ori_data_loader))
        ori_datas = ori_datas.view(n_batch, num_features, win_size)

        for i, type, lenn, ori_data, replacing_column in zip(range(n_batch), replacing_type, anomaly_len, ori_datas, replacing_columns):
            if lenn > 0:
                if type > self.blended_substitution_prob:
                    if self.vertical_flop:
                        ori_data = 1 - ori_data
                    if self.horizon_flop:
                        ori_data = torch.flip(ori_data, dims=(0, ))
                    ori_data[replacing_column] *= self.repalcing_weight
                    # 生成一段连续的，长度为 lenn 的区间异常
                    anomaly_index = np.random.randint(0, win_size - lenn + 1)
                    start_index = np.random.randint(0, win_size - lenn + 1)
                    s_x[i, replacing_column, anomaly_index: anomaly_index + lenn] = ori_data[replacing_column, start_index: start_index+lenn] # !!!
                    w_x[i, replacing_column, anomaly_index: anomaly_index + lenn] = ori_data[replacing_column, start_index: start_index+lenn]

                    true_anomaly_labels[i, anomaly_index: anomaly_index + lenn] = 1.
                elif type > self.static_override_prob:
                    anomaly_index = np.random.randint(0, win_size - lenn + 1)
                    static_value = (np.random.random() + 0.05) * 0.8

                    s_x[i, replacing_column, anomaly_index: anomaly_index + lenn] = static_value
                    w_x[i, replacing_column, anomaly_index: anomaly_index + lenn] = static_value

                    true_anomaly_labels[i, anomaly_index: anomaly_index + lenn] = 1.
                elif type > self.peak_implantation_prob:
                    anomaly_index = np.random.randint(0, win_size, size=lenn)
                    anomaly_index = np.sort(anomaly_index)
                    # !!
                    s_tmp = s_x[i, replacing_column].clone()
                    w_tmp = w_x[i, replacing_column].clone()
                    s_peak_value = s_tmp[:, anomaly_index] < 0.5
                    w_peak_value = w_tmp[:, anomaly_index] < 0.5

                    s_peak_value = s_peak_value
                    w_peak_value = w_peak_value
                    peak_value = sum_peak(s_peak_value, w_peak_value)
                    peak_value = peak_value + (0.1 * (1 - 2 * peak_value)) * np.random.random(lenn)

                    peak_value = peak_value.to(torch.float32)
                    s_tmp[:, anomaly_index] = peak_value
                    w_tmp[:, anomaly_index] = peak_value

                    s_x[i, replacing_column] = s_tmp
                    w_x[i, replacing_column] = w_tmp

                    true_anomaly_labels[i, anomaly_index] = 1.

                    del s_tmp, w_tmp

        s_x = s_x.view(n_batch, win_size, num_features)
        w_x = w_x.view(n_batch, win_size, num_features)

        return s_x, w_x, true_anomaly_labels


