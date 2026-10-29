# -*- coding = utf-8 -*-
# @Time: 2025/2/12 22:17
# @Author: wisehone
# @File: DAT.py
# @SoftWare: PyCharm

import torch.nn as nn
import torch
from models.transformer import get_transformer_encoder


# Dual Anomaly-aware Transformer
class DAT(nn.Module):
    def __init__(self, s_linear_embedding, w_linear_embedding, transformer_enconder, predictive_mlp, s_contextual_mlp, w_contextual_mlp, patch_size, max_seq_len):
        super(DAT, self).__init__()
        self.s_linear_embedding = s_linear_embedding
        self.w_linear_embedding = w_linear_embedding

        self.transformer_enconder = transformer_enconder
        self.predictive_mlp = predictive_mlp

        self.s_contextual_mlp = s_contextual_mlp
        self.w_contextual_mlp = w_contextual_mlp
        self.patch_size = patch_size

        self.max_seq_len = max_seq_len
        self.data_seq_len = patch_size * max_seq_len

    def forward(self, s_x, w_x):
        n_batch = s_x.shape[0]
        # 1. encoding: the module(mlp with one layer) doesn't share weights, but two different encoders are used for two different branches
        s_x = s_x.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.max_seq_len, -1)
        w_x = w_x.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.max_seq_len, -1)
        s_embedded_out = self.s_linear_embedding(s_x)
        w_embedded_out = self.w_linear_embedding(w_x)

        # 2.Transformer body: the module, Transformer encoder uses the same weights for both branches,
        s_transformer_out = self.transformer_enconder(s_embedded_out)
        w_transformer_out = self.transformer_enconder(w_embedded_out)

        # 3.project to anomaly score: the module (mlp with three layers) uses the same weights for both branches
        s_predictive_out = self.predictive_mlp(s_transformer_out)
        w_predictive_out = self.predictive_mlp(w_transformer_out)

        s_predictive_out = s_predictive_out.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.data_seq_len, -1)
        w_predictive_out = w_predictive_out.view(n_batch, self.max_seq_len, self.patch_size, -1).view(n_batch, self.data_seq_len, -1)

        # 3.non-linear projection: the module (mlp with three layers) uses the different weights for both branches
        s_contextual_out = self.s_contextual_mlp(s_transformer_out)
        w_contextual_out = self.w_contextual_mlp(w_transformer_out)

        return s_predictive_out, w_predictive_out, s_contextual_out, w_contextual_out

    # 添加获取嵌入向量的方法
    def extract_embeddings(self, x):
        self.eval()
        with torch.no_grad():
            # 数据增强
            x_s = self.aug_s(x)
            x_w = self.aug_w(x)
            # 异常注入
            x_s, _ = self.injector(x_s)
            x_w, _ = self.injector(x_w)
            # 编码器处理
            z_s = self.encoder(x_s)
            z_w = self.encoder(x_w)
        return z_s, z_w

    # 添加预测异常分数的方法
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            # 数据增强
            x_s = self.aug_s(x)
            x_w = self.aug_w(x)
            # 异常注入
            x_s, _ = self.injector(x_s)
            x_w, _ = self.injector(x_w)
            # 异常检测预测
            y_s = self.head_s(self.encoder(x_s))
            y_w = self.head_w(self.encoder(x_w))
            # 计算最终分数
            score = self.w_s * y_s + self.w_w * y_w
        return score

def get_contextual_mlp(win_size, d_embed, d_con_rate, output_d_data, patch_size):
    d_con = int(d_embed * d_con_rate)
    contextual_model = nn.Sequential(
        nn.Linear(d_embed, d_con),
        nn.BatchNorm1d(win_size),
        nn.ReLU(),
        nn.Linear(d_con, output_d_data * patch_size)
    )

    nn.init.xavier_uniform_(contextual_model[0].weight)
    nn.init.zeros_(contextual_model[0].bias)
    nn.init.xavier_uniform_(contextual_model[3].weight)
    nn.init.zeros_(contextual_model[3].bias)
    return contextual_model

def get_DAT_model(input_d_data,
                  output_d_data,
                  patch_size,
                  d_embed=512,
                  hidden_dim_rate=4.,
                  max_seq_len=512,
                  positional_encoding=None,
                  relative_position_embedding=True,
                  transformer_n_layer=1,
                  transformer_n_head=8,
                  dropout=0.1,
                  d_pre_rate=2.,
                  d_con_rate=3.):
    hidden_dim = int(d_embed * hidden_dim_rate)

    s_linear_embedding = nn.Linear(input_d_data * patch_size, d_embed)
    w_linear_embedding = nn.Linear(input_d_data * patch_size, d_embed)

    transformer_encoder = get_transformer_encoder(d_embed=d_embed,
                                                  positional_encoding=positional_encoding,
                                                  relative_position_embedding=relative_position_embedding,
                                                  n_layer=transformer_n_layer,
                                                  n_head=transformer_n_head,
                                                  d_ff=hidden_dim,
                                                  max_seq_len=max_seq_len,
                                                  dropout=dropout)

    d_pre = int(d_embed * d_pre_rate)
    predictive_mlp = nn.Sequential(
        nn.Linear(d_embed, d_pre),
        nn.GELU(),
        nn.Linear(d_pre, output_d_data * patch_size)
    )

    s_contextual_mlp = get_contextual_mlp(max_seq_len, d_embed, d_con_rate, output_d_data, patch_size)
    w_contextual_mlp = get_contextual_mlp(max_seq_len, d_embed, d_con_rate, output_d_data, patch_size)

    nn.init.xavier_uniform_(s_linear_embedding.weight)
    nn.init.zeros_(s_linear_embedding.bias)
    nn.init.xavier_uniform_(w_linear_embedding.weight)
    nn.init.zeros_(w_linear_embedding.bias)

    nn.init.xavier_uniform_(predictive_mlp[0].weight)
    nn.init.zeros_(predictive_mlp[0].bias)
    nn.init.xavier_uniform_(predictive_mlp[2].weight)
    nn.init.zeros_(predictive_mlp[2].bias)

    return DAT(s_linear_embedding, w_linear_embedding,
               transformer_encoder, predictive_mlp,
               s_contextual_mlp, w_contextual_mlp,
               patch_size, max_seq_len)


