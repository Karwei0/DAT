# -*- coding = utf-8 -*-
# @Time: 2025/2/13 14:56
# @Author: wisehone
# @File: transformer.py
# @SoftWare: PyCharm
import torch
from torch import nn
import torch.nn.functional as F

from utils.utils import clone_encoder

import numpy as np
from timm.layers import trunc_normal_

class TransformerEncoder(nn.Module):
    def __init__(self, positional_encoding_layer, one_encoder_layer, n_layer):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = clone_encoder(one_encoder_layer, n_layer) # serial the encoder layer

        self.is_positional_encoding = positional_encoding_layer is not None
        if self.is_positional_encoding is not None:
            self.positional_encoding_layer = positional_encoding_layer

    def forward(self, x):
        if self.is_positional_encoding:
            out = self.positional_encoding_layer(x)
        else:
            out = x

        for encoder in self.encoder_layers:
            out = encoder(out)
        return out

# one encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, feed_forward_layer, norm_layer, dropout=0.1):
        """
        :param attention_layer:
        :param feed_forward_layer:
        :param norm_layer: a list with two normlayers
        :param dropout:
        """
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layer = clone_encoder(norm_layer, 2)

        self.dropout = nn.Dropout(dropout)

        for parameters in self.attention_layer.parameters():
            if parameters.dim() > 1:
                nn.init.xavier_uniform_(parameters)
            else:
                nn.init.zeros_(parameters)
        for parameters in self.feed_forward_layer.parameters():
            if parameters.dim() > 1:
                nn.init.xavier_uniform_(parameters)
            else:
                nn.init.zeros_(parameters)

    def forward(self, x):
        out1 = self.norm_layer[0](x) # pre norm
        out1 = self.attention_layer(out1)
        out1 = self.dropout(out1) + x

        out2 = self.norm_layer[1](out1)
        out2 = self.feed_forward_layer(out2)
        out2 = self.dropout(out2) + out1

        return out2

class MulitHeadAttentionLayer(nn.Module):
    def __init__(self, d_embed, n_head, max_seq_len, relative_position_embedding=True):
        super(MulitHeadAttentionLayer, self).__init__()
        assert d_embed % n_head == 0

        self.d_embed = d_embed
        self.n_head = n_head
        self.d_k = d_embed // n_head
        self.scale = 1 / np.sqrt(self.d_k)
        self.max_seq_len = max_seq_len
        self.relative_position_embedding = relative_position_embedding

        self.word_fc_layers = clone_encoder(nn.Linear(d_embed, d_embed), 3)
        self.out_fc_layer = nn.Linear(d_embed, d_embed)

        if relative_position_embedding:
            # Table of 1D relative position embedding
            self.relative_position_embedding_table = nn.Parameter(torch.randn(2 * max_seq_len - 1, n_head))
            trunc_normal_(self.relative_position_embedding_table, std=.02)

            # Set 1D relative position embedding index.
            coords_h = np.arange(max_seq_len)
            coords_w = np.arange(max_seq_len - 1, -1, -1)
            coords = coords_h[:, None] + coords_w[None, :]  # [None, :] => extends to 2D
            self.relative_position_index = coords.flatten()

    def forward(self, x):
        n_batch = x.shape[0]
        device = x.device

        query = self.word_fc_layers[0](x)
        key = self.word_fc_layers[1](x)
        value = self.word_fc_layers[2](x)

        # split head
        query_out = query.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        key_out = key.view(n_batch, -1, self.n_head, self.d_k).contiguous().permute(0, 2, 3, 1)
        value_out = value.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)

        # Compute attention and concatenate matrices.
        scores = torch.matmul(query_out * self.scale, key_out)

        if self.relative_position_embedding:
            position_embedding = self.relative_position_embedding_table[self.relative_position_index].view(
                self.max_seq_len, self.max_seq_len, -1)
            position_embedding = position_embedding.permute(2, 0, 1).contiguous().unsqueeze(0)

            scores = scores + position_embedding

        probs = F.softmax(scores, dim=-1)
        att_out = torch.matmul(probs, value_out)

        # Convert 4d tensor to proper 3d output tensor.
        att_out = att_out.transpose(1, 2).contiguous().view(n_batch, -1, self.d_embed)

        return self.out_fc_layer(att_out)

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = nn.Linear(d_embed, d_ff)
        self.second_fc_layer = nn.Linear(d_ff, d_embed)
        self.dropout = nn.Dropout(dropout)
        self.activative_layer = nn.GELU()

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = self.activative_layer(out)
        out = self.dropout(out)

        out = self.second_fc_layer(out)
        return out

# Sinusoidal positional encoding
class SinusoidPositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len, dropout=0.1):
        super(SinusoidPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        denominators = torch.exp(torch.arange(0, d_embed, 2) * (np.log(0.0001) / d_embed)).unsqueeze(0)
        encoding_matrix = torch.matmul(positions, denominators)

        encoding = torch.empty(1, max_seq_len, d_embed)
        encoding[0, :, 0::2] = torch.sin(encoding_matrix)
        encoding[0, :, 1::2] = torch.cos(encoding_matrix[:, :(d_embed // 2)])

        self.register_buffer('encoding', encoding)

    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_embed) == (*, max_seq_len, d_embed) (default)
        """
        return self.dropout(x + self.encoding)

# Absolute position embedding
class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, d_embed, max_seq_len, dropout=0.1):
        super(AbsolutePositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_embed))
        trunc_normal_(self.embedding, std=.02)

    def forward(self, x):
        return self.dropout(x + self.embedding)

def get_transformer_encoder(d_embed,
                            n_head=8,
                            positional_encoding=None,
                            relative_position_embedding=True,
                            n_layer=1,
                            d_ff=2028,
                            max_seq_len=512,
                            dropout=0.1):
    if positional_encoding in ('Sinusoidal', 'sinusoidal', 'sin'):
        positional_encoding = SinusoidPositionalEncoding(d_embed, max_seq_len, dropout)
    elif positional_encoding in ('Absolute', 'absolute', 'abs'):
        positional_encoding = AbsolutePositionEmbedding(d_embed, max_seq_len, dropout)
    elif positional_encoding is None or positional_encoding == 'None':
        positional_encoding = None

    attention_layer = MulitHeadAttentionLayer(d_embed, n_head, max_seq_len, relative_position_embedding)
    feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
    encoder_layer = EncoderLayer(attention_layer, feed_forward_layer, norm_layer, dropout)

    return TransformerEncoder(positional_encoding, encoder_layer, n_layer)

