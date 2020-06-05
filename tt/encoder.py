import torch
import torch.nn as nn
from rnnt.transformer import RelLearnableDecoderLayer


class BaseEncoder(nn.Module):
    def __init__(self, k_len, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(BaseEncoder, self).__init__()

        self.r_emb = nn.Parameter(torch.Tensor(k_len, n_head, d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head))
        self.r_bias = nn.Parameter(torch.Tensor(k_len, n_head))

        self.MultiHeadAttention = RelLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, **kwargs)

    def forward(self, inputs, input_lengths, enc_attn_mask=None):

        assert inputs.dim() == 3

        outputs = self.MultiHeadAttention(inputs, self.r_emb, self.r_w_bias, self.r_bias, enc_attn_mask)

        return outputs


class BuildEncoder(nn.Module):
    def __init__(self, config):
        super(BuildEncoder, self).__init__()

        self.layers = nn.ModuleList([BaseEncoder(
            k_len=config.enc.d_model,
            n_head=config.enc.n_head,
            d_model=config.enc.d_model,
            d_head=config.enc.d_head,
            d_inner=config.enc.d_inner,
            dropout=config.dropout)
            for i in range(config.enc.n_layer)])

    def forward(self, inputs, input_lengths):
        for layer in self.layers:
            x = layer(inputs, input_lengths)

        return x


def build_encoder(config):
    if config.enc.type == 'attention':
        return BaseEncoder(
            n_layer=config.enc.n_layer,
            k_len=config.enc.d_model,
            n_head=config.enc.n_head,
            d_model=config.enc.d_model,
            d_head=config.enc.d_head,
            d_inner=config.enc.d_inner,
            dropout=config.dropout
        )
    else:
        raise NotImplementedError
