import torch
import torch.nn as nn
from tt.transformer import RelLearnableDecoderLayer


class BaseEncoder(nn.Module):
    def __init__(self, hidden_size, output_size, k_len, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(BaseEncoder, self).__init__()

        self.r_emb = nn.Parameter(torch.Tensor(k_len, n_head, d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head))
        self.r_bias = nn.Parameter(torch.Tensor(k_len, n_head))

        self.MultiHeadAttention = RelLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, **kwargs)

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, input_lengths):

        assert inputs.dim() == 3

        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]

        outputs = self.MultiHeadAttention(inputs, self.r_emb, self.r_w_bias, self.r_bias)

        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs = outputs[desorted_indices]

        outputs = self.output_proj(outputs)

        return outputs


def build_encoder(config):
    if config.enc.type == 'attention':
        return BaseEncoder(
            hidden_size=config.enc.hidden_size,
            output_size=config.enc.output_size,
            k_len=config.enc.d_model,
            n_head=config.enc.n_head,
            d_model=config.enc.d_model,
            d_head=config.enc.d_head,
            d_inner=config.enc.d_inner,
            dropout=config.dropout
        )
    else:
        raise NotImplementedError
