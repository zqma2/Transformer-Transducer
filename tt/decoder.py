import torch
import torch.nn as nn
from tt.transformer import RelLearnableDecoderLayer


class BaseDecoder(nn.Module):

    def __init__(self, vocab_size, n_layer, k_len, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(BaseDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.r_emb = nn.Parameter(torch.Tensor(k_len, n_head, d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head))
        self.r_bias = nn.Parameter(torch.Tensor(k_len, n_head))
#       self.embedding = self.generate_embedding(hidden_size)

        self.MultiHeadAttention = RelLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, **kwargs)

#        if share_weight:
#            self.embedding.weight = self.output_proj.weight

    def generate_embedding(self, hidden_size):
        sym_list = []
        idx_list = []
        with open('vocab.txt', 'r') as fid:
            for line in fid:
                sym, idx = line.strip().split(' ')
                sym_list.append(sym)
                idx_list.append(idx)
                idx_list = list(map(int, idx_list))
                embedding = torch.eye(hidden_size)[idx_list, :].cuda()
        return embedding

    def forward(self, inputs, seq_length=None, hidden=None, mems=None):

        qlen, dez = inputs.size(0), inputs.size(1)
        embed_inputs = self.embedding(inputs)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        attn_mask = torch.triu(inputs.new_ones(qlen, klen), diagonal=1+mlen).bool()[:, :, None]

        outputs = self.MultiHeadAttention(embed_inputs, self.r_emb, self.r_w_bias, self.r_bias, attn_mask)

        return outputs


def build_decoder(config):
    if config.dec.type == 'attention':
        return BaseDecoder(
            vocab_size=config.vocab_size,
            n_layer=config.dec.n_layer,
            k_len=config.dec.d_model,
            n_head=config.dec.n_head,
            d_model=config.dec.d_model,
            d_head=config.dec.d_head,
            d_inner=config.dec.d_inner,
            dropout=config.dropout
        )
    else:
        raise NotImplementedError
