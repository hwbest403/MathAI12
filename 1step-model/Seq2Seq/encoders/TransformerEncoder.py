import torch
import torch.nn as nn
from functools import partial
from Seq2Seq.modules import MultiHeadAttentionLayer
from Seq2Seq.modules import PositionwiseFeedforwardLayer
from Seq2Seq.modules import positional_encoding


class TransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 max_length,
                 device,
                 vectors=None):
        super().__init__()

        if vectors is not None:
            self.tok_embedding = nn.Embedding.from_pretrained(vectors, freeze=False)
        else:
            self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = partial(positional_encoding, d_model=hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.hid_dim = hid_dim

    @classmethod
    def from_args(cls, args, vocabs, device):
        return cls(
            len(vocabs['src']),
            args.hid_dim,
            args.enc_layers,
            args.enc_heads,
            args.enc_pf_dim,
            args.enc_dropout,
            args.max_length,
            device,
            vocabs['src'].vectors if args.vectors else None
        )

    def forward(self, src):

        src_mask = self.make_src_mask(src)

        # src = [batch size, src len]
        # src_mask = [batch size, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = self.pos_embedding(src_len).to(self.device)
        # pos = [batch size, src len]

        scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(src.device)
        src = self.dropout((self.tok_embedding(src) * scale) + pos)
        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]

        return src

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != 1).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device
                 ):
        super().__init__()

        # self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        # self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        # src = self.self_attn_layer_norm(src + self.dropout(_src))
        src = self.layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        # src = self.ff_layer_norm(src + self.dropout(_src))
        src = self.layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src
