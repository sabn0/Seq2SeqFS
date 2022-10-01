
# import packages
import torch
import torch.nn as nn

# a self implementation of attention is all you need
# partially assisted by https://github.com/aladdinpersson/Machine-Learning-Collection


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        pass

    def forward(self, queries, keys, values, mask):

        # queries, keys, values are of shape: (batch_size, seq_length, num_heads, embedding_dim // num_heads)
        # seq_length can be different for values than it is for queries/keys

        # the mask is used during decoding to avoid left directed flow (set to -inf)
        # during encoding the mask doesn't do anything

        # formula => softmax(queries*keys_T /(sqrt(keys_dim)))*values
        # making att of shape: (batch_size, values_len, num_heads, embedding_dim // num_heads)
        att = torch.einsum(
            'bhqk,bvhl->bvhl',
            torch.softmax(
                (
                        torch.einsum('bqhl,bkhl->bhqk', [queries, keys]) / (keys.shape[3]**0.5)
                ).masked_fill(mask == 0, float('-1e10')),
                dim=2
            ),
            values
        )

        return att


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()

        assert embedding_dim % num_heads == 0, "can't divide embedding_dim {} by num_heads {}".format(embedding_dim, num_heads)
        self.num_heads = num_heads

        self.queries_lin = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.keys_lin = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.values_lin = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.dp_att = ScaledDotProductAttention()
        self.out_lin = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, queries, keys, values, mask):

        # queries, keys and values are of dimension (batch_size, seq_length, embedding_dim)
        # linear projection keeps dimensions
        queries = self.queries_lin(queries)
        keys = self.keys_lin(keys)
        values = self.values_lin(values)

        # reshape for heads
        queries = queries.reshape(queries.shape[0], queries.shape[1], self.num_heads, queries.shape[2] // self.num_heads)
        keys = keys.reshape(keys.shape[0], keys.shape[1], self.num_heads, keys.shape[2] // self.num_heads)
        values = values.reshape(values.shape[0], values.shape[1], self.num_heads, values.shape[2] // self.num_heads)

        # send to DP attention, output of shape (batch_size, values_len, num_heads, embedding_dim // num_heads)
        att_out = self.dp_att(queries, keys, values, mask)

        # concat and final linear
        att_out = self.out_lin(att_out.reshape(values.shape[0], values.shape[1], -1))
        return att_out


class AddAndNorm(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float):
        super(AddAndNorm, self).__init__()

        self.norm = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x, z):

        # x is of shape : (batch_size, seq_length, embedding_dim)
        # z is of shape : (batch_size, seq_length, embedding_dim) , after attention
        # output stays the same shape
        x = self.norm(x + self.drop(z))
        return x


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ff_dim: int):
        super(FeedForward, self).__init__()

        self.in_lin = nn.Linear(in_features=embedding_dim, out_features=ff_dim)
        self.relu = nn.ReLU()
        self.out_lin = nn.Linear(in_features=ff_dim, out_features=embedding_dim)

    def forward(self, x):

        # x is of shape : (batch_size, seq_length, embedding_dim), out is the same
        x = self.out_lin(self.relu(self.in_lin(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim:int, num_heads: int, dropout: float, ff_dim: int):
        super(EncoderBlock, self).__init__()

        self.multi_head_att = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.add_norm = AddAndNorm(embedding_dim=embedding_dim, dropout=dropout)
        self.ff = FeedForward(embedding_dim=embedding_dim, ff_dim=ff_dim)

    def forward(self, x, y, z, mask):

        # x, y, z is of shape : (batch_size, seq_length, embedding_dim)
        out_att = self.multi_head_att(x, y, z, mask)
        out_norm = self.add_norm(z, out_att)
        out = self.ff(out_norm)
        out_enc = self.add_norm(out_norm, out)
        return out_enc


class Encoder(nn.Module):
    def __init__(self,
                 max_src_len: int,
                 max_vocab_len: int,
                 embedding_dim: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout: float,
                 device: torch.device,
                 N_blocks: int):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(max_vocab_len, embedding_dim=embedding_dim)
        self.positions = nn.Embedding(max_src_len, embedding_dim=embedding_dim)
        self.N_blocks = N_blocks
        self.device = device
        self.encoder_block = EncoderBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout
        )

    def forward(self, src, mask):

        # src is of shape: (batch_size, src_length)
        # int-based embeddings over original sinus embeddings
        positions = torch.arange(src.shape[1]).expand(src.shape[0], src.shape[1]).to(self.device)
        enc_src = self.embedding(src) + self.positions(positions)

        # run over encoding blocks
        for _ in range(self.N_blocks):
            enc_src = self.encoder_block(enc_src, enc_src, enc_src, mask)

        return enc_src


class ManualTransformer(nn.Module):
    def __init__(self,
                 trg_vocab_size: int,
                 trg_max_size: int,
                 src_vocab_size: int,
                 src_max_size: int,
                 embedding_dim: int,
                 N_encoder_blocks: int,
                 N_decoder_blocks: int,
                 num_heads: int,
                 device: torch.device,
                 dropout: float,
                 ff_dim: int):
        super(ManualTransformer, self).__init__()

        self.device = device

        # for src
        self.encoder = Encoder(
            N_blocks=N_encoder_blocks,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            device=device,
            ff_dim=ff_dim,
            dropout=dropout,
            max_src_len=src_max_size,
            max_vocab_len=src_vocab_size
        )

        # for trg
        self.decoder_N_blocks = N_decoder_blocks
        self.embedding = nn.Embedding(trg_vocab_size, embedding_dim=embedding_dim)
        self.positions = nn.Embedding(trg_max_size, embedding_dim=embedding_dim)
        self.multi_head_att = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.add_norm = AddAndNorm(embedding_dim=embedding_dim, dropout=dropout)
        self.decoder_block = EncoderBlock(
            num_heads=num_heads,
            ff_dim=ff_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        # linear and softmax
        self.out_lin = nn.Linear(in_features=embedding_dim, out_features=trg_vocab_size)

    def forward(self, src, trg):

        # src is of shape: (batch_size, src_length)
        # trg is of shape: (batch_size, trg_length)

        # mask prevents left dir flow for trg
        src_mask = torch.ones((src.shape[0], src.shape[1], src.shape[1])).to(self.device)
        trg_mask = torch.tril(torch.full((trg.shape[1], trg.shape[1]), float("1"))).to(self.device)

        enc_src = self.encoder(src, src_mask)
        positions = torch.arange(trg.shape[1]).expand(trg.shape[0], trg.shape[1]).to(self.device)
        enc_trg = self.embedding(trg) + self.positions(positions)

        # run over decoding blocks
        out = None
        for _ in range(self.decoder_N_blocks):

            out_att = self.multi_head_att(enc_trg, enc_trg, enc_trg, trg_mask)
            out_norm = self.add_norm(enc_trg, out_att)
            out = self.decoder_block(enc_src, enc_src, out_norm, src_mask)

        # out : (batch_size, trg_len, trg_vocab)
        # predictions: (batch_size, trg_len)
        out = self.out_lin(out)
        out = torch.softmax(out, dim=2)
        predictions = out.argmax(dim=2)

        return out, predictions
