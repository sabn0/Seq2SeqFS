
# import packages
import torch
import torch.nn as nn


class AutomaticTransformer(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 N_encoder_blocks: int,
                 ff_dim: int,
                 device: torch.device,
                 N_decoder_blocks: int,
                 src_vocab_size: int,
                 src_max_size: int,
                 trg_vocab_size: int,
                 trg_max_size: int,
                 dropout=0.0,
                 activation='relu',
                 batch_first=True,
                 ):
        super(AutomaticTransformer, self).__init__()
        assert embedding_dim % num_heads == 0, "num_heads cannot be divided by embedding dimension"
        assert N_encoder_blocks > 0, "invalid number of encoder blocks requested"
        assert N_decoder_blocks > 0, "invalid number of decoder blocks requested"

        self.device = device

        self.src_embeddings = nn.Embedding(src_vocab_size, embedding_dim=embedding_dim)
        self.src_positional = nn.Embedding(src_max_size, embedding_dim=embedding_dim)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, embedding_dim=embedding_dim)
        self.trg_positional = nn.Embedding(trg_max_size, embedding_dim=embedding_dim)

        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=N_encoder_blocks,
            num_decoder_layers=N_decoder_blocks,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            activation=activation
        )

        self.lin = nn.Linear(embedding_dim, trg_vocab_size)

    def forward(self, src, trg):

        src = self.src_positional(torch.arange(src.shape[1]).expand(src.shape[0], src.shape[1]).to(self.device)) + self.src_embeddings(src)
        trg = self.trg_positional(torch.arange(trg.shape[1]).expand(trg.shape[0], trg.shape[1]).to(self.device)) + self.trg_embeddings(trg)
        out = self.transformer(src, trg)

        # move to linear from output -> (batch_size, trg_len, trg_vocab)
        out = self.lin(out)
        predictions = out.argmax(dim=2)

        return out, predictions


