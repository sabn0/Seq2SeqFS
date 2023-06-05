
import torch
import torch.nn as nn
from abc import ABCMeta

# Encoder implementation is equal for both basic and attention encoder-decoders
# Decoder Step is implemented differently at the instance level
# EncoderDecoderModel is implemented differently at the instance level

# Implemented as a decorator


class EncoderDecoderBase(nn.Module):

    encoder_decoder_types = {}

    def _init__(self):
        pass

    @classmethod
    def register_EncoderDecoder(cls, encoder_decoder_type):
        def decorator(encoder_decoder):
            cls.encoder_decoder_types[encoder_decoder_type] = encoder_decoder
            return encoder_decoder
        return decorator

    @classmethod
    def createInstance(cls, encoder_decoder_type):
        if encoder_decoder_type not in cls.encoder_decoder_types:
            raise ValueError
        encoder_decoder = cls.encoder_decoder_types[encoder_decoder_type]()
        return encoder_decoder

    class EncoderDecoderModel(metaclass=ABCMeta):
        pass

    class DecoderStep(metaclass=ABCMeta):
        pass

    class RNN_Initializer:
        def __init__(self, rnn_type: str,
                     num_layers: int,
                     bidirectional: bool,
                     hidden_size: int,
                     input_size: int,
                     device: torch.device,
                     batch_size=1):

            self.rnn_type = rnn_type
            self.bidirint = 1 if not bidirectional else 2
            self.state_shape = (num_layers*self.bidirint, batch_size, hidden_size)
            self.device = device
            self.rnn_kwargs = {
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'batch_first': True,
                'bidirectional': bidirectional
            }

        # factory
        def init_rnn(self):
            if self.rnn_type == 'LSTM':
                return nn.LSTM(**self.rnn_kwargs)
            elif self.rnn_type == 'GRU':
                return nn.GRU(**self.rnn_kwargs)
            elif self.rnn_type == 'Vanilla':
                return nn.RNN(**self.rnn_kwargs)
            else:
                raise ValueError("option are LSTM, GRU or Vanila, but {} was given".format(self.rnn_type))

        def init_state(self):
            h0 = torch.zeros(self.state_shape).to(self.device)
            if self.rnn_type == 'LSTM':
                c0 = torch.zeros(self.state_shape).to(self.device)
                return (c0, h0)
            return h0

    class Encoder(nn.Module):
        def __init__(self,
                     num_layers: int,
                     hidden_size: int,
                     vocab_size: int,
                     embedding_dim: int,
                     max_source_len: int,
                     device: torch.device,
                     bidirectional: bool,
                     rnn_type: str,
                     use_positional_embedding: bool):
            super(EncoderDecoderBase.Encoder, self).__init__()

            self.rnn_type = rnn_type
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.bidirectional = bidirectional
            self.use_positional_embedding = use_positional_embedding
            self.device = device
            self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
            self.positions_embeddings = nn.Embedding(num_embeddings=max_source_len, embedding_dim=embedding_dim)

            self.rnn_obj = EncoderDecoderBase.RNN_Initializer(
                device=device,
                rnn_type=rnn_type,
                bidirectional=False,
                hidden_size=hidden_size,
                input_size=embedding_dim,
                num_layers=num_layers
            )
            self.rnn = self.rnn_obj.init_rnn()


        def forward(self, src):
            # src is of shape: (batch_size, sequence_length), but batch_size = 1 in this exercise
            # reshape to : (batch_size, sequence_length, embedding_dim)
            src = self.token_embeddings(src)
            if self.use_positional_embedding:
                positions = torch.arange(0, src.shape[1]).int().expand(src.shape[0], src.shape[1]).to(self.device)
                src += self.positions_embeddings(positions)

            # init hidden states for lstm, get tuples with of shape (dirs * num_layers, batch_size, hidden_size)
            state = self.rnn_obj.init_state()

            # run through layers of LSTM model
            # output is of shape (batch_size, sequence_length, dirs*hidden_size)
            # the last hidden state is of shape (dirs * num_layers, batch_size, hidden_size)
            output, state = self.rnn(src, state)

            state_r = None
            if self.bidirectional:
                src_r = torch.flip(src, (1,)).to(self.device)
                state_r = self.rnn_obj.init_state()
                output_r, state_r = self.rnn(src_r, state_r)
                output_r = torch.flip(output_r, (1,)).to(self.device)
                output = torch.cat((output, output_r), dim=2).to(self.device)

            # the states will be used in the decoder stage, concatenated to the input target at every position
            return output, state, state_r
