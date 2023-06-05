
# import packages
import torch
import torch.nn as nn
from Modules.EncoderDecoderBase import EncoderDecoderBase
ENCODER_DECODER_TYPE = 'Basic'


@EncoderDecoderBase.register_EncoderDecoder(encoder_decoder_type=ENCODER_DECODER_TYPE)
class EncoderDecoderBasic(EncoderDecoderBase):
    def __init__(self):
        super(EncoderDecoderBase, self).__init__()

    # Encoder inherits from base
    class Encoder(EncoderDecoderBase.Encoder):
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
            super(EncoderDecoderBasic.Encoder, self).__init__(
                num_layers=num_layers,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                max_source_len=max_source_len,
                device=device,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                use_positional_embedding=use_positional_embedding
            )

    # RNN initializer inherits from Base
    class RNN_Initializer(EncoderDecoderBase.RNN_Initializer):
        def __init__(self, rnn_type: str,
                     num_layers: int,
                     bidirectional: bool,
                     hidden_size: int,
                     input_size: int,
                     device: torch.device):
            super(EncoderDecoderBasic.RNN_Initializer, self).__init__(rnn_type=rnn_type,
                                                                      num_layers=num_layers,
                                                                      bidirectional=bidirectional,
                                                                      hidden_size=hidden_size,
                                                                      input_size=input_size,
                                                                      device=device)

    class DecoderStep(nn.Module):
        def __init__(self,
                     encoder_num_layers: int,
                     decoder_num_layers: int,
                     encoder_hidden_size: int,
                     decoder_hidden_size: int,
                     vocab_size:int,
                     embedding_dim: int,
                     device: torch.device,
                     bidirectional: bool,
                     rnn_type: str
                     ):
            super(EncoderDecoderBasic.DecoderStep, self).__init__()

            self.device = device
            self.num_layers = decoder_num_layers
            self.bidirint = 1 if not bidirectional else 2

            # input size for each decoder step is a concatenation of the
            # encoder output (dirs*hidden_size)
            # and the previous step decoder
            # output (embedding_dim)
            self.input_size = embedding_dim + encoder_num_layers * self.bidirint * encoder_hidden_size

            self.token_embeddings = nn.Embedding(num_embeddings=vocab_size,
                                                 embedding_dim=embedding_dim)

            self.rnn_obj = EncoderDecoderBasic.RNN_Initializer(
                device=device,
                rnn_type=rnn_type,
                bidirectional=False,
                hidden_size=decoder_hidden_size,
                input_size=self.input_size,
                num_layers=self.num_layers
            )
            self.rnn = self.rnn_obj.init_rnn()


        def forward(self, src_state_out, prev_pred, state=None):

            # decoder will output a prediction one symbol at a time

            # prev_pred is previous step output, of shape (batch_size, 1)
            # state is previous decoder lstm hidden state, of shape (1,1,hidden_size) . None at the first step
            # src_state_out is the encoder last hidden, of shape (1, 1, hidden size)

            # init hidden states for lstm, get tuples with of shape (dirs * num_layers, batch_size, hidden_size)
            if state is None:
                state = self.rnn_obj.init_state()

            # first reshape the target prediction to embedding size
            # (batch_size, 1) -> (batch_size, 1, embedding_dim)
            prev_pred = self.token_embeddings(prev_pred)

            # second, concat the encoder output and the prev_pred,
            # concat ((batch_size, 1, dirs*1*hidden_size),(batch_size, 1, embedding_dim)) ->
            # (batch_size,1,hidden_size+embedding_dim)
            step_input = torch.cat((src_state_out, prev_pred), dim=2).to(self.device)

            # run through decoding stage, output is (batch_size, sequence_length, 1*hidden_size)
            # sequence length = 1 because we predict one symbol at a time.
            # state is a tuple of two (1 * num_layers, batch_size, hidden_size)
            output, state = self.rnn(step_input, state)

            # output : (batch_size, 1, 1*hidden_size)
            return output, state

    class EncoderDecoderModel(nn.Module):
        def __init__(self,
                     encoder_hidden_size: int,
                     decoder_hidden_size: int,
                     num_layers: int,
                     embedding_dim: int,
                     src_vocab_size: int,
                     trg_vocab_size: int,
                     max_source_len: int,
                     eos_int: int,
                     device: torch.device,
                     bidirectional: bool,
                     rnn_type: str,
                     encoder_decoder_type: str,
                     use_positional_embedding: bool,
                     att_type=None
                     ):
            super(EncoderDecoderBasic.EncoderDecoderModel, self).__init__()

            assert att_type is None, "att_type should be None for Basic encoder decoder, but {} was given".format(att_type)

            # encoder_out dim decides on the decoder out state size (hidden or hidden*2)
            self.bidirectional = bidirectional
            self.bidirint = 1 if not bidirectional else 2
            self.eos_int = eos_int
            self.encoder = EncoderDecoderBasic.Encoder(
                num_layers=num_layers,
                hidden_size=encoder_hidden_size,
                vocab_size=src_vocab_size,
                embedding_dim=embedding_dim,
                max_source_len=max_source_len,
                device=device,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                use_positional_embedding=use_positional_embedding
            )
            self.decoder_step = EncoderDecoderBasic.DecoderStep(
                encoder_num_layers=num_layers,
                decoder_num_layers=1,
                encoder_hidden_size=encoder_hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                vocab_size=trg_vocab_size,
                embedding_dim=embedding_dim,
                device=device,
                bidirectional=bidirectional,
                rnn_type=rnn_type
            )
            self.device = device
            self.trg_vocab_size = trg_vocab_size
            # if testing time we won't use the trg length to stop the predictions of symbol, because we don't know

            self.lin = nn.Linear(in_features=decoder_hidden_size, out_features=trg_vocab_size)

        def forward(self, src, trg):

            # src is of shape: (batch_size, src_length), but batch_size = 1 in this exercise
            # trg is of shape: (batch_size, trg_length), but batch_size = 1 in this exercise

            # the encoder last states are of shapes (dirs * num_layers, batch_size, hidden_size)
            # I reshape it to (1, 1, hidden size) , I made hidden size adjusted to the number of layers and bidir option
            _, encoder_last_state, encoder_last_state_r = self.encoder(src)
            if isinstance(encoder_last_state, tuple):
                encoder_last_state, _ = encoder_last_state  #take only the hidden and not the cell state
            encoder_last_state = encoder_last_state.reshape(1,1,-1)
            if encoder_last_state_r is not None:
                if isinstance(encoder_last_state_r, tuple):
                    encoder_last_state_r, _ = encoder_last_state_r  # take only the hidden and not the cell state
                encoder_last_state_r = encoder_last_state_r.reshape(1,1,-1)
                encoder_last_state = torch.cat((encoder_last_state, encoder_last_state_r), dim=2).to(self.device)

            # take the target first symbol to be the first predicted (sos)
            # (keep dimension for batch >= 1)
            state = None
            next_pred = trg[:, 0:1]

            # outputs is going to be of shape (prediction_length, batch_size, vocab_size), for the loss
            # prediction_length is made of symbols (should learn to predict sos in it)

            # in training: loop until target length is met.
            # in inference: loop until eos symbol or max_prediction(without sos and eos)
            count_steps = 1
            outputs = []
            predictions = []

            while (count_steps < trg.shape[1]):

                # the decoder output is of shape (batch_size, 1, 1*hidden_size)
                decoder_out, state = self.decoder_step(encoder_last_state, next_pred, state)

                # next things is to predict the symbol for this step
                model_out_step = self.lin(decoder_out)
                next_pred = torch.argmax(model_out_step, dim=2)

                # the prediction is saved for the loss
                outputs += [model_out_step]
                predictions += [next_pred]
                count_steps += 1

            # outputs is a list of prediction_length tensors, each tensor of shape (batch_size, 1, trg_vocab_size)
            # make it (prediction_length, batch_size, trg_vocab_size)
            # note: for batch size > 1 I think I will have to change to batch_first=False, so that the dim=1 will
            # hold the batches and not the prediction_length (now both 1 so it doesn't matter)
            # also, the condition for training stop will have to change to do one example at a time
            outputs = torch.cat(outputs)
            predictions = torch.cat(predictions)
            return outputs, predictions, None



