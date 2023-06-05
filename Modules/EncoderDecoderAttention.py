
# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.AttentionVariants import AttentionSimpleConcat, AttentionVariants
from Modules.EncoderDecoderBase import EncoderDecoderBase
ENCODER_DECODER_TYPE = 'Attention'

@EncoderDecoderBase.register_EncoderDecoder(encoder_decoder_type=ENCODER_DECODER_TYPE)
class EncoderDecoderAttention(EncoderDecoderBase):
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
            super(EncoderDecoderAttention.Encoder, self).__init__(
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


    class DecoderStep(nn.Module):
        def __init__(self,
                     attended_dim: int,
                     query_dim: int,
                     att_type: str
                     ):
            super(EncoderDecoderAttention.DecoderStep, self).__init__()

            # length of src element representation ; length of trg element representation
            self.att = AttentionVariants(attended_dim=attended_dim, query_dim=query_dim, att_type=att_type)
            self.mlp_out = AttentionSimpleConcat(in_features=attended_dim+query_dim, out_features=query_dim)


        def forward(self, src_out, decoder_out):

            # decoder_out is of shape (batch_size, 1, hidden_size)
            # src_out is of shape (batch_size, src_length, dirs*hidden_size)

            # compute attention for each src element
            src_length = src_out.shape[1]
            scores = []
            for i in range(src_length):

                # src of shape (batch_size, 1, dirs*hidden_size)
                # to_score of shape (batch_size, 1, dirs*hidden_size+hidden_size)
                src_element = src_out[:, i:i+1, :]

                # score of shape (batch_size, 1, 1)
                score = self.att(decoder_out, src_element)

                scores += [score]

            # scores, norm_scores of shape (src_length, batch_size, 1)
            scores = torch.cat(scores)
            norm_scores = F.softmax(scores, dim=0)

            sum = []
            for i in range(src_length):
                src_element = src_out[:, i:i+1, :]
                alpha_element = norm_scores[i:i+1,:, :]
                sum += [torch.mul(src_element, alpha_element)]
            c_t = torch.sum(torch.cat(sum), dim=0).unsqueeze(1)

            # att_output : (batch_size, 1, dirs*hidden_size)
            att_output = self.mlp_out(c_t, decoder_out)

            # norm_scores are returned for dumping
            return att_output, norm_scores


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
                     att_type: str
                     ):
            super(EncoderDecoderAttention.EncoderDecoderModel, self).__init__()

            # encoder_out dim decides on the decoder out state size (hidden or hidden*2)
            self.bidirectional = bidirectional
            self.bidirint = 1 if not bidirectional else 2
            self.eos_int = eos_int
            self.encoder = EncoderDecoderAttention.Encoder(
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
            self.decoder_step = EncoderDecoderAttention.DecoderStep(
                att_type=att_type,
                attended_dim=decoder_hidden_size,
                query_dim=decoder_hidden_size,
            )
            self.device = device
            self.decoder_hidden_size = decoder_hidden_size
            self.trg_vocab_size = trg_vocab_size

            # if testing time we won't use the trg length to stop the predictions of symbol, because we don't know

            self.lin = nn.Linear(in_features=decoder_hidden_size, out_features=trg_vocab_size)

        def forward(self, src, trg):

            # src is of shape: (batch_size, src_length), but batch_size = 1 in this exercise
            # trg is of shape: (batch_size, trg_length), but batch_size = 1 in this exercise

            # the encoded outputs are (batch_size, src_length, dirs*hidden_size)
            encoder_out, _, __ = self.encoder(src)

            # take the target first symbol to be the first predicted (sos)
            # (keep dimension for batch >= 1)
            decoder_out = torch.zeros(src.shape[0], 1, self.decoder_hidden_size).to(self.device)

            # outputs is going to be of shape (prediction_length, batch_size, vocab_size), for the loss
            # prediction_length is made of symbols (should learn to predict sos in it)

            # in training: loop until target length is met.
            # in inference: loop until eos symbol or max_prediction(without sos and eos)
            count_steps = 1
            outputs = []
            predictions = []
            att_weights = []

            while (count_steps < trg.shape[1]):

                # the decoder output is of shape (batch_size, 1, bidir*hidden_size)
                # weights of shape (src_length, batch_size, 1)
                decoder_out, weights = self.decoder_step(encoder_out, decoder_out)
                att_weights += [weights]

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
            att_weights = torch.cat(att_weights, dim=2)
            # att weights is of shape (src_length, 1, trg_length)
            return outputs, predictions, att_weights