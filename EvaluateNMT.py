
# import packages
import os
import argparse
import pickle
from LoadData import load_dataset
import torch
import torch.nn as nn
import torch.utils.data as data
from Modules.EncoderDecoderBasic import EncoderDecoderBasic
from Modules.EncoderDecoderAttention import EncoderDecoderAttention
from Modules.EncoderDecoderBase import EncoderDecoderBase
from TrainNMT import calc_bleu, evaluate


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--SourceTest', type=str, required=True, help="path to source test file")
    parser.add_argument('-f', '--TargetTest', type=str, required=True, help="path to target test file")
    parser.add_argument('-m', '--SaveModel', default='CheckpointFiles', help='path to save kwargs and model checkpoint')
    parser.add_argument('-l', '--SaveProcess', default=None, help='path to log folder (save process)')
    args = parser.parse_args()

    assert args.SaveModel is not None, "Must insert a location for saved models with args.SaveModel but inserted None"

    # set hyper-parameters and other variables
    batch_size = 1
    sos_symbol = '<s>'
    eos_symbol = '</s>'
    unk_symbol = '<UNK>'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get the load kwargs from training
    load_args = [sos_symbol, eos_symbol, unk_symbol]
    load_kwargs_file = os.path.join(args.SaveModel, 'load_kwargs.pkl')
    with open(load_kwargs_file, 'rb') as f:
        load_kwargs = pickle.load(f)

    # load the test set
    test_loader, _ = load_dataset(args.SourceTest, args.TargetTest, batch_size, False, *load_args, **load_kwargs)

    # init model and load the state from training
    checkpoint_path = os.path.join(args.SaveModel, 'checkpoint')
    checkpoint = torch.load(checkpoint_path)

    model_kwargs_file = os.path.join(args.SaveModel, 'model_kwargs.pkl')
    with open(model_kwargs_file, 'rb') as f:
        model_kwargs = pickle.load(f)
    model = EncoderDecoderBase.createInstance(encoder_decoder_type=model_kwargs['encoder_decoder_type']).EncoderDecoderModel(**model_kwargs)
    model.load_state_dict(checkpoint['model'])

    # init criterion and load the state from training
    criterion = nn.CrossEntropyLoss()
    criterion.load_state_dict(checkpoint['criterion'])

    # test
    bleu_test, loss_test, test_time_elapse = evaluate(
        dev_loader=test_loader,
        model=model,
        criterion=criterion,
        device=device,
        int2target=load_kwargs['int2target'],
        int2source=load_kwargs['int2source']
    )
    print("test bleu: {}, test loss: {}".format(bleu_test, loss_test))

    # save test
    process = '\t'.join([
        "positional:{}",
        "encoder_decoder_type:{}",
        "num_layers:{}",
        "encoder_type:{}",
        "att_type:{}",
        "bidirectional:{}",
        "embedding_dim:{}",
        "encoder_hidden_size:{}",
        "decoder_hidden_size:{}",
        "test_bleu:{}",
        "test_loss:{}",
        "test_time:{}"
    ]).format(
        str(model_kwargs['use_positional_embedding']), model_kwargs['encoder_decoder_type'], model_kwargs['num_layers'],
        model_kwargs['rnn_type'], model_kwargs['att_type'], str(model_kwargs['bidirectional']),
        model_kwargs['embedding_dim'], model_kwargs['encoder_hidden_size'], model_kwargs['decoder_hidden_size'],
        bleu_test, loss_test, test_time_elapse
    )

    if args.SaveProcess is not None:
        log_file = os.path.join( args.SaveProcess, 'log.test.{}.txt'.format(model_kwargs['encoder_decoder_type']))
        with open(log_file, 'a+') as f:
            f.write("{}\n".format(process))


if __name__ == "__main__":
    main()