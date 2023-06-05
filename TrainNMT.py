
# import packages
import time
import os
import argparse
import pickle
from LoadData import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from Modules.EncoderDecoderBasic import EncoderDecoderBasic
from Modules.EncoderDecoderAttention import EncoderDecoderAttention
from Modules.EncoderDecoderBase import EncoderDecoderBase
from sacrebleu.metrics import BLEU


def calc_bleu(targets: list, predictions: list, int2target: dict):

    # targets, predictions are lists of tensors (for every pair of target - pred outputs)
    target_in_symbols = []
    prediction_in_symbols = []
    bleu_obj = BLEU()

    for i , (target, prediction) in enumerate(zip(targets, predictions)):

        target_in_symbol = ' '.join([int2target[i.item()] for i in target[:-1]])
        target_in_symbols += [target_in_symbol]

        prediction_in_symbol = ' '.join([int2target[i.item()] for i in prediction[:-1]])
        prediction_in_symbols += [prediction_in_symbol]

    bleu = bleu_obj.corpus_score(hypotheses=prediction_in_symbols, references=[target_in_symbols])
    return bleu.score


def evaluate(
        dev_loader: data.DataLoader,
        model: nn.Module,
        criterion: nn.CrossEntropyLoss,
        device: torch.device,
        int2target:dict,
        int2source: dict,
        save_dump=None
) -> tuple:

    # set model to evaluation mode
    model.eval()

    bleu_dev = loss_dev = total = 0
    predictions = []
    targets = []
    example_dump = None

    dev_time_elapse = time.time()

    dumped_example_src = dumped_example_trg = ''

    for j, (src, trg) in enumerate(dev_loader):

        # see training function for documentation
        src, trg = src.to(device), trg.to(device)
        output, prediction, dump = model(src, trg)

        # dump is the attention weights when using the attention mechanism
        # of shape (src_length, 1, trg_length)
        # it is used for the same -single- dev set example for each epoch
        # it takes the first example, and I don't shuffle the dev set

        trg = trg[0, 1:].reshape(-1)
        output = output.reshape(-1, output.shape[2])
        prediction = prediction.reshape(-1)
        assert trg.shape[0] == prediction.shape[0] == output.shape[0]

        predictions += [prediction]
        targets += [trg]

        # calculate the loss and opt step for each predicted token
        losses = []
        for k in range(min(len(output), len(trg))):
            losses += [criterion(output[k], trg[k])]
        loss = sum(losses) / len(losses)
        loss_dev += loss.item()
        total += src.shape[0]

        # dumping example
        if j == 0:
            src = src.reshape(-1)
            example_dump = dump
            dumped_example_trg = ' '.join([int2target[i.item()] for i in trg])
            dumped_example_src = ' '.join([int2source[i.item()] for i in src])

    dev_time_elapse = time.time() - dev_time_elapse

    # calculate bleu for the prediction
    bleu_dev += calc_bleu(predictions=predictions, targets=targets, int2target=int2target)

    if save_dump is not None:
        torch.save(example_dump, save_dump)
        with open('Plotters/Heatmap.labels.txt', 'w+') as f:
            f.write('{}\n'.format(dumped_example_src))
            f.write('{}\n'.format(dumped_example_trg))

    model.train()
    return bleu_dev, loss_dev/total, dev_time_elapse


def train(train_loader: data.DataLoader,
          dev_loader: data.DataLoader,
          learning_rate: float,
          max_iter: int,
          device: torch.device,
          model: nn.Module,
          param_str: str,
          int2target: dict,
          int2source: dict,
          save_dump: object,
          save_model_dir: str,
          log_file=None,
          ) -> tuple:

    checkpoint = os.path.join(save_model_dir, 'checkpoint')

    # define Adam optimizer and cross entropy loss
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # define lists for history of training process
    train_loss, train_bleu, dev_loss, dev_bleu = [],[],[],[]

    # set model to training mode
    model.train()

    # save best bleu on dev
    best_bleu_dev = -1

    # training loop
    for i in range(max_iter):

        iter_bleu = iter_loss = total = 0
        predictions, targets = [],[]

        iter_time_elapse = time.time()

        for j, (src, trg) in enumerate(train_loader):

            # src is of shape: (batch_size, src_length), batch_size = 1 in this exercise
            # trg is of shape: (batch_size, trg_length), batch_size = 1 in this exercise

            # if I want to use this code on GPU later
            src, trg = src.to(device), trg.to(device)
            opt.zero_grad()

            # pass through model, output is of shape (prediction_length, batch_size, trg_vocab_size)
            output, prediction, _ = model(src, trg)

            # cross entropy loss expects trg to be (num_classes,) and output to be (batch_size, num_classes)
            # so I reshape the trg: (batch_size, trg_length-1) -> (trg_length-1)
            # and the output: (prediction_length, batch_size, trg_vocab_size) -> (prediction_length, trg_vocab_size)
            # on training, trg_length-1 = prediction_length from the model
            trg = trg[0, 1:].reshape(-1)
            output = output.reshape(-1, output.shape[2])
            prediction = prediction.reshape(-1)
            assert trg.shape[0] == prediction.shape[0] == output.shape[0]

            predictions += [prediction]
            targets += [trg]

            # calculate the loss and opt step for each predicted token
            losses = []
            for k in range(min(len(output), len(trg))):
                losses += [criterion(output[k], trg[k])]
            loss = sum(losses) / len(losses)
            loss.backward()
            opt.step()
            iter_loss += loss.item()
            total += src.shape[0]

        # measure time for the epoch
        iter_time_elapse = time.time() - iter_time_elapse

        # calculate bleu for the prediction
        iter_bleu += calc_bleu(predictions=predictions, targets=targets, int2target=int2target)

        # eval on development set
        save_dump_file = os.path.join(str(save_dump), 'weights.{}.pt'.format(str(i))) if save_dump is not None else save_dump
        bleu_dev, loss_dev, dev_time_elapse = evaluate(dev_loader, model, criterion, int2target=int2target, int2source=int2source, device=device, save_dump=save_dump_file)

        # save history
        iter_loss /= total
        train_loss += [iter_loss]
        train_bleu += [iter_bleu]
        dev_loss += [loss_dev]
        dev_bleu += [bleu_dev]

        process = '\t'.join(["epoch:{}",
                             "train_loss:{}",
                             "dev_loss:{}",
                             "train_bleu:{}",
                             "dev_bleu:{}",
                             "epoch_time:{}",
                             "dev_time:{}"]).format(
            i, iter_loss, loss_dev, iter_bleu, bleu_dev, iter_time_elapse, dev_time_elapse
        )
        process = '\t'.join([param_str, process])

        if log_file is not None:
            with open(log_file, 'a+') as f:
                f.write("{}\n".format(process))

        print(process)

        # save best model on dev bleu
        if best_bleu_dev < bleu_dev:
            checkpoint_dict = {'model': model.state_dict(), 'criterion': criterion.state_dict()}
            torch.save(checkpoint_dict, checkpoint)
            best_bleu_dev = bleu_dev
            print("saved best bleu on dev")

    return train_loss, train_bleu, dev_loss, dev_bleu



def main():

    parser = argparse.ArgumentParser(description=__doc__)

    # data arguments
    parser.add_argument('-a', '--SourceTrain', type=str, required=True, help="path to source train file")
    parser.add_argument('-b', '--TargetTrain', type=str, required=True, help="path to target train file")
    parser.add_argument('-c', '--SourceDev', type=str, required=True, help="path to source development file")
    parser.add_argument('-d', '--TargetDev', type=str, required=True, help="path to target development file")

    # model arguments
    parser.add_argument('-u', '--Decoder', type=str, required=True, help="either Basic or Attention")
    parser.add_argument('-t', '--AttentionType', default=None, help='either general, dot, scaled_dot, MLP or concat for attention, None for Basic')
    parser.add_argument('-e', '--EncoderBidirectional', type=int, default=0, help="either 1(True) or 0(False)")
    parser.add_argument('-f', '--PositionalEncoding', type=int, default=0, help="positional embedding, 1 to use, 0 not to use")
    parser.add_argument('-r', '--TypeRNN', default='GRU', help='Choose rnn type, LSTM, GRU, Vanilla')
    parser.add_argument('-n', '--EncoderLayers', type=int, default=1, help='Choose encoder number of layers, 1 or 2')
    parser.add_argument('-k', '--HiddenSize', type=int, default=100, help='Choose hidden size')
    parser.add_argument('-g', '--EmbeddingDimToken', type=int, default=100, help='Choose token embedding size')

    # process arguments
    parser.add_argument('-s', '--SaveWeights', default=None, help='path to weights folder (save weights for heatmap), only for Attention')
    parser.add_argument('-l', '--SaveProcess', default=None, help='path to log folder (save process)')
    parser.add_argument('-m', '--SaveModel', default='CheckpointFiles', help='path to save kwargs and model checkpoint')
    parser.add_argument('-i', '--Iterations', default=10, type=int, help='number of iterations for the model to run')

    args = parser.parse_args()

    assert args.EmbeddingDimToken > 0, "invalid number for embedding size {}".format(args.HiddenSize)
    assert args.HiddenSize > 0, "invalid number for hidden size {}".format(args.HiddenSize)
    assert args.Iterations > 0, "invalid number of iterations {}".format(args.Iterations)
    assert args.EncoderLayers in [1, 2], "invalid number of layers for decoder {}".format(args.EncoderLayers)
    assert args.EncoderBidirectional in [0, 1], "invalid Bidir request for encoder {}".format(args.EncoderBidirectional)
    assert args.PositionalEncoding in [0, 1], "invalid positional request for encoder {}".format(args.PositionalEncoding)
    assert args.Decoder in ['Basic', 'Attention'], "invalid Decoder {}".format(args.Decoder)
    assert args.AttentionType in [None, 'general', 'scaled_dot', 'dot', 'MLP', 'concat'], "invalid att {}".format(args.AttentionType)
    assert args.SaveWeights is None or args.AttentionType is not None, "Weights can only be saved in Attention Decoding"
    assert args.TypeRNN in ['Vanilla', 'GRU', 'LSTM'], "invalid rnn type {}".format(args.TypeRNN)
    assert args.SaveModel is not None, "Must insert a location for models with args.SaveModel but inserted None"

    # set hyper-parameters and other variables
    batch_size = 1
    learning_rate = 0.001
    max_iter = args.Iterations
    encoder_hidden_size = decoder_hidden_size = args.HiddenSize
    encoder_hidden_size //= (args.EncoderBidirectional+1)
    encoder_hidden_size //= args.EncoderLayers if args.Decoder == 'Basic' else 1
    embedding_dim = args.EmbeddingDimToken
    num_layers = args.EncoderLayers
    bidirectional = True if args.EncoderBidirectional == 1 else False
    att_type = args.AttentionType
    encoder_decoder_type = args.Decoder
    rnn_type = args.TypeRNN
    use_positional_embedding = True if args.PositionalEncoding == 1 else False

    param_str = '\t'.join(["positional:{}",
                           "encoder_decoder_type:{}",
                           "num_layers:{}",
                           "encoder_type:{}",
                           "att_type:{}",
                           "bidirectional:{}",
                           "embedding_dim:{}",
                           "encoder_hidden_size:{}",
                           "decoder_hidden_size:{}"]).format(
        str(use_positional_embedding), encoder_decoder_type, str(num_layers), rnn_type, att_type, str(bidirectional),
        str(embedding_dim), str(encoder_hidden_size), str(decoder_hidden_size)
    )

    sos_symbol = '<s>'
    eos_symbol = '</s>'
    unk_symbol = '<UNK>'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # if weights to be saved, make the requested directory
    if args.SaveWeights is not None and not os.path.exists(args.SaveWeights):
        os.mkdir(args.SaveWeights)

    # if logs to be save, make the requested directory
    if args.SaveProcess is not None and not os.path.exists(args.SaveProcess):
        os.mkdir(args.SaveProcess)

    # create a folder for the requested string (holds model checkpoint, args , etc)
    if not os.path.exists(args.SaveModel):
        os.mkdir(args.SaveModel)

    # load data
    load_args = [sos_symbol, eos_symbol, unk_symbol]
    train_loader, load_kwargs = load_dataset(args.SourceTrain, args.TargetTrain, batch_size, True, *load_args)
    dev_loader, _= load_dataset(args.SourceDev, args.TargetDev, batch_size, False, *load_args, **load_kwargs)

    # save the loading kwargs for the testing script
    load_kwargs_file = os.path.join(args.SaveModel, 'load_kwargs.pkl')
    with open(load_kwargs_file, 'wb+') as f:
        pickle.dump(load_kwargs, f)

    # init attention or basic encoder decoder based on input
    model_kwargs = {
        'encoder_hidden_size': encoder_hidden_size,
        'decoder_hidden_size': decoder_hidden_size,
        'num_layers': num_layers,
        'embedding_dim': embedding_dim,
        'src_vocab_size': len(load_kwargs['source2int']),
        'trg_vocab_size': len(load_kwargs['target2int']),
        'max_source_len': load_kwargs['max_source_len'],
        'eos_int': load_kwargs['target2int'][eos_symbol],
        'device': device,
        'bidirectional': bidirectional,
        'rnn_type': rnn_type,
        'encoder_decoder_type': encoder_decoder_type,
        'att_type': att_type,
        'use_positional_embedding': use_positional_embedding
    }
    # save the model kwargs for the testing script
    model_kwargs_file = os.path.join(args.SaveModel, 'model_kwargs.pkl')
    with open(model_kwargs_file, 'wb+') as f:
        pickle.dump(model_kwargs, f)

    # create log file
    log_file = None if args.SaveProcess is None else os.path.join(args.SaveProcess, 'log.train.{}.txt'.format(args.Decoder))

    # initialize model for training
    model = EncoderDecoderBase.createInstance(encoder_decoder_type=encoder_decoder_type).EncoderDecoderModel(**model_kwargs)
    # train model and evaluate on dev, save best model to file
    train(
        train_loader=train_loader,
        dev_loader=dev_loader,
        learning_rate=learning_rate,
        max_iter=max_iter,
        device=device,
        model=model,
        param_str=param_str,
        int2target=load_kwargs['int2target'],
        int2source=load_kwargs['int2source'],
        save_model_dir=args.SaveModel,
        log_file=log_file,
        save_dump=args.SaveWeights
    )

    print("finished training")


if __name__ == "__main__":
    main()