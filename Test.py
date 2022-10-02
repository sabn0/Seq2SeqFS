

# import packages
import torch
import torch.utils.data as data
import argparse
import os
import pickle
from Train import test
from src.LoadDataBPE import LoadDataBPE
from src.LoadDataSimple import LoadDataSimple
from src.LoadDataBase import PytorchCustomLoader
from src.TransformerModelManual import ManualTransformer
from src.TransformerModelAuto import AutomaticTransformer

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--SourcePath', required=True, type=str, help='path to source file')
    parser.add_argument('-t', '--TargetPath', required=True, type=str, help='path to target file')
    parser.add_argument('-d', '--Debug', default=0, type=int)
    parser.add_argument('-a', '--AutomaticModel', default=1, type=int)
    parser.add_argument('-b', '--BPE', default=1, type=int)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # test model
    # load saved model
    checkpoint_path = os.path.join(os.getcwd().rsplit('/', 1)[0], 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_path, 'checkpoint')
    model_kwargs_file = os.path.join(checkpoint_path, 'model_kwargs')
    env_kwargs_file = os.path.join(checkpoint_path, 'env_kwargs')
    assert os.path.exists(checkpoint_file) and os.path.exists(model_kwargs_file), "run Train.py before Test.py"
    with open(model_kwargs_file, 'rb') as f:
        model_kwargs = pickle.load(f)
    with open(env_kwargs_file, 'rb') as f:
        env_kwargs = pickle.load(f)

    if args.AutomaticModel:
        model = AutomaticTransformer(**model_kwargs)
    else:
        model = ManualTransformer(**model_kwargs)
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model'])

    symbol_kwargs = {
        'SOS': env_kwargs['SOS'],
        'EOS': env_kwargs['EOS'],
        'UNK': env_kwargs['UNK'],
        'PAD': env_kwargs['PAD'],
        'EOW': env_kwargs['EOW'],
        'EOC': env_kwargs['EOC']
    }

    # choose simple or BPE tokenizer
    if args.BPE:
        source_loader = LoadDataBPE(file_path=args.SourcePath, debug=args.Debug)
        target_loader = LoadDataBPE(file_path=args.TargetPath, debug=args.Debug)
    else:
        source_loader = LoadDataSimple(file_path=args.SourcePath, debug=args.Debug)
        target_loader = LoadDataSimple(file_path=args.TargetPath, debug=args.Debug)

    # load source sentences
    sources = source_loader.readFile()
    sources = source_loader.tokenize(w2i=env_kwargs['s2i'], sentences=sources, eow=symbol_kwargs['EOW']).getTokenizedSentences()

    # load target sentences
    targets = target_loader.readFile()
    targets = target_loader.tokenize(w2i=env_kwargs['t2i'], sentences=targets, eow=symbol_kwargs['EOW']).getTokenizedSentences()

    test_loader = PytorchCustomLoader(
        sources=sources, targets=targets, sources2int=env_kwargs['s2i'], targets2int=env_kwargs['t2i'], **symbol_kwargs
    )
    test_loader = data.DataLoader(dataset=test_loader, batch_size=env_kwargs['batch_size'], shuffle=False)

    test_bleu = test(test_loader, model=model, device=device, int2target=env_kwargs['i2t'])
    print("bleu on test set: {}".format(test_bleu))




if __name__ == "__main__":
    main()