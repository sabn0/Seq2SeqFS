
import torch
import torch.utils.data as data


# pytorch custom dataset
# only supports batch size of 1 (no padding / truncating), with variable sequences length
# source language vocab -> numbers
# target language vocab -> characters
class CustomWordLoader(data.Dataset):
    def __init__(self,
                 sources: list,
                 targets: list,
                 sources2int: dict,
                 targets2int: dict,
                 sos_symbol: str,
                 eos_symbol: str,
                 unk_symbol: str):
        self.sources = sources
        self.targets = targets
        self.sources2int = sources2int
        self.targets2int = targets2int
        self.sos_symbol = sos_symbol
        self.eos_symbol = eos_symbol
        self.unk_symbol = unk_symbol

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):

        # convert source item and target item to index
        sources = [self.sources2int[c] if c in self.sources2int else self.sources2int[self.unk_symbol] for c in self.sources[item].split()]
        targets = [self.targets2int[c] if c in self.targets2int else self.targets2int[self.unk_symbol] for c in self.targets[item].split()]

        # add start of sequence and end of sequence to sequences
        sources = [self.sources2int[self.sos_symbol]] + sources + [self.sources2int[self.eos_symbol]]
        targets = [self.targets2int[self.sos_symbol]] + targets + [self.targets2int[self.eos_symbol]]

        return torch.tensor(sources), torch.tensor(targets)


def load_file(file_name: str) -> list:
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()
    lines = [l.strip() for l in lines]
    return lines


def create_vocabulary(items: list, *args) -> tuple:

    # given a dataset (list of items in source or target) creates vocabulary for the dataset
    # args holds special symbols: [sos_symbol, eos_symbol, unk_symbol]
    # returns bidrectional mappings from symbol to int

    # get all unique symbols and add given symbols
    symbols = list(set(' '.join(items).split()))
    assert [True] == list(set([s not in symbols for s in args]))
    symbols += args

    # create mappings for symbol to int and int to symbol
    symbol2int = {s:i for i, s in enumerate(symbols)}
    int2symbol = {i:s for s, i in symbol2int.items()}

    return symbol2int, int2symbol


def load_dataset(source_file: str, target_file: str, batch_size: int, shuffle: bool, *args, **kwargs) -> tuple:

    # given source and target paths to file, and special symbols, create pytorch data loader
    # kwargs holds mappings done for training set, if passed - skip vocab creation

    # load translation items and get dictionaries
    sources = load_file(source_file)
    targets = load_file(target_file)

    # this is for positional embeddings
    max_source_len = max([len(s.split()) for s in sources]) + 10

    if not kwargs:
        source2int, int2source = create_vocabulary(sources, *args)
        target2int, int2target = create_vocabulary(targets, *args)
        kwargs = {'source2int': source2int,
                  'int2source': int2source,
                  'target2int': target2int,
                  'int2target':int2target,
                  'max_source_len': max_source_len}

    # create a data loader for the data, batch_size is 1 in the exercise
    ds = CustomWordLoader(sources, targets, kwargs['source2int'], kwargs['target2int'], *args)
    loader = data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle)
    return loader, kwargs