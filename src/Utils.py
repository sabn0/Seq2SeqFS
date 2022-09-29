
# import packages
from abc import abstractmethod
from collections import Counter
import torch
import torch.utils.data as data


class LoadData:
    def __init__(self, file_path: str, debug=False):
        self.file_path = file_path
        self.debug = debug

    def readFile(self) -> list:
        with open(self.file_path, 'r') as f:
            content = f.read().splitlines()

        if self.debug:
            N = int(len(content) * 0.01)
            content = content[:N]
        return content

    @abstractmethod
    def getMaxSentenceLength(self) -> int:
        pass

    @abstractmethod
    def load(self) -> list:
        pass

    @abstractmethod
    def createVocab(self, max_vocab: int, **kwargs) -> tuple:
        pass


class LoadDataSimple(LoadData):
    def __init__(self, file_path: str, debug=False):
        super(LoadDataSimple, self).__init__(file_path=file_path, debug=debug)

    def load(self) -> list:
        lines = [l.strip() for l in self.readFile()]
        return lines

    def getMaxSentenceLength(self) -> int:
        sentences = self.load()
        max_len = max([len(s.split()) for s in sentences])
        assert max_len > 0
        return max_len + 2

    def createVocab(self, max_vocab: int, **kwargs) -> tuple:
        words = ' '.join(self.load()).split()
        words = list(dict(Counter(words).most_common(max_vocab)).keys())
        w2i = {w: i for i, w in enumerate(words)}
        i2w = {i: w for w, i in w2i.items()}

        for i, (k, v) in enumerate(kwargs.items()):
            w2i[v] = max_vocab+i
            i2w[(max_vocab+i)] = v

        return w2i, i2w


# pytorch loader
class PytorchCustomLoader(data.Dataset):
    def __init__(self,
                 sources: list,
                 targets: list,
                 sources2int: dict,
                 targets2int: dict,
                 **kwargs
                 ):
        self.sources = sources
        self.targets = targets
        self.sources2int = sources2int
        self.targets2int = targets2int
        self.sos_symbol = kwargs['SOS']
        self.eos_symbol = kwargs['EOS']
        self.unk_symbol = kwargs['UNK']

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