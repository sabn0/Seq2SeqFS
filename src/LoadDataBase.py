
# import packages
from abc import abstractmethod
import torch
import torch.utils.data as data


class LoadData:
    def __init__(self, file_path: str, debug=False):
        self.file_path = file_path
        self.debug = debug
        self.tokenized_sentences = None

    def readFile(self) -> list:
        with open(self.file_path, 'r') as f:
            content = f.read().splitlines()

        if self.debug:
            N = int(len(content) * 0.01)
            content = content[:N]
        return content

    def load(self) -> list:
        lines = [l.strip() for l in self.readFile()]
        return lines

    def enumerateVocabDicts(self, words: list, **kwargs) -> tuple:

        w2i = {w: i for i, w in enumerate(words)}
        i2w = {i: w for w, i in w2i.items()}
        N = len(w2i)

        for i, (k, v) in enumerate(kwargs.items()):
            w2i[v] = N+i
            i2w[(N+i)] = v

        return w2i, i2w

    def getTokenizedSentences(self) -> list:
        assert self.tokenized_sentences is not None
        return self.tokenized_sentences

    @abstractmethod
    def tokenize(self, w2i: dict, sentences: list, eow: str):
        pass

    @abstractmethod
    def createVocab(self, max_vocab: int, **kwargs) -> tuple:
        pass

    @abstractmethod
    def getMaxSentenceLength(self) -> int:
        pass


# pytorch loader
class PytorchCustomLoader(data.Dataset):
    def __init__(self,
                 sources: list,
                 targets: list,
                 sources2int: dict,
                 targets2int: dict,
                 bpe_tokenizer: bool,
                 **kwargs
                 ):
        self.sources = sources
        self.targets = targets
        self.sources2int = sources2int
        self.targets2int = targets2int
        self.sos_symbol = kwargs['SOS']
        self.eos_symbol = kwargs['EOS']
        self.unk_symbol = kwargs['UNK']
        self.bpe_tokensize = bpe_tokenizer

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):

        # for BPE
        if self.bpe_tokensize:
            sources = [[self.sources2int[c] for c in w] for w in self.sources[item]]
            sources = [item for sublist in sources for item in sublist]
            targets = [[self.targets2int[c] for c in w] for w in self.targets[item]]
            targets = [item for sublist in targets for item in sublist]
        else:
            # convert source item and target item to index witn UNK
            sources = [self.sources2int[w] if w in self.sources2int else self.sources2int[self.unk_symbol] for w in self.sources[item].split()]
            targets = [self.targets2int[w] if w in self.targets2int else self.targets2int[self.unk_symbol] for w in self.targets[item].split()]

        # add start of sequence and end of sequence to sequences
        sources = [self.sources2int[self.sos_symbol]] + sources + [self.sources2int[self.eos_symbol]]
        targets = [self.targets2int[self.sos_symbol]] + targets + [self.targets2int[self.eos_symbol]]

        return torch.tensor(sources), torch.tensor(targets)