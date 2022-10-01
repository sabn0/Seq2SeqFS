
# import packages
from abc import abstractmethod
from collections import defaultdict, Counter
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

class LoadDataBPE(LoadData):
    def __init__(self, file_path: str, debug=False):
        super(LoadDataBPE, self).__init__(file_path=file_path, debug=debug)

    def createVocab(self, max_vocab: int, **kwargs) -> tuple:

        # implementation based on the BPE algorithm in "Neural Machine Translation of Rare Words with Subword Units"
        # https://arxiv.org/abs/1508.07909

        eow = kwargs['EOW']
        eoc = kwargs['EOC']

        # get word counts in the sentences
        sentences = self.load()
        text = ' '.join(sentences)
        words = [eoc.join(list(w))+eoc+eow for w in text.split()]
        word2count = dict(Counter(words).most_common())

        def countTokens(word2count: dict) -> defaultdict:
            char2count = defaultdict(lambda: 0)
            for word, count in word2count.items():
                chars = word.split(eoc)
                for char, next_char in zip(chars[:-1], chars[1:]):
                    char2count[(char, next_char)] += word2count[word]
            return char2count

        def replaceBest(word2count: dict, best_pair: tuple) -> dict:
            new_word2count = dict()
            for word, count in word2count.items():

                # find occurrence of best pair in the word
                wrapped_word = eoc + word + eoc
                pair_str_mid = eoc.join(best_pair)
                pair_str = eoc + pair_str_mid + eoc

                if pair_str in wrapped_word:
                    clean_pair = eoc + ''.join(best_pair) + eoc
                    new_word = wrapped_word.replace(pair_str, clean_pair).strip(eoc)
                    new_word2count[new_word] = count
                else:
                    new_word2count[word] = count
            return new_word2count

        # merge symbols *max_vocab* times
        for _ in range(max_vocab):

            # count tokens
            token2count = countTokens(word2count)

            # replace best pair
            best_pair = max(token2count, key=token2count.get)
            word2count = replaceBest(word2count, best_pair=best_pair)

        # make vocab
        vocab = [w.split(eoc) for w in list(word2count.keys())]
        vocab = list(set([item for sublist in vocab for item in sublist]))
        w2i, i2w = self.enumerateVocabDicts(words=vocab, **kwargs)
        return w2i, i2w

    def tokenize(self, w2i: dict, sentences: list, eow: str):
        tokenized_sentences = []
        for sentence in sentences:
            tokenized_words = []
            for word in sentence.split():
                word = list(word) + [eow]
                i, j = 0, len(word)
                tokenized = []
                while i < len(word):
                    sub_word = ''.join(word[i:j])
                    if sub_word in w2i:
                        tokenized += [sub_word]
                        i += len(sub_word)
                        j = len(word)
                    else:
                        j -= 1
                tokenized_words += [tokenized]
            tokenized_sentences += [tokenized_words]

        self.tokenized_sentences = tokenized_sentences
        return self

    def getMaxSentenceLength(self) -> int:
        sentences = self.getTokenizedSentences()
        max_len = max([sum([len(w) for w in s]) for s in sentences])
        assert max_len > 0
        return max_len + 2


class LoadDataSimple(LoadData):
    def __init__(self, file_path: str, debug=False):
        super(LoadDataSimple, self).__init__(file_path=file_path, debug=debug)

    def createVocab(self, max_vocab: int, **kwargs) -> tuple:
        words = ' '.join(self.load()).split()
        words = list(dict(Counter(words).most_common(max_vocab)).keys())
        return self.enumerateVocabDicts(words=words, **kwargs)

    def tokenize(self, w2i: dict, sentences: list, eow: str):
        self.tokenized_sentences = sentences
        return self

    def getMaxSentenceLength(self) -> int:
        sentences = self.getTokenizedSentences()
        max_len = max([len(s.split()) for s in sentences])
        assert max_len > 0
        return max_len + 2


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