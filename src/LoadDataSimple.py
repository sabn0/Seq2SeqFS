
# import packages
from src.LoadDataBase import LoadData
from collections import Counter


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
