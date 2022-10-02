
# import packages
from collections import defaultdict, Counter
from src.LoadDataBase import LoadData

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

