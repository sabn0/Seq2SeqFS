# Seq2Seq with a transformer
This program runs a simple transformer for machine translation.\
Data can be tokenized using space delimiter and UNK words, or with the [BPE algorithm](https://arxiv.org/abs/1508.07909).\
Written as a self-learning experience with transformers and the BPE method, not tested on real data.

## Data
The program translates from a source language to a target language, supplied in two plain text files with the same number of rows.\
Each row in the source file should match to the target at the same row. Example data is given in the data folder.

## How to run
First train the model, then test it:
```
python Train.py -s=PATH_TO_TRAIN_SRC_DATA -t=PATH_TO_TRAIN_TRG_DATA
python Test.py -s=PATH_TO_TEST_SRC_DATA -t=PATH_TO_TEST_TRG_DATA
```

both Train.py and Test.py take source and target files as args. In addition, they take the following optional arguments:
```
-d  : 0 for full data, 1 for 1% of the data (for debug). default to 0.
-a  : 0 for manual implementation, 1 for the PyTorch library implementation. default to 1. 
-b  : 0 for simple tokenization, 1 for BPE. default to 1.
```

The program evaluates based on BLEU score.

## References
The BPE algorithm was self-implemented following instructions in the paper: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909).
The manual transformer was implemented following instructions in the paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), and partially assisted by an implementation from [here](https://github.com/aladdinpersson/Machine-Learning-Collection).
