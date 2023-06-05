# Translation with attention and RNNs

## Software
python 3.8, pytorch 1.11.
matplotlib, pandas, seaborn and sacrebleu installed.

## Data
The software writes translations from one sequence to another.\
The data need to be organized in 6 files within a data directory - train.src, train.trg, dev.src, dev.trg, test.src, test.trg.\
Each line in a src file should match to the corresponding line in the trg file. Tokens separated by space.

## How to train
Use the file TrainNMT.py. It must be supplied with 5 arguments:
```
--SourceTrain, -a : the path to train.src
--TargetTrain, -b : the path to train.trg
--SourceDev, -c : the path to dev.src
--TargetDev, -d : the path to dev.trg
--Decoder, -u : switch between 'Basic' (Part1) and 'Attention' (Part2)
```

The basic training uses RNNs for encoding and decoding, run by using this command:

```
python TrainNMT.py -a=PATH_TO_TRAIN_SRC -b=PATH_TO_TRAIN_TRG -c=PATH_TO_DEV_SRC -d=PATH_TO_DEV_TRG -u=Basic
```

This command uses the default arguments, trains a unidirectional GRU with one layer for 10 epochs, 
evaluates on the dev set after each epoch for BLEU and loss (also prints it), and saves the best checkpoint on the dev set according to BLEU. 
If you want the BLEU and loss results to be saved to a file (and not only printed), you can add -l=Logs to the command.


In addition, the software supports some more (optional) arguments:
```
--EncoderBidirectional, -e : 0 (default) or 1 for bidirectional
--PositionalEncoding, -f : 0 (default) or 1 for adding positional encodings
--TypeRNN, -r : GRU(default), LSTM or Vanilla
--EncoderLayers, -n : 1 (default) or 2
--HiddenSize, -k : 100 (default) or any other int > 0
--EmbeddingDimToken, -g : 100 (default) or any other int > 0
--Iterations, -i : 10 (default), or any other int > 0
--AttentionType, -t : switch between 'general', 'scaled_dot', 'dot', 'MLP', 'concat'
--SaveWeights, -s : directory name to save weights (the program will create it).
```

For example, you can switch to and attention-based decoder (dot product attention), train for 20 epochs, with a biLSTM encoder with the following command:

```
python TrainNMT.py -a=PATH_TO_TRAIN_SRC -b=PATH_TO_TRAIN_TRG -c=PATH_TO_DEV_SRC -d=PATH_TO_DEV_TRG -u=Attention -t=dot -i=20 -r=LSTM -e=1
```

By adding the -s=Weights flag, the program will save the attention weights for a specific dev example.


## How to evaluate

After running the TrainNMT.py program (whatever that was ran from the above), use the file EvaluateNMT.py. 
It must be supplied with 2 arguments:
```
--SourceTest, -e : the path to test.src
--TargetTest, -f : the path to test.trg
```

So you can run the evaluating part by running the following command from the command line:

	python EvaluateNMT.py -e=PATH_TO_TEST_SRC -f=PATH_TO_TEST_TRG

This command will load the saved model and print the BLEU and loss scores. 
If you want the BLEU and loss results to be saved to a file (and not only printed), you can add -l=Logs to the command.


## How to paint

You can plot the attention weights for a specific example after training if you used the -s flag. Simply run:

```
python -c 'from Plotters.PlotterHeatmap import main; main()' -w=Weights -o=Figures
```

