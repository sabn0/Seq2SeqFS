
# import packages
import matplotlib.pyplot as plt
import pandas as pd


def garnish(ax: plt.Axes, **kwargs):

    plt.rcParams["font.weight"] = "bold"
    ax.tick_params(axis="x", labelsize=kwargs['font_size'])
    ax.tick_params(axis="y", labelsize=kwargs['font_size'])
    ax.set_xlabel(kwargs['x_label'], fontsize=kwargs['font_size'], fontweight='bold')
    ax.set_ylabel(kwargs['y_label'], fontsize=kwargs['font_size'], fontweight='bold')
    try:
        ax.set_yticks(kwargs['set_yticks'])
        ax.set_xticks(kwargs['set_xticks'])
        ax.set_title(kwargs['title'], fontsize=kwargs['font_size'], fontweight='bold')
    except:
        pass

    _ = [label.set_fontweight('bold') for label in ax.get_xticklabels()]
    _ = [label.set_fontweight('bold') for label in ax.get_yticklabels()]

    if 'legend_loc' in kwargs:
        ax.legend(prop={'size': kwargs['font_size'] - 5, 'weight': 'bold'}, loc=kwargs['legend_loc'])


# read log file
def readFile(file_name) -> pd.DataFrame:
    with open(file_name, 'r') as f:
        content = f.read().splitlines()
    rows = []
    for line in content:
        fields = line.split('\t')
        values = [field.split(':')[1] for field in fields]
        rows += [values]

    columns = ['positional', 'encoder_decoder_type', 'num_layers', 'encoder_type', 'att_type', 'bidirectional',
               'embedding_dim', 'encoder_hidden_size', 'decoder_hidden_size', 'epoch', 'train_loss', 'dev_loss',
               'train_bleu', 'dev_bleu', 'epoch_time', 'dev_time']
    df = pd.DataFrame(rows, columns=columns, dtype=float)
    df = df.round(3)
    df.num_layers = df.num_layers.astype(int)
    df.decoder_hidden_size = df.decoder_hidden_size.astype(int)
    df.encoder_hidden_size = df.encoder_hidden_size.astype(int)
    df.embedding_dim = df.embedding_dim.astype(int)
    df['epoch'] += 1

    return df