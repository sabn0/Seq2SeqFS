
# import packages
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import torch
from Plotters.PlotterGarnish import garnish


def plot_heatmap_sns(df: pd.DataFrame, out_path: str, x_ax_name: str, y_ax_name: str):

    df = df.T
    ax = sns.heatmap(data=df, cmap="Reds")

    garnish_kwargs = {
        'x_label': x_ax_name,
        'y_label': y_ax_name,
        'font_size':15
    }

    garnish(ax=ax, **garnish_kwargs)

    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    ax.get_figure().savefig(out_path, bbox_inches='tight')
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-w', '--WeightsDir', type=str, required=True, help="path to weights files")
    parser.add_argument('-o', '--FiguresDir', type=str, required=True, help="path to figures")
    parser.add_argument('-l', '--Labels', type=str, default='Plotters/Heatmap.labels.txt', help='path to file with index and column labels')
    args = parser.parse_args()

    # create Figures dir if not exists
    if not os.path.exists(args.FiguresDir):
        os.mkdir(args.FiguresDir)

    # get labels for heatmap
    with open(args.Labels, 'r') as f:
        content = f.read().splitlines()
    i=0
    source_labels, target_labels = content[i+0].split(), content[i+1].split()
    source_labels = {i:l for i, l in enumerate(source_labels)}
    target_labels = {i: l for i, l in enumerate(target_labels)}

    # create heatmaps
    weights_files = os.listdir(args.WeightsDir)
    for i, weights_file in enumerate(weights_files):

        weight_file_path = os.path.join(args.WeightsDir, weights_file)
        weights = torch.load(weight_file_path)

        # weights is of shape (src_length, 1, trg_length), remove middle
        weights = weights.squeeze(1).detach().numpy().astype("float")
        df = pd.DataFrame(weights).rename(columns=target_labels).rename(index=source_labels)

        out_path = os.path.join(args.FiguresDir, 'weights{}'.format(str(i)))
        plot_heatmap_sns(df=df, out_path=out_path, x_ax_name='src', y_ax_name='trg')



if __name__ == "__main__":

    main()