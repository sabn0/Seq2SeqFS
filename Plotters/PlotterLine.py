
# import packages
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import torch
from Plotters.PlotterGarnish import garnish, readFile


def plot_line_sns(df: pd.DataFrame,
                  x_ax_name: str,
                  y_ax_name: str,
                  hue_ax_name:str,
                  style_ax_name: str,
                  size_ax_name: str,
                  sizes:tuple,
                  out_path: str,
                  title: str,
                  y_ticks: list,
                  legend_loc='best',
                  palette=None
                  ):

    sns.set_style('whitegrid')
    ax = sns.lineplot(
        data=df,
        x=x_ax_name,
        y=y_ax_name,
        hue=hue_ax_name,
        palette=palette,
        style=style_ax_name,
        size=size_ax_name,
        sizes=sizes
    )

    max_x_ax = int(max(df[x_ax_name].tolist()))
    garnish_kwargs = {
        'x_label': x_ax_name,
        'y_label': y_ax_name,
        'font_size':15,
        'title': title,
        'legend_loc': legend_loc,
        'set_xticks': list(range(2, max_x_ax+1, 2)),
        'set_yticks': y_ticks
    }

    garnish(ax=ax, **garnish_kwargs)
    ax.get_figure().savefig(out_path, bbox_inches='tight')
    plt.clf()



def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f', '--LogFile', type=str, required=True, help="path to weights files")
    parser.add_argument('-o', '--FiguresDir', type=str, required=True, help="path to figures")
    parser.add_argument('-p', '--Part', type=int, required=True, help="part of the exercise, 1 or 2 or 11")
    args = parser.parse_args()

    df = readFile(args.LogFile)
    # levels
    df['encoder'] = 'Complex'
    df.loc[(df['bidirectional'] == 'False') & (df['num_layers'] == 1), 'encoder'] = 'Simple'

    loss_y_ax = [0.5, 1, 1.5, 2, 2.5]
    palette = None
    if args.Part == 1:
        bleu_y_ax = [10,20,30,40,50]
        hue_ax_name = 'encoder_type'
        size_ax_name = 'num_layers'
        sizes = (1,2)
        style_ax_name = 'bidirectional'
    elif args.Part == 2:
        bleu_y_ax = [10,20,30,40,50,60,70,80, 90,100]
        hue_ax_name = 'att_type'
        palette = {'None': '#000000', 'general': '#0066CC', 'scaled_dot': '#CC0000', 'dot': '#FF8000', 'concat': '#7F00FF', 'MLP':'#33FFFF'}
        size_ax_name = 'encoder'
        sizes = (2,1)
        style_ax_name = None
    elif args.Part == 11:
        df['addition'] = 'base'
        df.loc[(df['positional'] == 'True'), 'addition'] = 'pos_encoding'
        df.loc[(df['positional'] == 'False') & (df['embedding_dim'] == 200), 'addition'] = 'big_embedding'
        df.loc[(df['positional'] == 'False') & (df['decoder_hidden_size'] == 200), 'addition'] = 'big_hidden'
        bleu_y_ax = [10,20,30,40,50]
        hue_ax_name = 'addition'
        palette = {'base': '#000000', 'pos_encoding': '#FF0000', 'big_embedding': '#33FFFF', 'big_hidden': '#9933FF'}
        size_ax_name = 'encoder'
        sizes = (2,1)
        style_ax_name = None
    else:
        raise ValueError("unkown part supplied {}".format(args.Part))

    params = {'loss':'Loss', 'bleu':'BLEU'}
    for param in params:
        legend_loc = 'upper left' if param == 'bleu' else 'best'
        y_ticks = bleu_y_ax if param == 'bleu' else loss_y_ax
        out_path = os.path.join(args.FiguresDir, '-'.join(['Part{}'.format(args.Part), param]))
        plot_line_sns(
            df=df,
            x_ax_name='epoch',
            y_ax_name='dev_{}'.format(param),
            hue_ax_name=hue_ax_name,
            size_ax_name=size_ax_name,
            style_ax_name=style_ax_name,
            sizes=sizes,
            palette=palette,
            out_path=out_path,
            title=params[param],
            y_ticks=y_ticks,
            legend_loc=legend_loc
        )


if __name__ == "__main__":
    main()