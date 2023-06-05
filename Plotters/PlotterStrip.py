
# import packages
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import torch
from Plotters.PlotterGarnish import garnish, readFile

def plot_strip_sns(df: pd.DataFrame,
                  x_ax_name: str,
                   x_order:list,
                  y_ax_name: str,
                  hue_ax_name:str,
                  out_path: str
                  ):

    df = df.sort_values(by=[y_ax_name]).reset_index()

    sns.set_style('whitegrid')
    ax = sns.stripplot(
        data=df,
        order=x_order,
        x=x_ax_name,
        y=y_ax_name,
        hue=hue_ax_name,
        size=8,
        jitter=0.05
    )

    df['encoder_complexity'] = 'Complex'
    df.loc[(df['bidirectional'] == 'False') & (df['num_layers'] == '1'), 'encoder_complexity'] = 'Simple'
    df['annot'] = df['encoder_complexity'] + '-' + df['att_type']
    counter = 0
    for i, txt in enumerate(df['annot']):
        if df[x_ax_name][i] == 'train_time' and 'Attention' == df[hue_ax_name][i]:
            sign = 1 if counter % 2 ==1 else 0
            counter+= 1
            ax.annotate(txt, (-0.5+sign*0.6, df[y_ax_name][i]),fontweight='bold')

    garnish_kwargs = {
        'x_label': x_ax_name,
        'y_label': y_ax_name,
        'font_size':15,
        'legend_loc': 'best'
    }

    garnish(ax=ax, **garnish_kwargs)
    ax.get_figure().savefig(out_path, bbox_inches='tight')
    plt.clf()


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--LogFile1', type=str, required=True, help="path to weights files")
    parser.add_argument('-f', '--LogFile2', type=str, required=True, help="path to weights files")
    parser.add_argument('-o', '--FiguresDir', type=str, required=True, help="path to figures")
    parser.add_argument('-p', '--Part', type=int, required=True, help="part of the exercise, 1 or 2")
    args = parser.parse_args()

    headers = ['encoder_decoder_type', 'num_layers', 'encoder_type', 'att_type', 'bidirectional']
    df1 = readFile(args.LogFile1).groupby(headers).mean().reset_index()
    df2 = readFile(args.LogFile2).groupby(headers).mean().reset_index()
    df = pd.concat([df1,df2])
    df.num_layers = df.num_layers.astype(str)
    df = df.rename(columns={"epoch_time": "train_time", "dev_time": "inference_time"})
    df = df.melt(id_vars=headers, var_name='set', value_vars=['train_time','inference_time'], value_name='Avg epoch time (sec)')
    order = ['train_time', 'inference_time']

    out_path = os.path.join(args.FiguresDir, '-'.join(['Part{}'.format(args.Part), 'time']))
    plot_strip_sns(
        df=df,
        x_ax_name='set',
        x_order=order,
        y_ax_name='Avg epoch time (sec)',
        hue_ax_name='encoder_decoder_type',
        out_path=out_path,
    )


if __name__ == "__main__":
    main()