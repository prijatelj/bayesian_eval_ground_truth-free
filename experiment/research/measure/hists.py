import argparse
import os

from matplotlib import pyplot as plt
import pandas as pd

from experiment import io


def hist_plots(
    measures_csv,
    output_filepath,
    bins,
    title,
    xlabel,
    ylabel,
    density=False,
    color='b',
    dpi=400,
):
    """The Histogram plots used for visualizing the measures in experiments 1
    and 2.
    """
    measures = pd.read_csv(measures_csv, header=None)
    col_size = len(measures.columns)
    measures = pd.DataFrame(measures.values.flatten())

    measures.hist(bins=bins, color=color)

    if '{bins}' in title:
        title.replace('{bins}', str(bins))
    if '{bnn_draws}' in title:
        title.replace('{bnn_draws}', str(col_size))

    if '{bins}' in xlabel:
        xlabel.replace('{bins}', str(bins))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(io.create_filepath(output_filepath), dpi=dpi)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        'measures_csv',
        help='CSV containing measurements.',
    )

    parser.add_argument(
        'output_filepath',
        default=None,
        help='The output file path of converted json.',
    )

    parser.add_argument(
        '-b',
        '--bins',
        default=1000,
        type=int,
        help='The number of bins to use for the histogram.',
    )

    parser.add_argument(
        '--dpi',
        default=400,
        type=int,
        help='The dpi of the histogram.',
    )

    parser.add_argument(
        '-c',
        '--color',
        default='b',
        help='The color of the histogram.',
    )

    parser.add_argument(
        '-t',
        '--title',
        default=None,
        help='The title of the histogram.',
    )

    parser.add_argument(
        '-x',
        '--xlabel',
        default='Normalized Euclidean Distance (bins={bins})',
        help='The label along the x axis.',
    )

    parser.add_argument(
        '-y',
        '--ylabel',
        default='Occurrence Count',
        help='The label along the y axis.',
    )

    parser.add_argument(
        '--density',
        action='store_true',
        help='If given, the hidden layers DO NOT use biases (set to zeros)',
    )

    args = parser.parse_args()

    if args.density and args.ylabel == 'Occurrence Count':
        args.ylabel = 'Occurrence Density'

    return args

if __name__ == '__main__':
    hist_plots(**vars(parse_args()))
