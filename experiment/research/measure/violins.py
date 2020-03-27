import argparse
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiment import io


def split_violins(
    train_csv,
    test_csv,
    output_filepath,
    #bins,
    title=None,
    #xlabel,
    measure_name='Normalized Euclidean Distance',
    conditional_models=None,
    #density=False,
    #color=None,
    dpi=400,
    measure_range=(0.0, 1.0),
    font_size=None,
    pad_inches=0.1,
    scale='area',
    orient='v',
    linewidth=None,
    overwrite=False,
):
    """The violin plots used for visualizing the measures in experiments 1
    and 2 using seaborn violinplot.

    Parameters
    ----------
    """
    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})


    # TODO load all of the csvs.
    if isinstance(train_csv, str) and isinstance(test_csv, str):
        train = pd.read_csv(train_csv, header=None).values.flatten()
        test = pd.read_csv(test_csv, header=None).values.flatten()

        # Create DataFrame for plotting
        measures = pd.DataFrame(
            np.concatenate([train, test]),
            columns=[measure_name],
        )
        measures['data_split'] = ['train'] * len(train) + ['test'] * len(train)
        measures['model'] = conditional_models

        # Plot violin
        sns.violinplot(
            'model',
            measure_name,
            hue='data_split',
            split=True,
            palette={'train': 'b', 'test':'darkgoldenrod'},
            data=measures,
            scale=scale,
            order=conditional_models,
            orient=orient,
        )
    elif (
        isinstance(train_csv, list)
        and isinstance(test_csv, list)
        and len(train_csv) == len(test_csv)
        and conditional_models is not None
        and len(train_csv) == len(conditional_models)
    ):
        measures = []
        model_ids = []
        data_splits = []
        for i in range(len(train_csv)):
            train = pd.read_csv(train_csv[i], header=None).values.flatten()
            measures += [
                train,
                pd.read_csv(test_csv[i], header=None).values.flatten(),
            ]

            model_ids += [conditional_models[i]] * len(train) * 2
            data_splits += ['train'] * len(train) + ['test'] * len(train)

        # Create DataFrame for plotting
        measures = pd.DataFrame(
            np.concatenate(measures),
            columns=[measure_name],
        )
        measures['model_ids'] = model_ids
        measures['data_split'] = data_splits

        # Plot violin
        if orient == 'v':
            ax = sns.violinplot(
                'model_ids',
                measure_name,
                hue='data_split',
                split=True,
                palette={'train': 'Default blue', 'test':'darkgoldenrod'},
                data=measures,
                scale=scale,
                order=conditional_models,
                orient=orient,
                linewidth=linewidth,
            )
        else:
            ax = sns.violinplot(
                measure_name,
                'model_ids',
                hue='data_split',
                split=True,
                palette={'train': 'Default blue', 'test':'darkgoldenrod'},
                data=measures,
                scale=scale,
                order=conditional_models,
                orient=orient,
                linewidth=linewidth,
            )

        if measure_range is not None:
            if orient == 'v':
                ax.set(ylim=measure_range)
            else:
                ax.set(xlim=measure_range)
    else:
        raise TypeError('Expected either pair of strs, or pair of lists.')

    if title is not None:
        #if '{bins}' in title:
        #    title = title.replace('{bins}', str(bins))
        #if '{bnn_draws}' in title:
        #    title = title.replace('{bnn_draws}', str(col_size))

        #title = title.replace('\n', '\n')
        plt.title(title)

    #if '{bins}' in xlabel:
    #    xlabel = xlabel.replace('{bins}', str(bins))
    #if '{bnn_draws}' in xlabel:
    #    xlabel = xlabel.replace('{bnn_draws}', str(col_size))
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)

    plt.savefig(
        io.create_filepath(output_filepath, overwrite=overwrite),
        dpi=dpi,
        pad_inches=pad_inches,
    )
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
        '--pad_inches',
        default=0.1,
        type=float,
        help='The padding in inches of the histogram.',
    )

    parser.add_argument(
        '--font_size',
        default=14,
        type=int,
        help='The font size of labels on the histogram.',
    )

    parser.add_argument(
        '-c',
        '--color',
        default=None,
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
        default='Normalized Euclidean Distance (bins={bins}, {bnn_draws}draws/pred)',
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
