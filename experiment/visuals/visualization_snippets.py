import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns


def corr_2d_heatmap(df, suptitle=None):
    corr = df.corr()

    f, ax = plt.subplots(figsize=(10,6))
    hm = sns.heatmap(
        round(corr,2),
        annot=True,
        ax=ax,
        cmap="coolwarm",
        fmt='.2f',
        linewidths=.05,
    )

    f.subplots_adjust(top=0.93)

    if isinstance(suptitle, str):
        t=f.suptitle(suptitle, fontsize=14)

    plt.show()


def many_jointplots(df, title=None, axes_lim=None, diag_kind='kde'):
    plot_kws = {'edgecolor': 'k', 'linewidth': 0.5}

    pp = sns.pairplot(
        df,
        size=1.8,
        aspect=1.8,
        plot_kws=plot_kws,
        diag_kind=diag_kind,
    )

    pp.fig.subplots_adjust(top=0.93, wspace=0.3)

    if axes_lim:
        for i, row in enumerate(pp.axes):
            for j, ax in enumerate(row):
                if i == j: continue
                ax.set_xlim(axes_lim[:2])
                ax.set_ylim(axes_lim[2:])

    if isinstance(title, str):
        t = pp.fig.suptitle(title, fontsize=14)

    plt.show()

def scatter3d(df, cols):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs = df[cols[0]]
    ys = df[cols[1]]
    zs = df[cols[2]]
    ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])

    plt.show()


def aligned_hists(df, bins=10, title=None, xaxis_label=None, yaxis_label=None):
    fig, axes = plt.subplots()

    df.hist(bins=bins, ax=axes, sharex=True)

    if isinstance(title, str):
        plt.suptitle(title)

    if isinstance(xaxis_label, str):
        fig.text(0.5, 0.04, xaxis_label, ha='center')
    if isinstance(yaxis_label, str):
        fig.text(0.04, 0.5, yaxis_label, va='center', rotation='vertical')

    plt.show()


def func_on_scatter(
    df1,
    df2,
    df1_label,
    df2_label,
    title=None,
    axes_lim=None,
):
    """Plots two sets of data, one as datapoints and the other as a function"""
    plot_kws = {'edgecolor': 'face', 'linewidth': 0.2, 'alpha': 0.6}

    #append them while adding another column indicating
    df1['plot_id'] = [df1_label] * len(df1)
    df2['plot_id'] = [df2_label] * len(df2)

    df = df1.append(df2, True)

    pp = sns.pairplot(
        df,
        hue='plot_id',
        size=1.8,
        aspect=1.8,
        palette={df1_label: "#6666FF", df2_label: "#FF6666"},
        markers=['o', 's'],
        #markers=['+', 'x'],
        plot_kws=plot_kws,
    )

    pp.fig.subplots_adjust(top=0.93, wspace=0.3)

    if axes_lim:
        for i, row in enumerate(pp.axes):
            for j, ax in enumerate(row):
                if i == j: continue
                ax.set_xlim(axes_lim[:2])
                ax.set_ylim(axes_lim[2:])

    if isinstance(title, str):
        t = pp.fig.suptitle(title, fontsize=14)

    plt.show()


def hist_kde(df):
    """Plot a kde on top of a histogram."""
    pass


def pair_plot_info(
    df,
    title=None,
    axes_lim=None,
    diag_kind='kde',
    diag_kws=None,
    lower_kws=None,
    upper_kws=None,
):
    """Creates a paired plot with hist/kde down the diagonal, scatter plot on
    the upper right triangle, and density plot on the lower right triangle.
    """
    sns.set_style("whitegrid")
    pg = sns.PairGrid(df)

    # Set diagonal hist or kde
    if diag_kind == 'kde':
        diag_args = {
            'shade': True,
            'legend': False,
        }
        if diag_kws:
            diag_args.update(diag_kws)

        pg.map_diag(sns.kdeplot, **diag_args)
    elif diag_kind == 'hist':
        diag_args = {
            'bins': 20,
        }
        if diag_kws:
            diag_args.update(diag_kws)

        pg.map_diag(plt.hist, **diag_args)
    else:
        # distance plot: a hist with a overlayed kde
        diag_args = {
            'bins': 20,
            'kde_kws': {'legend': False, 'color':'k', 'alpha': 0.5},
            'hist_kws': {'alpha': 1.0},
            #'norm_hist': True, # Needs its own scale for this to be useful
        }
        if diag_kws:
            diag_args.update(diag_kws)

        pg.map_diag(sns.distplot, **diag_args)

    # Set lower left triangle to density
    lower_args = {
        #'shade': True,
        #'shade_lowest': False,
        'legend': False,
        #cbar=True, # need to provide a c_bar alignment (ie. 1 cbar for all)
    }
    if lower_kws:
        lower_args.update(lower_kws)

    pg.map_lower(sns.kdeplot, **lower_args)

    # Set upper right triangle to scatter plot
    upper_args = {
        'edgecolor': 'face',
        'linewidth': 0.2,
        'alpha': 0.5,
    }
    if upper_kws:
        upper_args.update(upper_kws)

    pg.map_upper(plt.scatter, **upper_args)

    if axes_lim:
        for i, row in enumerate(pg.axes):
            for j, ax in enumerate(row):
                if i == j:
                    # Turn off grid for diagonal, because it is only meant to
                    # indicate shape, nothing quantifiable or about magnitudes;
                    # Needs a separate plot for that (histogram).
                    ax.grid(False)
                    continue
                ax.set_xlim(axes_lim[:2])
                ax.set_ylim(axes_lim[2:])

    if isinstance(title, str):
        t = pg.fig.suptitle(title, fontsize=14)

    plt.show()
