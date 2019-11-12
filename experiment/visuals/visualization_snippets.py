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


def many_jointplots(df, suptitle=None, axes_lim=None, diag_kind='kde'):
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

    if isinstance(suptitle, str):
        t = pp.fig.suptitle(suptitle, fontsize=14)

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
