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


def many_jointplots(df, title=None, axes_lim=None, diag_kind='kde', color=None):
    plot_kws = {'edgecolor': 'k', 'linewidth': 0.5}

    pp = sns.pairplot(
        df,
        c=color,
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


def scatter3d(df, cols, **scatter_kws):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs = df[cols[0]]
    ys = df[cols[1]]
    zs = df[cols[2]]
    ax.scatter(xs, ys, zs, s=50, alpha=0.5, edgecolors='w', **scatter_kws)

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])

    # for use of rotating orientation.
    return fig, ax


def aligned_hists(df, bins=10, title=None, xaxis_label=None, yaxis_label=None):
    fig, axes = plt.subplots()

    df.hist(bins=bins, ax=axes, sharex=True)

    if isinstance(title, str):
        plt.suptitle(title)

    if isinstance(xaxis_label, str):
        fig.text(0.5, 0.04, xaxis_label, ha='center')
    if isinstance(yaxis_label, str):
        fig.text(0.04, 0.5, yaxis_label, va='center', rotation='vertical')


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


def hist_func(
    data,
    #func_range,
    func,
    bins=40,
    title=None,
    xaxis_label=None,
    yaxis_label=None,
    nrows=2,
    ncols=2,
    axes_lim=None,
    density=True,
):
    """Plot a function line on top of a histogram."""
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10, 10),
    )

    plot_count = 0
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]

            if plot_count < len(data.columns):
                # Draw histogram
                ax.hist(
                    data.iloc[:, plot_count],
                    bins=bins,
                    color=(0.0, 0.0, 1.0, 0.5),
                    label=str(data.columns[plot_count]),
                    density=density,
                )

                if axes_lim:
                    # Set axis limits
                    ax.set_xlim(axes_lim[:2])
                    # No setting of the y axis limits due to not necessarily
                    # being bounded by the same anyways.
                    #ax.set_ylim(axes_lim[2:])

                # Draw axis values or not (if aligned only draw on left and bottom)
                if j == 0:
                    # Draw the y axis values
                    pass

                if i == nrows - 1:
                    # Draw the x axis values
                    pass

                # Draw the function over the histogram
                #ax.plot(
                # plot the histogram of samples to compare to.
                # look at densities
                ax.hist(
                    func[:, plot_count],
                    bins=bins,
                    color=(1.0, 0.0, 0.0, 0.5),
                    density=density,
                )
            else:
                # Turn off the axes for the blank plots, if any.
                ax.set_axis_off()

            plot_count += 1

    # TODO draw the legend of entire figure, data and function

    if isinstance(title, str):
        plt.suptitle(title)

    if isinstance(xaxis_label, str):
        fig.text(0.5, 0.04, xaxis_label, ha='center')
    if isinstance(yaxis_label, str):
        fig.text(0.04, 0.5, yaxis_label, va='center', rotation='vertical')


def pair_plot_info(
    df,
    title=None,
    axes_lim=None,
    diag_kind='kde',
    diag_kws=None,
    lower_kws=None,
    upper_kws=None,
    pair_grid_args=None,
):
    """Creates a paired plot with hist/kde down the diagonal, scatter plot on
    the upper right triangle, and density plot on the lower right triangle.
    """
    if pair_grid_args is None:
        pair_grid_args = {}

    sns.set_style("whitegrid")
    pg = sns.PairGrid(df, **pair_grid_args)

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


def overlaid_pairplot(dfs, *args, **kwargs):
    """Creates a pair plot of overlaid data."""

    # TODO create / get joint axes for each plot

    # TODO get optional

    for df in dfs:
        pair_plot_info(df, *args, **kwargs)
