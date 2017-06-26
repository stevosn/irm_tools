# -*- coding: utf-8 -*-
# """
# - Author: steve simmert
# - E-mail: steve.simmert@uni-tuebingen.de
# - Copyright: 2015
# """

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

col_dict = {'x': 'blue',
            'y': 'green',
            'z': 'orange',
            'psdX': 'blue',
            'psdY': 'green',
            'psdZ': 'orange',
            'drag': 'DarkTurquoise',
            'ox': 'LightSteelBlue',
            'oy': 'DarkSeaGreen',
            'oz': 'BlanchedAlmond',
            'opsdX': 'LightSteelBlue',
            'opsdY': 'DarkSeaGreen',
            'opsdZ': 'BlanchedAlmond',
            'odrag': 'LightCyan',
            0: 'blue',
            1: 'green',
            2: 'orange',
            3: 'DarkTurquoise',
            4: 'MediumOrchid',
            5: 'LimeGreen',
            6: 'DeepSkyBlue',
            7: 'Teal',
            8: 'Olive',
            9: 'Tomato',
            10: 'LightSteelBlue',
            11: 'DarkSeaGreen',
            12: 'BurlyWood',
            13: 'LightCyan',
            14: 'Plum',
            15: 'LightGreen',
            16: 'LightBlue',
            17: 'PowderBlue',
            18: 'DarkKhaki',
            19: 'LightCoral'
            }


def set_axis_font_size(axis, font_size):
    """ Set the font size of the title, axis labels and tick labels. """
    for item in ([axis.title, axis.xaxis.label, axis.yaxis.label] +
                 axis.get_xticklabels() + axis.get_yticklabels()):
        item.set_fontsize(font_size)


def set_sym_y_labels(axis, mean=None):
    """
    Set symmetric y-limits for the specified axis.
    """
    low, upp = axis.get_ylim()

    if mean is None:
        mean = (upp - low) / 2

    if mean - low <= upp - mean:
        upp_ = upp
        low_ = mean - (upp - mean)
    else:
        upp_ = mean + (mean - low)
        low_ = low

    axis.set_ylim(low_, upp_)


def add_plot_to_figure(figure,
                       xdata,
                       ydata,
                       xerr=[],
                       yerr=[],
                       subplot=(1, 1, 1),
                       axis=None,
                       sharex=None,
                       sharey=None,
                       fmt='-',
                       fontsize=16,
                       markersize=8,
                       linewidth=1.5,
                       alpha=1.0,
                       label='',
                       xlabel='',
                       ylabel='',
                       title='',
                       nbins=4,
                       logplot= False,
                       showLegend=False,
                       figsize=(9, 6),
                       legend_kwargs={},
                       **plot_kwargs):
    """
    Adds a subplot to an existing figure and returns the added axis.

    The function discriminates between normal plots and plots with error bars.
    Errorbars are used if xerr AND yerr OR yerr are not empty arrays.

    Arguments
    ---------
    figure : matplotlib.pyplot.figure or None
    xdata : array
        x data to be plotted.
    ydata : array
        y data to be plotted.
    xerr : array or list
        errors for the x data. Default is an empty list.
    yerr : array or list
        errors for the y data. Default is an empty list.
    subplot : tuple
        Three-element tuple defining the subplot to be used. Defaults to
        (1,1,1)
    axis : matplotlib.Axis or None
        Axis that is used for the plot. If None a new one is added.
    sharex : matplotlib.Axis or None
        shared x axis to be used
    sharey : matplotlib.Axis or None
        shared y axis to be used
    fmt : str
        format string: e.g. '+b'
    fontsize : int
    markersize : int
    linewidth : float
    alpha : float
        transparency for the line/marker
    xlabel : str
    ylabel : str
    title : str
    nbins : int
        number of bins for the axes ticks and labels
    showLegend : bool
        If True, also plots the legend
    legend_kwargs : dict
        dict of legend arguments, e.g. a = {'bbox_to_anchor': (1.05, 1),
        'loc': 2, 'borderaxespad': 0}
    **plot_kwargs : keyword arguments
        keyword arguments that are handed over to pyplot.plot()
        or errorbar function, respectively
    """
    if figure is None:
        figure = plt.figure(figsize=figsize)

    if axis is not None:
        ax = axis
    else:
        ax = figure.add_subplot(*subplot, sharex=sharex, sharey=sharey)

    if len(xerr) == 0 and len(yerr) == 0:
        ax.plot(xdata, ydata, fmt,
                markersize=markersize, linewidth=linewidth, label=label,
                alpha=alpha, **plot_kwargs)
    elif ((xerr == [] or sum(xerr) == 0) and len(yerr) > 0):
        ax.errorbar(xdata, ydata, yerr=yerr, fmt=fmt,
                    markersize=markersize, linewidth=linewidth, label=label,
                    alpha=alpha, **plot_kwargs)
    else:
        ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt=fmt,
                    markersize=markersize, linewidth=linewidth, label=label,
                    alpha=alpha, **plot_kwargs)
    if logplot:
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xscale('linear')
        ax.set_yscale('linear')

    if showLegend:
        if (legend_kwargs != {}):
            ax.legend(**legend_kwargs)
        else:
            ax.legend()

    if ax.get_xscale() is not 'log' and ax.get_yscale() is not 'log':
        ax.locator_params(nbins=nbins)
        ax.xaxis.get_major_formatter().set_powerlimits((-3, 3))
        ax.yaxis.get_major_formatter().set_powerlimits((-3, 3))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    set_axis_font_size(ax, fontsize)

    return ax


def get_gridded_axis(figure=None,
                     figsize=(9, 6),
                     grid=(1, 1),
                     axslice=(slice(None), slice(None)),
                     **axes_kwargs):
    """
    Return an axis of a grid.

    Arguments
    ---------
    figure : matplotlib.Figure
    figsize : tuple
        Size of the figure in inch.
    grid : tuple
        Size of the grid, e.g. grid=(9, 7) for a grid of 9 rows and 7 columns.
    axslice : tuple(row_slice, col_slice)
        Tuple of slices that define the rows and cols to be used for the axis
        that shall be returned. E.g. axslice=(slice(0,4,1), slice(3,6,1))
        for an axis expanding from row 0 to 3 and columns 3 to 5.
    **axes_kwargs : axis keywordarguments
    """
    if figure is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = figure
    gs = gridspec.GridSpec(*grid)
    ax = fig.add_subplot(gs[axslice], **axes_kwargs)
    return ax


def get_residual_plot_axes(nrows=7, ncols=1,
                           row_lim=5, figure=None, figsize=(9, 6)):
    """
    Generate an axes grid in a figure and return a list of 2-tuples with
    the upper axis and lower axis.

    Arguments
    ---------
    nrows, ncols : int
        Number of rows and columns of the axes-grid.
    row_lim : int
        Row from which the lower axis starts.
    figure : Figure
    figsize : tuple(width, height) in inches
    """
    if figure is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = figure

    grid = (nrows, ncols)

    axes = []

    for col in range(ncols):
        ax1 = get_gridded_axis(figure=fig,
                               figsize=figsize,
                               grid=grid,
                               axslice=(slice(0, row_lim, 1),
                                        slice(col, col + 1, 1)))
        if row_lim < nrows:
            ax2 = get_gridded_axis(figure=fig,
                                   figsize=figsize,
                                   grid=grid,
                                   axslice=(slice(row_lim, nrows, 1),
                                            slice(col, col + 1, 1)))
        else:
            ax2 = None

        axes.append((ax1, ax2))

    return axes
