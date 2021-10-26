# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/yuchaz/homology_emb/blob/main/LICENSE

import matplotlib.pyplot as plt
from configs import color_hex, create_folder
import itertools
import numpy as np
from matplotlib.lines import Line2D
import os
import matplotlib as mpl


def ordinal(n):
    return "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])


def _flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def figlegend_to_buttom(ncol=3, dot_size=6, transpose=True, x0=0.5, y0=0.05,
                        handles=None, labels=None, **lgd_dict):
    fig = plt.gcf()
    subplot_axes = fig.get_axes()
    for ax in subplot_axes:
        box = ax.get_position()
    lgd_dict.update(dict(loc='center', bbox_to_anchor=(x0, y0),
                    fancybox=True, ncol=ncol, framealpha=0.0))

    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if transpose:
        legend = plt.figlegend(
            _flip(handles, ncol), _flip(labels, ncol), **lgd_dict)
    else:
        legend = plt.figlegend(handles, labels, **lgd_dict)

    for idx in range(len(legend.legendHandles)):
        legend.legendHandles[idx]._sizes = [dot_size]


def pairplot(hloop, figsize=(20, 20)):
    beta1 = hloop.harmonic_evects.shape[1]
    __, axes = plt.subplots(beta1, beta1, figsize=figsize)
    limits_hemb = (hloop.harmonic_evects.min(), hloop.harmonic_evects.max())
    limits_iemb = (hloop.independent_harmonic_evects.min(),
                   hloop.independent_harmonic_evects.max())
    limits = [min(limits_hemb[0], limits_iemb[0]) * 0.95,
              max(limits_hemb[1], limits_iemb[1]) * 1.05]
    for i in range(beta1):
        for j in range(beta1):
            tickpm_kwargs = dict(bottom=True, left=True, right=True, top=True,
                                 direction='in', labelleft=False,
                                 labelbottom=False)
            if i == beta1-1:
                tickpm_kwargs.update(dict(labelbottom=True))
                axes[i, j].set_xlabel(ordinal(j+1))
            if j == 0:
                tickpm_kwargs.update(dict(labelleft=True))
                axes[i, j].set_ylabel(ordinal(i+1))
            if i == 0:
                tickpm_kwargs.update(dict(labeltop=True))
                axes[i, j].set_xlabel(ordinal(j+1))
                axes[i, j].xaxis.set_label_position('top')
            if j == beta1-1:
                tickpm_kwargs.update(dict(labelright=True))
                axes[i, j].set_ylabel(ordinal(i+1))
                axes[i, j].yaxis.set_label_position('right')
            if j == i:
                tickpm_kwargs.update(dict(labelleft=True, right=False))
            if j+1 == i:
                tickpm_kwargs.update(dict(right=False))


            axes[i, j].tick_params(**tickpm_kwargs)
            axes[i, j].grid(False)
            axes[i,j].set_xlim(limits)

            if i == j:
                axes[i, j].grid(False)
                bins = np.linspace(limits[0], limits[1], 100)
                axes[i, j].hist(
                    hloop.harmonic_evects[:, i], bins=bins,
                    color=color_hex(2),
                    histtype='step'
                )
                axes[i, j].hist(
                    hloop.independent_harmonic_evects[:, i], bins=bins,
                    color=color_hex(4),
                    histtype='step'
                )
            else:
                used_emb = (hloop.independent_harmonic_evects[:, [j, i]]
                            if i > j else
                            hloop.harmonic_evects[:, [j, i]])
                color = color_hex(4) if i > j else color_hex(2)
                axes[i, j].scatter(*used_emb.T, s=.1, edgecolors='none',
                                   rasterized=True, c=color)
                axes[i,j].set_ylim(limits)

    plt.subplots_adjust(hspace=.0, wspace=.0)
    custom_lines = [Line2D([0], [0], color=color_hex(4)),
                    Line2D([0], [0], color=color_hex(2))]
    figlegend_to_buttom(
        handles=custom_lines, labels=[
            r'Indep. harmonic embedding $\vec{z}$',
            r'Harmonic embedding $\vec{y}$'],
        x0=0.4,
    )


def savefig_function_gen(working_dir):
    create_folder(working_dir)
    savedpi = mpl.rcParams['savefig.dpi']

    def savefig(name, subdir='', nopdf=False, **kwargs):
        folder = os.path.join(working_dir, subdir)
        if not os.path.exists(folder):
            os.makedirs(folder)

        if not nopdf:
            plt.savefig(os.path.join(folder, f'{name}.pdf'), **kwargs)
        dpi = savedpi if not nopdf else 600
        if 'dpi' not in kwargs:
            kwargs['dpi'] = dpi
        plt.savefig(os.path.join(folder, f'{name}.png'), **kwargs)

    savefig.__doc__ = f'''
        Save pdf/png plots under {working_dir}

        Inputs
        ------
        name: str
            Name of the file, will be saved under working_dir/name.png or .pdf
        subdir: str
            Subdirectory for the figure to be saved, create if not exist
        nopdf: bool
            if save figures with pdf, if True, will save png with dpi=600


        Returns
        -------
        None
        '''

    return savefig
