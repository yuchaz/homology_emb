# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD https://github.com/yuchaz/homology_emb/blob/main/LICENSE

from __future__ import division, print_function, absolute_import
import time
import matplotlib as mpl
import seaborn as sns
from seaborn import xkcd_rgb as xkcd
import os
import errno

def isnotebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ImportError):
        return False      # Probably standard Python interpreter


if not isnotebook():
    import matplotlib
    matplotlib.use('agg')

colors_dict = ['purple', 'goldenrod', 'scarlet', 'lawn green', 'windows blue',
               'purplish pink', 'orange', 'teal', 'denim', 'tomato red']
marker_list = ['o', '^', 'p', '*', 'P', 'X', 'D']


def setup_color_palettes(load_extra_typeset=True, reset_ffmpeg_path=True,
                         fs=2.0):
    latex_preamble = [
        r'\usepackage{amssymb}',
        r'\usepackage{amsmath}',
        r'\usepackage{bm}',
        r'\usepackage{palatino}',
        r'\usepackage{mathpazo}',
        r'\usepackage[OT1]{eulervm}',

        r'\usepackage{color}',
        r'\definecolor{scarlet}{RGB}{190, 1, 25}',
        r'\definecolor{crimson}{RGB}{153, 0, 0}',
        r'\definecolor{waterblue}{RGB}{55, 120, 191}',
        r'\definecolor{tangerine}{RGB}{249, 115, 6}',
        r'\definecolor{grassgreen}{RGB}{77, 164, 9}',
        r'\definecolor{gold}{RGB}{250, 194, 5}',

        r'\let\vect\vec',
        r'\renewcommand{\vec}[1]{\bm{\mathbold{#1}}}',
        # r'\renewcommand{\vec}[1]{\mathbf{\bm{#1}}}',
        r'\newcommand{\inv}[1]{\frac{1}{#1}}',
        r'\DeclareMathOperator*{\argmin}{argmin}',
        r'\DeclareMathOperator*{\argmax}{argmax}',
    ]

    if load_extra_typeset:
        latex_preamble.append(r'\usepackage{siunitx}')
        latex_preamble.append(r'\usepackage{mhchem}')

    pgf_preamble = latex_preamble + [
        r'\usepackage{fontspec}',
        r'\setmainfont{Libertinus Serif}',
        r'\setsansfont{Libertinus Sans}',
    ]

    mpl_rc = {
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.preamble': pgf_preamble,
        'text.latex.preamble': latex_preamble,

        'figure.dpi': 150.0,
        'savefig.dpi': 300.0,
        'savefig.bbox': 'tight',
        'image.cmap': 'plasma',

        'font.family': ['serif'],
        'font.serif': ['Linux Libertine'],
    }

    if reset_ffmpeg_path:
        mpl_rc['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    sns.set(font_scale=fs, style='whitegrid', rc=mpl_rc)
    sns.set_palette(sns.xkcd_palette(colors_dict))


def inline_mpl(dpi=150, savedpi=300):
    if isnotebook():
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
    mpl.rcParams['figure.dpi'] = dpi
    mpl.rcParams['savefig.dpi'] = savedpi


def nbook_mpl(dpi=120, savedpi=300):
    if isnotebook():
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'notebook')
    mpl.rcParams['figure.dpi'] = dpi
    mpl.rcParams['savefig.dpi'] = savedpi


def current_palettes():
    sns.palplot(sns.color_palette())


def color_hex(idx):
    return xkcd[colors_dict[idx % len(colors_dict)]]


def makrer_loop(idx):
    return marker_list[idx % len(marker_list)]


def time_counter():
    try:
        return time.perf_counter()
    except AttributeError:  # for python 2
        return time.time()


def create_folder(folder_name, num_try=10):
    if num_try == 0:
        raise OSError(
            'Failed to create folder after several tries -- permission issue')
    try:
        os.makedirs(folder_name, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.EACCES:
            time.sleep(2)
            create_folder(folder_name, num_try-1)
        else:
            raise OSError('Failed to create folder -- %s' % exc)
