from pathlib import Path
root = Path(__file__).resolve().parents[1]
import matplotlib as mpl
rc_path = Path(__file__).with_name("matplotlibrc")
mpl.rcParams.update(mpl.rc_params_from_file(rc_path, use_default_template=False))
ORANGE, GREEN, PINK = '#F49227', '#194955', '#F26A9D'

import networkx as nx
import numpy as np
np.seterr(all="ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
@FuncFormatter
def sci_formatter(x, pos):
    return r'$10^{%d}$'%x

from pdif.pdif import CausalityEstimator
from pdif.utils import match_features, reconstruction_analysis_TE
from pdif.downsample import downsample, population_synchrony_index

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def add_log_minor_ticks(ax, vrange, where='x'):
    ticks_minor = [np.arange(1,10)*10**i for i in range(*vrange)]
    ticks_minor = np.log10(np.asarray(ticks_minor).flatten())
    if where == 'x':
        ax.set_xticks(ticks_minor, minor=True)
    elif where == 'y':
        ax.set_yticks(ticks_minor, minor=True)
    return ax

def make_graph_diagram(G, ax, pos,
    node_size=2000, font_size=26,
    arrow_size=20, arrow_width=2,
    node_color='#F49227'):
    nx.draw_networkx_nodes(
        G, pos=pos, ax=ax,
        node_color=node_color,
        edgecolors='k',
        node_size=node_size,
    )
    nx.draw_networkx_labels(
        G, pos=pos, ax=ax,
        labels={n:n for n in G},
        font_size=font_size,
        font_weight='bold',
    )
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax,
        width=arrow_width,
        node_size=node_size,
        arrowsize=arrow_size,
        arrows=True,
    )
    ax.set_clip_on(False)
    ax.axis('equal')
    ax.axis('off')


class zoomedAxes(object):
    def __init__(self, ax, zoom_xrange, zoom_yrange, inset_anchor):
        self.ax = ax
        self.zax = self.ax.inset_axes(inset_anchor)
        self.zax.set_xlim(*zoom_xrange)
        self.zax.set_ylim(*zoom_yrange)
        self.zax.spines['top'].set_visible(True)
        self.zax.spines['right'].set_visible(True)
        self.rect = plt.Rectangle(
            (zoom_xrange[0], zoom_yrange[0]),
            zoom_xrange[1] - zoom_xrange[0],
            zoom_yrange[1] - zoom_yrange[0],
            edgecolor='#777777', facecolor='none', linestyle='--')
        self.ax.add_patch(self.rect)


    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)
        self.zax.plot(*args, **kwargs)

    def axhline(self, *args, **kwargs):
        self.ax.axhline(*args, **kwargs)
        self.zax.axhline(*args, **kwargs)


from scipy.optimize import curve_fit
def linearfit(x, y, bias=False):
    if bias:
        def func(x, a, b):
            return a*x + b
    else:
        def func(x, a):
            return a*x
    popt, pcov = curve_fit(func, x, y)
    fit = func(x, *popt)
    ss_res = np.sum((y - fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0
    se = np.sqrt(np.diag(pcov))
    ci95 = 1.96 * se
    if bias:
        print(f"linearfit: slope={popt[0]:.2e} ± {ci95[0]:.2e} (95% CI), intercept={popt[1]:.2e} ± {ci95[1]:.2e} (95% CI), R^2={r_squared:.3f}")
    else:
        print(f"linearfit: slope={popt[0]:.2e} ± {ci95[0]:.2e} (95% CI), R^2={r_squared:.3f}")
    
    return lambda x: func(x, *popt)

def squarefit(x, y):
    def func(x, a):
        return a*x**2
    popt, pcov = curve_fit(func, x, y)
    fit = func(x, *popt)
    ss_res = np.sum((y - fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0
    se = np.sqrt(np.diag(pcov))[0]
    ci95 = 1.96 * se
    print(f"squarefit: A={popt[0]:.2e} ± {ci95:.2e} (95% CI), R^2={r_squared:.3f}")
    return lambda x: func(x, *popt)

def create_fig1x4():
    fig = plt.figure(figsize=(20,5))
    axes = fig.subplots(1, 4, 
        gridspec_kw=dict(wspace=0.4, hspace=0.5,
                        left=0.05, right=0.98,
                        top=0.93, bottom=0.18),)
    return fig, axes

def create_fig2x4():
    fig = plt.figure(figsize=(20,10.71))
    axes = fig.subplots(2, 4, 
        gridspec_kw=dict(wspace=0.4, hspace=0.5,
                        left=0.05, right=0.98,
                        top=0.95, bottom=0.075),)
    return fig, axes

def plot_conn_matrix(ax, conn, title=None):
    ax.pcolormesh(conn, cmap='viridis')
    ax.set_ylabel('From', fontsize=26)
    ax.set_xlabel('To', fontsize=26)
    ax.invert_yaxis()
    ax.set_rasterized(True)
    if title:
        ax.set_title(title, fontsize=22, pad=-10)

def plot_spike_train(ax, N, spk, spk_all=None, title=None):
    ax.plot(spk[:, 0], spk[:, 1], 'k|')
    if spk_all is not None:
        spk_diff = spk_all[~np.isin(spk_all[:, 0], spk[:, 0])]
        ax.plot(spk_diff[:, 0], spk_diff[:, 1], 'r|')
    ax.set_xlabel('time (ms)', fontsize=26)
    ax.set_ylabel('neuron ID', fontsize=26)
    ax.set_ylim(0, N)
    ax.set_xlim(600,800)
    if title:
        ax.set_title(title, fontsize=22, pad=-10)

def motif_path_counts(conn):
    """Per-pair counts of second-order motif paths, for each unconnected
    ordered pair (i,j) in binary adjacency `conn` (W[i,j] = edge i->j):
    common-driver (confounder) paths -- number of k with k->i and k->j -- and
    2-step mediator (chain) paths -- number of k with i->k and k->j.

    Returns:
        (confounder_counts, chain_counts): 1D arrays, one entry per
        unconnected ordered pair (i,j), i.e. length == number of unconnected
        ordered pairs in `conn`.
    """
    A = (conn > 0).astype(float)
    off_diag = ~np.eye(A.shape[0], dtype=bool)
    unconnected = (A == 0) & off_diag
    confounder_counts = (A.T @ A)[unconnected]
    chain_counts = (A @ A)[unconnected]
    return confounder_counts, chain_counts


def maybe_downsample(path, fname, ra, do_downsample):
    """Downsample a spike-train file when `do_downsample` is True, else pass it
    through unchanged.

    Returns:
        (spk_fname, raw_fname_or_None): `spk_fname` is the file to run causality
        estimation on; `raw_fname_or_None` is the pre-downsample file (for
        highlighting removed spikes via `plot_spike_train`'s `spk_all`), or
        None if no downsampling was performed.
    """
    if do_downsample:
        ofname = fname.replace('_spike_train.dat', f'_ra={ra:.2f}_spike_train.dat')
        downsample(fname, ofname, path, ra)
        return Path(path/ofname), Path(path/fname)
    return Path(path/fname), None


def run_reconstruction_TE(path, spk_fname, N, conn_file, T, dt=0.5, delay=3,
                           order=(1, 1), n_thread=128, match_kwargs=None, recon_kwargs=None):
    """Run the CausalityEstimator -> match_features -> reconstruction_analysis_TE
    pipeline shared by the figR4* reviewer-response scripts.

    Returns:
        (df_recon, df_fig)
    """
    estimator = CausalityEstimator(
        path, spk_fname, N, delay=delay, T=T, dt=dt,
        n_thread=n_thread, order=order,
    )
    data = estimator.fetch_data(new_run=True)
    data_matched = match_features(data, N, conn_file, **(match_kwargs or {}))
    return reconstruction_analysis_TE(data_matched, **(recon_kwargs or {}))


def plot_pdif_hist(series, ax, threshold=True):
    real_xlim = []
    for hist_key, color in zip(('hist_conn', 'hist_disconn'), (ORANGE, GREEN)):
        edges = series['edges'] + (series['edges'][1] - series['edges'][0])/2
        counts = series[hist_key]
        mask = counts > 0
        real_xlim.append([edges[mask][0], edges[mask][-1]])
        ax.plot(edges[mask], counts[mask], color=color, lw=5, clip_on=True)
        ax.fill_between(edges[mask], 0, counts[mask], color=color, alpha=0.5)
    if threshold:
        ax.axvline(series['th_gauss'], ls='-', color=PINK, lw=4)
    ax.xaxis.set_major_formatter(sci_formatter)
    ax.set_ylim(0)
    ax.set_xlim(min(x[0] for x in real_xlim), max(x[-1] for x in real_xlim))
    ax.set_xlabel('PDIF value', fontsize=26)
    ax.set_ylabel('density', fontsize=26)
