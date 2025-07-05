# -*- coding: utf-8 -*-
# Author: Kai Chen

#%%
from pathlib import Path
root = Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from causal4.Causality import CausalityEstimator
from causal4.utils import match_features, reconstruction_analysis_TE
from causal4.myplot import ReconstructionFigureTE
import networkx as nx
import causal4.utils as c4u
import pickle as pkl
from figrc import *

def get_vfname(fname: str, key:str, sfx:str=None):
    if key == 'Gaussian':
        fname = fname.replace('th=0.020','')
    if fname.startswith('Lp') or fname.startswith('Lcon'):
        fname = fname + '_x'
    else:
        fname = fname + '_voltage'
    if sfx is not None:
        fname += sfx
    return fname + '.dat'

#%%
fig = plt.figure(figsize=(20,5),)
ax = fig.subplots(1, 4, 
    gridspec_kw=dict(wspace=0.4, hspace=0.5,
                     left=0.05, right=0.98,
                     top=0.90, bottom=0.15),)
ax = ax.reshape(2,2)

spk_fnames = ['HHp=0.25s=0.020f=0.080u=0.150', 'HHp=0.25s=0.020f=0.080u=0.150_noisy']
for axi, spk_fname in zip(ax, spk_fnames):
    subfolder = 'HH100'
    N = 100
    T = 1e7
    spks = c4u.load_spike_data(root/subfolder/('HHp=0.25s=0.020f=0.080u=0.150' + '_spike_train.dat'), xrange=(1000, 1100))
    axi[0].plot(spks[:,0], spks[:,1], 'k|', markersize=10, mew=3, clip_on=True, label='shuffled spikes')
    axi[0].set_xlim(1000, 1100)
    axi[0].set_ylim(0, N)
    axi[0].set_xlabel('time (ms)', fontsize=26)
    axi[0].set_ylabel('neuron ID', fontsize=26)
    estimator = CausalityEstimator(
        root/subfolder, spk_fname, N, delay=3, T=T, dt=0.5,
        n_thread=64, order=(1,1),
    )
    data = estimator.fetch_data(new_run=True)
    data_matched = match_features(data, N, root/subfolder/'connect_matrix-p=0.250.dat')
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=60, hist_range=None, algorithm='EM')
    RED, GREEN = '#F49227', '#194955'
    tmp = df_fig.loc['TE']
    for hist_key, color in zip(('hist_conn', 'hist_disconn'), (RED, GREEN)):
        edges = tmp['edges'] + (tmp['edges'][1] - tmp['edges'][0])/2
        counts = tmp[hist_key]
        mask = counts > 0
        axi[1].plot(edges[mask], counts[mask], color=color, lw=5, clip_on=True)
        axi[1].fill_between(edges[mask], 0, counts[mask], color=color, alpha=0.5)
    axi[1].axvline(tmp['th_svm'], ls='-', color='#F26A9D', lw=4)
    axi[1].set_xlim(-8, -4)
    axi[1].xaxis.set_major_formatter(sci_formatter)
    axi[1].set_ylim(0)
    axi[1].set_yticks([0,0.5,1.0])
    axi[1].set_xlabel('PTD-TE value', fontsize=26)
    axi[1].set_ylabel('density', fontsize=26)

spks = c4u.load_spike_data(root/subfolder/(spk_fnames[1] + '_spike_train.dat'), xrange=(1000, 1100))
ax[1,0].plot(spks[:,0], spks[:,1], 'r|', markersize=10, mew=2, clip_on=True, label='raw spikes')
ax[1,0].set_xlim(1000, 1100)
ax[1,0].set_ylim(0, N)
ax[1,0].set_xlabel('time (ms)', fontsize=26)
ax[1,0].set_ylabel('neuron ID', fontsize=26)
# ax[1,0].legend(fontsize=16, loc=(0.35,0.9), frameon=True)

for x, tag in enumerate('abcd'):
    fig.text(x*0.25, 0.995, tag, fontsize=35, fontweight='bold', va='top')

fig.savefig(root/'fig_nc/pdf'/'fig4.pdf', transparent=True)
#%%
