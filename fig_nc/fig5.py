# -*- coding: utf-8 -*-
# Author: Kai Chen

#%%
from pathlib import Path
root = Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib.pyplot as plt
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
fig = plt.figure(figsize=(24,14),)
axt = fig.subplots(1, 4, 
    gridspec_kw=dict(wspace=0.4, hspace=0.5,
                     left=0.05, right=0.98,
                     top=1.0, bottom=0.8))
ax = fig.subplots(3, 4, 
    gridspec_kw=dict(wspace=0.4, hspace=0.5,
                     left=0.05, right=0.98,
                     top=0.74, bottom=0.08, height_ratios=[1.2,0.8,2.3]),)

keys = ['Lorenz', 'Logistic', 'Rcon', 'RNN']
spk_fnames = ['Lp=0.25s=0.500f=0.000u=0.000',
              'Logp=0.25s=0.005',
              'Rconp=0.25s=0.002',
              'RNNp=0.25s=0.030tau=20ref=10th=0.200',
            #   'Gaussianp=0.25s=0.030tau=20ref=10th=0.020',
              ]
vol_fnames = ['Lp=0.25s=0.500f=0.000u=0.000_x',
              'Logp=0.25s=0.005_voltage',
              'Rconp=0.25s=0.002_x',
              'RNNp=0.25s=0.030tau=20ref=10_voltage',
            #   'Gaussianp=0.25s=0.030tau=20ref=10_voltage',
              ]

dts = [0.02, 1.0, 3.0, 5.0]
Ts = [1e6, 1e8, 1e7, 1e8]
orders = [(1,1), (2,1), (5,5), (5,1)]
delays = [0, 0, 0, 16]
ths = [10, 0.9, 8, 0.2]
Tranges = [10, 30, 50, 600]


regen=False
for axti, ax_col, spk_fname, key, dt, T, delay, order, vol_fname, th, Trange in zip(axt, ax.T, spk_fnames, keys, dts, Ts, delays, orders, vol_fnames, ths, Tranges):
    subfolder = root / f'benchmark/N100/{key:s}'
    N = 100

    print(np.fromfile(subfolder/'connect_matrix-p=0.250.dat', dtype=float).reshape(N, N)[0,1])

    img = plt.imread(key+'.png')  # Replace with the actual path to your PNG file
    axti.imshow(img, aspect='equal')
    axti.axis('off')  # Hide axes if desired

    spks = c4u.load_spike_data(subfolder/(spk_fname + '_spike_train.dat'), xrange=(0, Trange))
    if (subfolder/(vol_fname + '.npy')).exists():
        voltages = np.load(subfolder/(vol_fname + '.npy'), mmap_mode='r')
        voltage_dt = voltages[1,0] - voltages[0,0]
        Tn = int(Trange / voltage_dt)
        voltages = voltages[:Tn, :]
    elif (subfolder/(vol_fname + '.dat')).exists():
        voltages = c4u.fetch_voltage(subfolder/(vol_fname + '.dat'), N=N, voltage_range=(0, Trange))
    ax_col[0].plot(voltages[:,0], voltages[:,1], lw=2, color='#3532A0', clip_on=True)
    ax_col[0].plot(voltages[:,0], voltages[:,2], lw=2, color='C2', clip_on=True)
    ax_col[0].axhline(th, ls='--', color='r', lw=2, clip_on=True)
    ax_col[0].set_ylabel('activity', fontsize=26)
    ax_col[0].set_xlabel('time (ms)', fontsize=26)
    ax_col[0].set_xlim(0, Trange)
    mask = spks[:,1] == 0
    ax_col[1].plot(spks[mask,0], spks[mask,1], '|', color='C0', ms=30, mew=3, clip_on=True)
    mask = spks[:,1] == 1
    ax_col[1].plot(spks[mask,0], spks[mask,1], '|', color='C2', ms=30, mew=3, clip_on=True)
    ax_col[1].axhline(0, ls='--', color='k', lw=1, clip_on=True)
    ax_col[1].axhline(1, ls='--', color='k', lw=1, clip_on=True)
    ax_col[1].spines['left'].set_visible(False)
    ax_col[1].set_xlim(0, Trange)
    ax_col[1].set_ylim(-0.5, 1.5)
    ax_col[1].set_xlabel('time (ms)', fontsize=26)
    ax_col[1].set_yticks([])
    # ax_col[1].set_ylabel('neuron ID', fontsize=26)

    estimator = CausalityEstimator(
        subfolder, spk_fname, N, delay=delay, T=T, dt=dt,
        n_thread=128, order=order,
    )
    data = estimator.fetch_data(new_run=True)
    data_matched = match_features(data, N, subfolder/'connect_matrix-p=0.250.dat')
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=50, algorithm='EM')
    print('acc: %.4f, auc: %.4f'%(df_fig['acc_svm']['TE'], df_fig['auc_svm']['TE']))
    RED, GREEN = '#F49227', '#194955'
    tmp = df_fig.loc['TE']
    for hist_key, color in zip(('hist_conn', 'hist_disconn'), (RED, GREEN)):
        edges = tmp['edges'] + (tmp['edges'][1] - tmp['edges'][0])/2
        counts = tmp[hist_key]
        mask = counts > 0
        ax_col[-1].plot(edges[mask], counts[mask], color=color, lw=5, clip_on=True)
        ax_col[-1].fill_between(edges[mask], 0, counts[mask], color=color, alpha=0.5)
    ax_col[-1].axvline(tmp['th_svm'], ls='-', color='#F26A9D', lw=4)
    if key == 'Lorenz':
        ax_col[-1].set_xlim(-11, -4)
        ax_col[-1].set_xticks([-10, -8, -6, -4])
    elif key == 'Logistic':
        ax_col[-1].set_xlim(-7, -4)
    ax_col[-1].xaxis.set_major_formatter(sci_formatter)
        # format_xticks(ax_col[-1], (-7, -4))
    ax_col[-1].set_ylim(0)
    ax_col[-1].set_xlabel('PTD-TE value', fontsize=26)
    ax_col[-1].set_ylabel('density', fontsize=26)

xx, yy = np.meshgrid(np.arange(4), np.arange(3), indexing='ij')
for tag, x in zip(['abc','def','ghi','jkl'], range(4)):
    fig.text(0.02+x*0.25, 1.00, tag[0], fontsize=35, fontweight='bold', va='top')
    fig.text(0.02+x*0.25, 0.78, tag[1], fontsize=35, fontweight='bold', va='top')
    fig.text(0.02+x*0.25, 0.38, tag[2], fontsize=35, fontweight='bold', va='top')

fig.savefig(root/'fig_nc/pdf'/'fig5.pdf', transparent=True)
#%%
