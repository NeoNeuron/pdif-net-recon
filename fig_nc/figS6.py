# -*- coding: utf-8 -*-
# Author: Kai Chen

#%%
from pathlib import Path
root = Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib.pyplot as plt
from causal4.Causality import CausalityEstimator
from causal4.utils import match_features, reconstruction_analysis_TE
import causal4.utils as c4u
from figrc import *
RED, GREEN = '#F49227', '#194955'

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


def _plot_figures(key, spk_fname, vol_fname, dt, T, order, delay, ths, Trange, labels, axes):
    for ax_col, th in zip(axes.T, ths):
        subfolder = root / f'benchmark/N100/{key:s}'
        N = 100

        print(np.fromfile(subfolder/'connect_matrix-p=0.250.dat', dtype=float).reshape(N, N)[0,1])

        spks = c4u.load_spike_data(subfolder/(spk_fname(th) + '_spike_train.dat'), xrange=(0, Trange))
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
        ax_col[0].set_title(r'$x^\mathrm{th}$ = %.2f'%th, fontsize=26)
        mask = spks[:,1] == 0
        ax_col[1].plot(spks[mask,0], spks[mask,1], '|', color='#3532A0', ms=30, mew=3, clip_on=True)
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
            subfolder, spk_fname(th), N, delay=delay, T=T, dt=dt,
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
            ax_col[-1].set_xticks([-10, -8, -6, -4])
            add_log_minor_ticks(ax_col[-1], (-11,-4), where='x')
            ax_col[-1].set_xlim(-11, -4)
        elif key == 'Logistic':
            add_log_minor_ticks(ax_col[-1], (-7,-4), where='x')
            ax_col[-1].set_xlim(-7, -4)
        elif key == 'Rcon':
            ax_col[-1].set_xticks([-4, -3])
            add_log_minor_ticks(ax_col[-1], (-5,-3), where='x')
            ax_col[-1].set_xlim(-4.2, -3)
        elif key == 'RNN':
            ax_col[-1].set_xticks([-6, -5, -4])
            add_log_minor_ticks(ax_col[-1], (-7,-3), where='x')
            ax_col[-1].set_xlim(-6.3, -4)
        ax_col[-1].tick_params(axis='x', which='major', length=8)
        ax_col[-1].tick_params(axis='x', which='minor', length=4)
        ax_col[-1].xaxis.set_major_formatter(sci_formatter)
        ax_col[-1].tick_params(axis='both', which='major', labelsize=22)
            # format_xticks(ax_col[-1], (-7, -4))
        ax_col[-1].set_ylim(0)
        ax_col[-1].set_xlabel('PDIF value', fontsize=26)
        ax_col[-1].set_ylabel('density', fontsize=26)

    for tag, axi in zip(labels, axes[0]):
        axi.text(-0.25, 1.5, tag, fontsize=35, va='top', transform=axi.transAxes)


#%%
fig = plt.figure(figsize=(28,15),)
axes0 = fig.subplots(3, 5, 
    gridspec_kw=dict(wspace=0.5, hspace=0.7,
                     left=0.05, right=0.88,
                     top=0.95, bottom=0.58, height_ratios=[1.2,0.8,1.8]),)

key = 'Rcon'
spk_fname = lambda th: f'Rconp=0.25s=0.002_th={th:.2f}ref=5.00'
vol_fname = 'Rconp=0.25s=0.002_x'
dt = 3.0
T = 1e7
order = (5,5)
delay = 0
ths = [2.5, 5.0, 7.5, 10.0,12.5]
Trange = 50
_plot_figures(key, spk_fname, vol_fname, dt, T, order, delay, ths, Trange, 'ABCDE', axes0)

axes01 = fig.subplots(1, 1, 
    gridspec_kw=dict(left=0.94, right=0.98,
                     top=0.95, bottom=0.58))

ths = np.arange(2.5, 15.1, 2.5)
acc_list, auc_list = [], []
for th in ths:
    subfolder = root / f'benchmark/N100/{key:s}'
    N = 100
    estimator = CausalityEstimator(
        subfolder, spk_fname(th), N, delay=delay, T=T, dt=dt,
        n_thread=128, order=order,
    )
    data = estimator.fetch_data(new_run=True)
    data_matched = match_features(data, N, subfolder/'connect_matrix-p=0.250.dat')
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=50, algorithm='EM')
    print('acc: %.4f, auc: %.4f'%(df_fig['acc_svm']['TE'], df_fig['auc_svm']['TE']))
    acc_list.append(df_fig['acc_svm']['TE'])
    auc_list.append(df_fig['auc_svm']['TE'])
    
axes01.plot(ths, auc_list, marker='o', label='accuracy', ms=8, c=GREEN, clip_on=False)
axes01.set_xlabel(r'$x^\mathrm{th}$', fontsize=26)
axes01.set_ylabel('AUC', fontsize=26)
axes01.set_xticks([0,15])
axes01.set_xlim(0, 15)
axes01.set_ylim(0.5, 1)
axes01.text(-0.5, 1.11, 'F', fontsize=35, va='top', transform=axes01.transAxes)

axes1 = fig.subplots(3, 5, 
    gridspec_kw=dict(wspace=0.5, hspace=0.7,
                     left=0.05, right=0.88,
                     top=0.44, bottom=0.07, height_ratios=[1.2,0.8,1.8]),)

key = 'RNN'
spk_fname = lambda th: f'RNNp=0.25s=0.030tau=20ref=10_th={th:.2f}ref=3.00'
vol_fname = 'RNNp=0.25s=0.030tau=20ref=10_voltage'
dt = 5.0
T = 1e8
order = (5,1)
delay = 16
ths = [0.1, 0.15, 0.2, 0.25, 0.3]
Trange = 600

_plot_figures(key, spk_fname, vol_fname, dt, T, order, delay, ths, Trange, 'GHIJK', axes1)

axes11 = fig.subplots(1, 1, 
    gridspec_kw=dict(left=0.94, right=0.98,
                     top=0.44, bottom=0.07))

ths = np.arange(0.10, 0.41, 0.05)
acc_list, auc_list = [], []
for th in ths:
    subfolder = root / f'benchmark/N100/{key:s}'
    N = 100
    estimator = CausalityEstimator(
        subfolder, spk_fname(th), N, delay=delay, T=T, dt=dt,
        n_thread=128, order=order,
    )
    data = estimator.fetch_data(new_run=True)
    data_matched = match_features(data, N, subfolder/'connect_matrix-p=0.250.dat')
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=50, algorithm='EM')
    print('acc: %.4f, auc: %.4f'%(df_fig['acc_svm']['TE'], df_fig['auc_svm']['TE']))
    acc_list.append(df_fig['acc_svm']['TE'])
    auc_list.append(df_fig['auc_svm']['TE'])

axes11.plot(ths, auc_list, marker='o', label='accuracy', ms=8, c=GREEN, clip_on=False)
axes11.set_xlabel(r'$x^\mathrm{th}$', fontsize=26)
axes11.set_ylabel('AUC', fontsize=26)
axes11.set_xticks([0,0.4])
axes11.set_xlim(0, 0.4)
axes11.set_ylim(0.5, 1)
axes11.text(-0.5, 1.11, 'L', fontsize=35, va='top', transform=axes11.transAxes)


fig.savefig(root/'fig_nc/pdf'/'figS8.pdf', transparent=True)
#%%
