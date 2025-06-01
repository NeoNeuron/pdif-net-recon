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
fig = plt.figure(figsize=(24,10),)

ax = fig.subplots(2, 4, 
    gridspec_kw=dict(wspace=0.4, hspace=0.5,
                     left=0.05, right=0.98,
                     top=0.94, bottom=0.08),)

spk_fnames = ['HHp=0.25s=0.020f=0.080u=0.150', 'HHp=0.25s=0.020f=0.080u=0.150_noisy']
for axi, spk_fname in zip(ax[:,0], spk_fnames):
    subfolder = 'HH100'
    N = 100
    T = 1e7
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
        axi.plot(edges[mask], counts[mask], color=color, lw=5, clip_on=True)
        axi.fill_between(edges[mask], 0, counts[mask], color=color, alpha=0.5)
    axi.axvline(tmp['th_svm'], ls='-', color='#F26A9D', lw=4)
    axi.set_xlim(-8, -4)
    axi.xaxis.set_major_formatter(sci_formatter)
    axi.set_ylim(0)
    axi.set_xlabel('PTD-TE value', fontsize=26)
    axi.set_ylabel('density', fontsize=26)

spk_fnames = ['Lp=0.25s=0.500f=0.000u=0.000',
              'Logp=0.25s=0.005f=0.000u=0.000']

keys = ['Lorenz', 'Logistic']

sss = [np.arange(0.000,1.01,0.04),
       np.arange(0.000,0.021,0.001)]

dts = [0.02, 1.0]
acf_xmaxs = [10, 20]

regen=False
for ax_col, spk_fname, key, ss, dt, acf_xmax in zip(ax[:,1:], spk_fnames, keys, sss, dts, acf_xmaxs):

    zax = zoomedAxes(ax_col[0], (-0.1, acf_xmax/2), (-0.1,0.1), [0.4, 0.5, 0.6, 0.55])

    subfolder = key+'3_scan_S'
    for fname, C in zip([f'ACF_SPK_{key:s}3_scan_S_{spk_fname:s}.npz',
                         f'ACF_VOL_{key:s}3_scan_S_{spk_fname:s}.npz'], ['#F49227', '#194955']):
        if not (root / subfolder / fname).exists() or regen:
            tmp = (root / subfolder / fname).stem.split('_')
            data_type = tmp[1]
            spk_fname = tmp[-1]
            if data_type == 'SPK':
                spk_data = c4u.load_spike_data(root / subfolder / f'{spk_fname:s}_spike_train.dat',
                                        xrange=(0,1e7))
                dt = 0.5
                time_series = c4u.spk2bin(spk_data[spk_data[:,1]==0], dt=dt)
            elif data_type == 'VOL':
                vol_data = c4u.fetch_voltage(root / subfolder / get_vfname(spk_fname, key),
                                        N=3, voltage_range=None)
                dt = vol_data[1,0] - vol_data[0,0]
                time_series = vol_data[:,1]
            dTn = int(time_series.shape[0]/100)
            acf = []
            if key == 'Lorenz':
                nlags = 500 if data_type == 'SPK' else 1000
            else:
                nlags = 100 if data_type == 'SPK' else 250
            for i in range(100):
                seg = time_series[i*dTn:(i+1)*dTn]
                acf_seg = c4u.ACF(seg, nlags=nlags)
                acf.append(acf_seg)
            acf = np.array(acf)
            t_lag = np.arange(acf.shape[-1])*dt
            np.savez(root / subfolder / fname, t_lag=t_lag, acf=acf)
        else:
            data = np.load(root / subfolder / fname)
            t_lag = data['t_lag']
            acf = data['acf']
        zax.plot(t_lag, acf.mean(0), '-', lw=2.5, color=C)
        zax.axhline(0, ls='--', color='#AAAAAA')
        ax_col[0].set_xlabel('time-lag (ms)', fontsize=26)
        ax_col[0].set_ylabel('ACF', fontsize=26, labelpad=-20)
        zax.zax.set_xlabel('time-lag (ms)', fontsize=14, labelpad=0)
        zax.zax.set_ylabel('ACF', fontsize=14, labelpad=-10)
        for tick in zax.zax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in zax.zax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        if key == 'HH':
            zax.zax.set_xticks([0, 25])
    ax_col[0].set_xlim(-0.1, acf_xmax)
    ax_col[0].set_ylim(-0.3, 1.00)
    ax_col[0].set_yticks([-0.3, 0, 0.5, 1.0])
    # ax_col[0].set_title(key+' networks', fontsize=24, pad=22)

    subfolder = key+'3_scan_S'
    N = 3
    order = (2,1) if key == 'Logistic' else (1,1)
    delay = 3 if key == 'HH' else 0
    estimator = CausalityEstimator(
        root/subfolder, '', N, delay=delay, T=1e7, dt=dt, order=order,
    )

    ptdte = []
    dps = []
    for s in ss:
        idx = spk_fname.find('s=')
        if idx != -1:
            estimator.spk_fname = spk_fname[:idx+2] + f"{s:.3f}" + spk_fname[idx+7:]
        data = estimator.fetch_data(new_run=False)
        data01 = data[data['pre_id'].eq(0) * data['post_id'].eq(1)]['TE'].values[0]
        data10 = data[data['pre_id'].eq(1) * data['post_id'].eq(0)]['TE'].values[0]
        data02 = data[data['pre_id'].eq(0) * data['post_id'].eq(2)]['TE'].values[0]
        dps.append(data[data['pre_id'].eq(0) * data['post_id'].eq(1)]['dp1'].values[0])
        ptdte.append([data01, data10, data02])
    ptdte = np.array(ptdte)
    dps = np.array(dps)
    line_settings = [
        {'color':'#1E3B7A', 'ls':'none', 'marker':'o', 'ms':14, 'mec':'none'},
        {'color':'#EE781F', 'ls':'none', 'marker':'*', 'ms':14 },
        {'mec':'#108B96', 'mfc':'none', 'mew':2, 'ls':'none', 'marker':'o', 'ms':16},
    ]
    #108B96
    # CA462F
    for i, (line_setting, label) in enumerate(zip(line_settings, ['X->Y', 'Y->X', 'X->Z'])):
        ax_col[1].plot(ss, ptdte[:,i], **line_setting, clip_on=False)
    ffit = squarefit(ss, ptdte[:,0])
    ss_fit = np.linspace(ss[0], ss[-1], 100)
    ax_col[1].plot(ss_fit, ffit(ss_fit), '-', color='#F26A9D', lw=4, zorder=-1)
    # ax_col[1].plot(ss, ptdte, '-o', ms=8, mec='w', clip_on=False)
    # ax_col[1].legend([r'$T^\mathrm{PTD}_{X\to Y}$',
    #                   r'$T^\mathrm{PTD}_{Y\to X}$',
    #                   r'$T^\mathrm{PTD}_{X\to Z}$'], fontsize=14, loc='lower right')
    ax_col[1].set_xlabel(r'S', fontsize=26)
    ax_col[1].set_ylabel('PTD-TE value', fontsize=26)
    ax_col[1].set_xlim(0, ss[-1])
    ax_col[1].set_ylim(0, ffit(ss[-1])*1.3)
    ax_col[1].ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
    ax_col[1].tick_params(axis='x', pad=10)  # Increase x-axis tick label padding
    ax_col[1].yaxis.get_offset_text().set_x(-0.2)  # Move y-axis offset label to the left

    # s vs dp LARGE
    axins = ax_col[1].inset_axes([0.18, 0.65, 0.5, 0.4])
    axins.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)

    ffit = linearfit(ss, dps)
    # print('R^2 for dp vs S is %0.4f\n'%R)

    axins.plot(ss,dps,'.', color='#1E3B7A', ms=8, clip_on=False)
    axins.plot([0,ss[-1]],[0,ffit(ss[-1])],c='#F26A9D',lw=1.5)
    axins.set_xlabel('S', fontsize=20, labelpad=-17)
    # axins.set_ylabel(r'$\Delta$p', fontsize=20, rotation=0, ha='center', va='center')
    axins.text(-0.15, 0.5, r'$\Delta p_{0,1}$',
        rotation=90, fontsize=20,
        transform=axins.transAxes,
        ha='center', va='center',
    )
    axins.tick_params(direction="in")
    axins.set_xlim(0, ss[-1])
    axins.tick_params(axis='x', pad=10)
    if key == 'HH':
        axins.set_xlim(0, 5e-2)
        axins.set_xticks([0,5e-2])
        axins.set_ylim(0, 1e-2)
        axins.set_yticks([0, 1e-2])
    elif key == 'Lorenz':
        axins.set_xlim(0, 1)
        axins.set_xticks([0,1])
        axins.set_ylim(-2e-3, 0.0)
        axins.set_yticks([-2e-3, 0])
    else:
        axins.set_xlim(0, 2.e-2)
        axins.set_xticks([0,2e-2])
        axins.set_ylim(-2e-2, 0.0)
        axins.set_yticks([-2e-2, 0])

    subfolder = key+'100'
    if key == 'Logistic':
        spk_fname = 'Logp=0.25s=0.0050'
    N = 100
    T = 1e6 if key == 'Lorenz' else 1e7
    estimator = CausalityEstimator(
        root/subfolder, spk_fname, N, delay=delay, T=T, dt=dt,
        n_thread=64, order=order,
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
        ax_col[-1].plot(edges[mask], counts[mask], color=color, lw=5, clip_on=True)
        ax_col[-1].fill_between(edges[mask], 0, counts[mask], color=color, alpha=0.5)
    ax_col[-1].axvline(tmp['th_svm'], ls='-', color='#F26A9D', lw=4)
    if key == 'HH':
        ax_col[-1].set_xlim(-9, -4)
        # format_xticks(ax_col[-1], (-9, -4))
    elif key == 'Lorenz':
        ax_col[-1].set_xlim(-11, -4)
        ax_col[-1].set_xticks([-10, -8, -6, -4])
    elif key == 'Logistic':
        ax_col[-1].set_xlim(-7, -4)
    ax_col[-1].xaxis.set_major_formatter(sci_formatter)
        # format_xticks(ax_col[-1], (-7, -4))
    ax_col[-1].set_ylim(0)
    ax_col[-1].set_xlabel('PTD-TE value', fontsize=26)
    ax_col[-1].set_ylabel('density', fontsize=26)

xx, yy = np.meshgrid(np.arange(4), np.arange(2), indexing='ij')
for tag, x, y in zip('abcdefgh', xx.flatten(), yy.flatten()):
    fig.text(x*0.25, 1.005-y*0.5, tag, fontsize=35, fontweight='bold', va='top')

fig.savefig(root/'fig_nc/pdf'/'fig4.pdf', transparent=True)
#%%
