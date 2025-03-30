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

def plot_s_vs_ptdte(data_path, ax, spk_fname, ss, dt, order, delay):
    N = 3
    estimator = CausalityEstimator(
        data_path, spk_fname, N, delay=delay, T=1e7, dt=dt, order=order,
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
        ax.plot(ss, ptdte[:,i], **line_setting, clip_on=False)
    ffit = squarefit(ss, ptdte[:,0])
    ss_fit = np.linspace(ss[0], ss[-1], 100)
    ax.plot(ss_fit, ffit(ss_fit), '-', color='#F26A9D', lw=4, zorder=-1)
    # ax.plot(ss, ptdte, '-o', ms=8, mec='w', clip_on=False)
    # ax.legend([r'$T^\mathrm{PTD}_{X\to Y}$',
    #                   r'$T^\mathrm{PTD}_{Y\to X}$',
    #                   r'$T^\mathrm{PTD}_{X\to Z}$'], fontsize=14, loc='lower right')
    ax.set_xlabel(r'S', fontsize=26)
    ax.set_ylabel('PTD-TE value', fontsize=26)
    ax.set_xlim(0, ss[-1])
    ax.set_ylim(0, ffit(ss[-1])*1.3)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
    ax.tick_params(axis='x', pad=10)  # Increase x-axis tick label padding
    ax.yaxis.get_offset_text().set_x(-0.2)  # Move y-axis offset label to the left

    # s vs dp LARGE
    axins = ax.inset_axes([0.18, 0.65, 0.5, 0.4])
    axins.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)

    ffit = linearfit(ss, dps)
    # print('R^2 for dp vs S is %0.4f\n'%R)

    axins.plot(ss,dps,'.', color='#1E3B7A', ms=8, clip_on=False)
    axins.plot([0,ss[-1]],[0,ffit(ss[-1])],c='#F26A9D',lw=1.5)
    axins.set_xlabel('S', fontsize=20, labelpad=-17)
    # axins.set_ylabel(r'$\Delta$p', fontsize=20, rotation=0, ha='center', va='center')
    axins.text(-0.15, 0.5, r'$\Delta p_{a,b}$',
        rotation=90, fontsize=20,
        transform=axins.transAxes,
        ha='center', va='center',
    )
    axins.tick_params(direction="in")
    axins.set_xlim(0, ss[-1])
    axins.tick_params(axis='x', pad=10)
    axins.set_xlim(0, 5e-2)
    axins.set_xticks([0,5e-2])
    axins.set_ylim(0, 1e-2)
    axins.set_yticks([0, 1e-2])
    return ax


def plot_acf(ax, spk_fname, key, acf_xmax, regen=False):
    zax = zoomedAxes(ax, (-0.1, acf_xmax/2), (-0.1,0.1), [0.4, 0.5, 0.6, 0.5])

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
        ax.set_xlabel('time-lag (ms)', fontsize=26)
        ax.set_ylabel('ACF', fontsize=26, labelpad=-20)
        zax.zax.set_xlabel('time-lag (ms)', fontsize=14, labelpad=-10)
        zax.zax.set_ylabel('ACF', fontsize=14, labelpad=-10)
        for tick in zax.zax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in zax.zax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        if key == 'HH':
            zax.zax.set_xticks([0, 25])
    ax.set_xlim(-0.1, acf_xmax)
    ax.set_ylim(-0.3, 1.00)
    ax.set_yticks([-0.3, 0, 0.5, 1.0])
    # ax.set_title(key+' networks', fontsize=24, pad=22)


#%%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig = plt.figure(figsize=(18,10))
ax = fig.subplots(2, 1,
    gridspec_kw=dict(hspace=0.5,
                     left=0.08, right=0.28,
                     top=0.90, bottom=0.09),)

spk_fname = 'HHp=0.25s=0.020f=0.080u=0.150'
keys = ['chain', 'confounder']
ss = np.arange(0.000,0.051,0.003)
order = (1,1)
delay = 3
plot_s_vs_ptdte(
    root/'HH3_scan_S', ax[0], spk_fname, ss, dt=0.5, order=order, delay=delay)
plot_acf(ax[1], spk_fname, 'HH', acf_xmax=50)

ax = fig.subplots(2, 2,
    gridspec_kw=dict(hspace=0.5, wspace=0.3,
                     left=0.38, right=0.95,
                     top=0.90, bottom=0.09),)
for axi, key in zip(ax.T, keys):
    dt = 0.5
    acf_xmax = 50
    subfolder = f'HH3-{key:s}'
    N = 3
    estimator = CausalityEstimator(
        root/subfolder, spk_fname, N, delay=0, T=1e7, dt=dt, order=(1,1),
    )

    orders = np.arange(1,10).astype(int)
    delays = np.arange(10)
    oo, dd = np.meshgrid(orders, delays)
    ptdte = []
    for o_, d_ in zip(oo.flatten(), dd.flatten()):
        estimator.order = (1,o_)
        data = estimator.fetch_data(d_, new_run=True)
        data = match_features(data, N, root/subfolder/'connect_matrix-p=0.250.npy')
        data01 = data[data['connection'].eq(1)]['TE'].mean()
        # data10 = data[data['pre_id'].eq(1) * data['post_id'].eq(0)]['TE'].values[0]
        data02 = data[data['connection'].eq(0)]['TE'].mean()
        ptdte.append(data01/data02)
    ptdte = np.array(ptdte).reshape(oo.shape)
    pax = axi[1].pcolormesh(oo, dd, np.log10(ptdte), lw=.01, ec='w', vmin=0)#vmax=2)
    cb = fig.colorbar(pax, ax=axi[1], ticks=[0,1,2], orientation='vertical', label='ratio')
    cb.ax.set_yticklabels([r'$10^{0}$', r'$10^{1}$',r'$10^{2}$'])
    axi[1].set_ylabel('delay (ms)', fontsize=26)
    axi[1].set_xlabel(r'order $l$', fontsize=26)
    axi[1].set_xlim(-0.5, 9.5)
    axi[1].set_ylim(0.5, 9.5)
    axi[1].set_yticks([0, 2, 4, 6, 8])
    axi[1].set_xticks([1, 3, 5, 7, 9])
    axi[1].axis('scaled')

    cmap='viridis'
    with open(root / f'data/HH3_{key:s}.pkl', 'rb') as f:
        buff = pkl.load(f)['TE']
        direct=buff['direct']
        indirect=buff['indirect']
        S=buff['S']

    ax_TE = inset_axes(axi[0], width="100%", height="100%",
                       bbox_to_anchor=(.65, .25, .4, .5),
                       bbox_transform=axi[0].transAxes, loc='center')

    ax_TE.scatter(direct, indirect, s=40, c=S, cmap=cmap, vmax=0.03, vmin=0.01, ec='w', lw=0.1)
    ax_TE.ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)

    pval = np.polyfit(direct, indirect, deg=1)
    ax_TE.plot(direct, np.polyval(pval, direct), color='#F26A9D', lw=2, zorder=-1)
    label_fs = 17
    if key == 'confounder':
        ax_TE.set_xlabel(r'$T_{Y\to X}^\mathrm{PTD}\cdot T_{Y\to Z}^\mathrm{PTD}$', fontsize=label_fs, usetex=False)
    elif key == 'chain':
        ax_TE.set_xlabel(r'$T_{X\to Y}^\mathrm{PTD}\cdot T_{Y\to Z}^\mathrm{PTD}$', fontsize=label_fs, usetex=False)
    ax_TE.set_ylabel(r'$T_{X\to Z}^\mathrm{PTD}$', fontsize=label_fs, usetex=False)
    # ax_dp.set_title(r'$R^2=%.3f$'%(Linear_R2(direct, indirect, pval)), fontsize=14)
    ax_TE.set_xlim(-2e-12,3.5e-11)
    ax_TE.xaxis.get_offset_text().set_x(1.3)
    ax_TE.xaxis.get_offset_text().set_fontsize(12)
    ax_TE.yaxis.get_offset_text().set_fontsize(12)

    if key == 'chain':
        axcb = inset_axes(axi[0], width="45%", height="8%",
                        bbox_to_anchor=(0.66, 0.92, 1, 1),
                        bbox_transform=axi[0].transAxes, loc=3)
        gradient = np.atleast_2d(np.linspace(0, 1, 301))
        axcb.imshow(gradient, aspect='auto', cmap=cmap, alpha=1)
        axcb.set_yticks([])
        axcb.set_xticks([0, 150, 300])
        axcb.xaxis.set_ticks_position('top')
        axcb.set_xticklabels(['$0.01$', '$0.02$', '$0.03$'], fontsize=14)
        axcb.xaxis.set_label_position('top')
        axcb.set_xlabel(r'$S$ $(\mathrm{mS}\,\mathrm{cm}^{-2})$', fontsize=16, usetex=False)

    # ax[2].set_title(r'$R^2=%.3f$'%(Linear_R2(S[::2], dp[::2], [pval[0], 0])), fontsize=14)

    with open(root/f'data/HH3_{key:s}.pkl', 'rb') as f:
        buff = pkl.load(f)['dp']
        direct=buff['direct']
        indirect=buff['indirect']
        S=buff['S']

    axi[0].scatter(direct, indirect, s=150, c=S, cmap=cmap, vmax=0.03, vmin=0.01, ec='w', clip_on=False)
    axi[0].ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)

    pval = np.polyfit(direct, indirect, deg=1)
    axi[0].plot(direct, np.polyval(pval, direct), color='#F26A9D', lw=3, zorder=-1)
    label_fs = 25
    if key == 'confounder':
        axi[0].set_xlabel(r'$\Delta p^{Y\to X}_{a,b}\cdot \Delta p^{Y\to Z}_{a,b}$', fontsize=label_fs, usetex=False)
    elif key == 'chain':
        axi[0].set_xlabel(r'$\Delta p^{X\to Y}_{a,b}\cdot \Delta p^{Y\to Z}_{a,b}$', fontsize=label_fs, usetex=False)
    axi[0].set_ylabel(r'$\Delta p^{X\to Z}_{a,b}$', fontsize=label_fs, usetex=False)
    # axi.set_title(r'$R^2=%.3f$'%(Linear_R2(direct, indirect, pval)), fontsize=14)
    axi[0].set_xlim(0.6e-6,2.0e-5)
    axi[0].xaxis.get_offset_text().set_x(1.05)

    axins = inset_axes(axi[0], width="100%", height="100%",
                    bbox_to_anchor=(-.05, .45, .5, .5),
                    bbox_transform=axi[0].transAxes, loc='center')

    G = nx.DiGraph()
    if key == 'chain':
        G.add_edges_from([(1,2), (2,3)])
    elif key == 'confounder':
        G.add_edges_from([(1,2), (1,3)])
    G = nx.relabel_nodes(G, {1:'X', 2:'Y', 3:'Z'})
    pos = {n: coordinate 
        for n, coordinate in zip(G,((0,1),(0,0),(1,1),))}
    
    make_graph_diagram(G, axins, pos, font_size=20, node_size=1400)
    axins.set_xlim(-0.8,1.8)
    axins.set_ylim(-0.8,1.8)

for i, tag in enumerate('ab'):
    fig.text(0.02, 0.48+(1-i)*0.5, tag, fontsize=35, fontweight='bold', va='top')

for i, tag in enumerate('cd'):
    fig.text(0.35, 0.48+(1-i)*0.5, tag, fontsize=35, fontweight='bold', va='top')

fig.savefig(root/'fig_nc/pdf'/'fig3.pdf', transparent=True)
#%%
