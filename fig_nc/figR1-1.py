# -*- coding: utf-8 -*-
# Author: Kai Chen

#%%
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

def plot_s_vs_ptdte(data_path, axes, spk_fname, ss, dt, order, delay):
    N = 3
    estimator = CausalityEstimator(
        data_path, spk_fname, N, delay=delay, T=1e7, dt=dt, order=order,
    )

    pdif = []
    dps = []
    for s in ss:
        idx = spk_fname.find('s=')
        if idx != -1:
            estimator.spk_fname = spk_fname[:idx+2] + f"{s:.3f}" + spk_fname[idx+7:]
        data = estimator.fetch_data(new_run=True)
        data01 = data[data['pre_id'].eq(0) * data['post_id'].eq(1)]['TE'].values[0]
        data10 = data[data['pre_id'].eq(1) * data['post_id'].eq(0)]['TE'].values[0]
        data02 = data[data['pre_id'].eq(0) * data['post_id'].eq(2)]['TE'].values[0]
        dps.append(data[data['pre_id'].eq(0) * data['post_id'].eq(1)]['dp1'].values[0])
        pdif.append([data01, data10, data02])
    pdif = np.array(pdif)
    dps = np.array(dps)
    line_settings = [
        {'color':'#1E3B7A', 'ls':'none', 'marker':'o', 'ms':10, 'mec':'none'},
        {'color':'#EE781F', 'ls':'none', 'marker':'*', 'ms':10 },
        {'mec':'#108B96', 'mfc':'none', 'mew':2, 'ls':'none', 'marker':'o', 'ms':12},
    ]
    #108B96
    # CA462F
    for i, (line_setting, label) in enumerate(zip(line_settings, ['X->Y', 'Y->X', 'X->Z'])):
        axes[0].plot(ss, pdif[:,i], **line_setting, clip_on=False)
    ffit = squarefit(ss[:15], pdif[:15,0])
    ss_fit = np.linspace(ss[0], ss[-1], 100)
    axes[0].plot(ss_fit, ffit(ss_fit), '-', color='#F26A9D', lw=4, zorder=10)
    axes[0].set_ylabel('PDIF value', fontsize=26)
    axes[0].set_ylim(0)
    axes[0].ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
    axes[0].tick_params(axis='x', pad=10)  # Increase x-axis tick label padding
    # ax.set_yscale('log')
    

    # s vs dp LARGE
    axes[1].ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)

    ffit = linearfit(ss[:15], dps[:15])
    # print('R^2 for dp vs S is %0.4f\n'%R)

    axes[1].plot(ss,dps,'o', color='#1E3B7A', ms=10, clip_on=False)
    axes[1].plot([0,ss[-1]],[0,ffit(ss[-1])],c='#F26A9D',lw=4, zorder=10)
    axes[1].text(-0.15, 0.5, r'$\Delta p_{0,1}$',
        rotation=90, fontsize=26,
        transform=axes[1].transAxes,
        ha='center', va='center',
    )
    axes[1].set_ylim(0)
    for axi in axes:
        axi.set_xlabel(r'$S$ $(\mathrm{mS}\cdot\mathrm{cm}^{-2})$', fontsize=26, usetex=False)
        axi.tick_params(axis='both', labelsize=22)
        axi.set_xlim(0, 0.1)
        axi.set_xticks([0,0.05,0.1])
        axi.yaxis.get_offset_text().set_fontsize(20)
    return axes

fig, axes = plt.subplots(1,2, figsize=(12,4.5), gridspec_kw=dict(wspace=0.3))
data_path = root / 'raw_data'
ss = np.arange(0.000,0.1,0.003)
spk_fname = 'HHp=0.25s=0.020f=0.080u=0.150'
plot_s_vs_ptdte(
    data_path/'HH3_scan_S', axes, spk_fname, ss, dt=0.5, order=(1,1), delay=3)

for i, tag in enumerate('AB'):
    fig.text(0.08+i*0.44, 0.95, tag, fontsize=35, va='top')

fig.savefig(root/'fig_nc/pdf'/'figR1-1.pdf', transparent=True)
#%%
