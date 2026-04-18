# -*- coding: utf-8 -*-
# Author: Kai Chen

#%%
from causal4.Causality import CausalityEstimator
from causal4.utils import match_features, reconstruction_analysis_TE
import causal4.utils as c4u
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

#%
fig, ax = create_fig1x4()

subfolder = '../causal4_data/HH100_main'
spk_fnames = ['HHp=0.25s=0.020f=0.080u=0.150',
              'HHp=0.25s=0.020f=0.080u=0.150',
              'HHp=0.25s=0.020f=0.080u=0.150_noisy']
N = 100
T = 1e7

spks = c4u.load_spike_data(root/subfolder/(spk_fnames[0] + '_spike_train.dat'), xrange=(1000, 1100))
ax[0].plot(spks[:,0], spks[:,1], 'k|', markersize=14, mew=4, clip_on=True, label='raw spikes')
spks = c4u.load_spike_data(root/subfolder/(spk_fnames[2] + '_spike_train.dat'), xrange=(1000, 1100))
ax[0].plot(spks[:,0], spks[:,1], 'r|', markersize=14, mew=3, clip_on=True, label='raw spikes')
ax[0].set_xlim(1000, 1100)
ax[0].set_ylim(0, N)
ax[0].set_xlabel('time (ms)')
ax[0].set_ylabel('neuron ID', labelpad=-5)
# Draw a square box to cover the spikes of bottom 20 neurons
rect = plt.Rectangle((1000, 55), 100, 30, lw=4, edgecolor='limegreen',
                      facecolor='none', alpha=0.7, zorder=10, clip_on=False)
ax[0].add_patch(rect)
# ax[0].legend(fontsize=16, loc=(0.35,0.9), frameon=True)

for i, (axi, spk_fname) in enumerate(zip(ax[1:], spk_fnames)):
    estimator = CausalityEstimator(
        root/subfolder, spk_fname, N, delay=3, T=T, dt=0.5,
        n_thread=64, order=(1,1),
    )
    data = estimator.fetch_data(new_run=True)
    data_matched = match_features(data, N, root/subfolder/'connect_matrix-p=0.250.dat')
    if i == 1:
        mask = (data_matched['pre_id'] >= 55)*(data_matched['pre_id'] < 85)*(data_matched['post_id'] >= 55)*(data_matched['post_id'] < 85)
        data_matched = data_matched[mask]
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=40, hist_range=(-8,-4), algorithm='EM')
    RED, GREEN = '#F49227', '#194955'
    tmp = df_fig.loc['TE']
    for hist_key, color in zip(('hist_conn', 'hist_disconn'), (RED, GREEN)):
        edges = tmp['edges'] + (tmp['edges'][1] - tmp['edges'][0])/2
        counts = tmp[hist_key]
        mask = counts > 0
        axi.plot(edges[mask], counts[mask], color=color, lw=5, clip_on=True)
        axi.fill_between(edges[mask], 0, counts[mask], color=color, alpha=0.5)
    axi.axvline(tmp['th_svm'], ls='-', color='#F26A9D', lw=4)
    add_log_minor_ticks(axi, (-8,-4), where='x')
    axi.tick_params(axis='x', which='major', length=8)
    axi.tick_params(axis='x', which='minor', length=4)
    axi.xaxis.set_major_formatter(sci_formatter)
    axi.set_xlim(-8, -4)
    axi.set_ylim(0)
    axi.set_xlabel('PDIF value')
    axi.set_ylabel('density')

for x, tag in enumerate('abcd'):
    fig.text(x*0.25, 0.995, tag, fontsize=35, fontweight='bold', va='top')

fig.savefig(root/'fig_nc/pdf'/'fig3a-d.pdf', transparent=True)
#%%
