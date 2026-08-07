# -*- coding: utf-8 -*-
# Author: Kai Chen

#%%
from causal4.Causality import CausalityEstimator
from causal4.utils import match_features, reconstruction_analysis_TE
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
root = root / 'data'
#%

fig, ax = create_fig1x4()
ax = ax.reshape(2,2)

spk_fnames = ['PC_DCN_100_s0.1/PC_DCN_100', 'CA1_PYPV/CA1_PYPV_24']
conn_fnames = ['connection100.npy', 'connection24.npy']
for axi, spk_fname, conn_fname in zip(ax, spk_fnames, conn_fnames):
    axi[0].axis('off')
    subfolder = (root / spk_fname).parent
    N = 100
    T = 1e6
    estimator = CausalityEstimator(
        subfolder, spk_fname.split('/')[-1], N, delay=3, T=T, dt=0.5,
        n_thread=128, order=(1,5),
    )
    data = estimator.fetch_data(new_run=True)
    data_matched = match_features(data, N, root/subfolder/conn_fname)
    if 'PC_DCN' in spk_fname:
        data_matched = data_matched[data_matched['pre_id'].lt(50) * data_matched['post_id'].gt(50)]
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=60, hist_range=None, algorithm='EM')
    ORANGE, GREEN = '#F49227', '#194955'
    tmp = df_fig.loc['TE']
    for hist_key, color in zip(('hist_conn', 'hist_disconn'), (ORANGE, GREEN)):
        edges = tmp['edges'] + (tmp['edges'][1] - tmp['edges'][0])/2
        counts = tmp[hist_key]
        mask = counts > 0
        axi[1].plot(edges[mask], counts[mask], color=color, lw=5, clip_on=True)
        axi[1].fill_between(edges[mask], 0, counts[mask], color=color, alpha=0.5)
    axi[1].axvline(tmp['th_svm'], ls='-', color='#F26A9D', lw=4)
    axi[1].set_ylim(0)
    axi[1].set_yticks([0,0.5,1.0])
    axi[1].set_xlabel('PDIF value')
    axi[1].set_ylabel('density')

ax[0,1].xaxis.set_major_formatter(sci_formatter)
add_log_minor_ticks(ax[0,1], (-7,-3), where='x')
ax[0,1].set_xlim(-7, -3)
ax[0,1].tick_params(axis='x', which='major', length=8)
ax[0,1].tick_params(axis='x', which='minor', length=4)

ax[1,1].xaxis.set_major_formatter(sci_formatter)
add_log_minor_ticks(ax[1,1], (-8,-2), where='x')
ax[1,1].set_xlim(-8, -2)
ax[1,1].tick_params(axis='x', which='major', length=8)
ax[1,1].tick_params(axis='x', which='minor', length=4)

for x, tag in enumerate('EFGH'):
    fig.text(x*0.25, 0.995, tag, fontsize=35, va='top')

fig.savefig(root.parent/'fig_nc/pdf'/'fig3e-h.pdf', transparent=True)
#%%
