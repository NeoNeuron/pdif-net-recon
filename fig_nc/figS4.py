#%%
from figrc import *
import pickle
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize']  = 26
plt.rcParams['axes.titlesize']  = 26
import numpy as np
from pathlib import Path
data_path = Path(__file__).resolve().parents[1] / 'data'
with open(data_path/'spk_HH_sync_f=0.080_u=0.150_DSra=0.50.pkl', 'rb') as file:
    spk_data = pickle.load(file)
xlim = (800, 1800)
spk_raw = spk_data['raw']
spk_filtered = spk_data['filtered']
spk_raw = spk_raw[(spk_raw[:,0]>xlim[0])*(spk_raw[:,0]<xlim[1]), :]
spk_filtered = spk_filtered[(spk_filtered[:,0]>xlim[0])*(spk_filtered[:,0]<xlim[1]), :]
mask = np.in1d(spk_raw[:,0], spk_filtered[:,0])

fig, ax = plt.subplots(
    2,1,figsize=(22,5), sharex=True,
    gridspec_kw=dict(hspace=0.25, left=0.05, right=0.50, top=0.90, bottom=0.06))
ax[0].plot(spk_raw[~mask,0], spk_raw[~mask,1],'|',mec='red', mfc='none', ms=5, label='raw')
ax[0].plot(spk_filtered[:,0], spk_filtered[:,1], '|', ms=4, mec=GREEN, mfc='none',
         label='after downsampling')
ax[0].set_xlim(xlim)
ax[0].set_ylim(0,100)
ax[0].set_yticks([1,50,100])
ax[0].tick_params(axis='both', labelsize=22)
# ax[0].set_xlabel('Time (ms)')
# ax.legend()
ax[1].plot(spk_filtered[:,0], spk_filtered[:,1], '|', ms=4, mec=GREEN, mfc='none',
         label='after downsampling')
ax[1].set_xlim(xlim)
ax[1].set_ylim(0,100)
ax[1].set_yticks([1,50,100])
ax[1].set_xlabel('Time (ms)')
ax[1].tick_params(axis='both', labelsize=22)
[axi.set_ylabel('neuron ID') for axi in ax]

import pandas as pd
gs = fig.add_gridspec(1,2, left=0.55, right=0.98, top=0.90, bottom=0.06,)
ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])]

datafiles = [
    'fig_HH_sync_f=0.080_u=0.150_DSra=0.50.pkl',
    'fig_PHH_DSra=0.50.pkl',
    ]

for idx, f in enumerate(datafiles):

    df = pd.DataFrame(pd.read_pickle(data_path/f))
    data = df.loc['TE']
    # TE histogram
    # mask = np.ones_like(data['hist_conn'], dtype=bool)
    mask = data['hist_conn']>0
    ax[idx].plot(data['edges'][mask], data['hist_conn'][mask], color=ORANGE, lw=5, label='PDIF with A_{ij}=1')
    ax[idx].fill_between(data['edges'][mask], 0, data['hist_conn'][mask], color=ORANGE, alpha=0.5)
    # mask = np.ones_like(data['hist_disconn'], dtype=bool)
    mask = data['hist_disconn']>0
    ax[idx].plot(data['edges'][mask], data['hist_disconn'][mask], color=GREEN, lw=5, label='PDIF with A_{ij}=0')
    ax[idx].fill_between(data['edges'][mask], 0, data['hist_disconn'][mask], color=GREEN, alpha=0.5)
    ymax = np.hstack((data['hist_conn'], data['hist_disconn'])).max()
    ax[idx].set_ylim(0)
    ax[idx].set_xlabel('PDIF value')
    ax[idx].set_ylabel('density')
    # print(f"{conn_name_:15s} recon acc : {data['acc_gauss']*100:6.3f} %")
    ax[idx].axvline(data['kmean_th'], ymax=ymax/ax[0].get_ylim()[1], color='#F26A9D',lw=4, label='Threshold')
    if idx == 0:
        ax[idx].set_xlim(-7,-4)
        ax[idx].set_xticks([-6, -5, -4])
    else:
        ax[idx].set_xlim(-8,-4)
        ax[idx].set_xticks([-7, -6, -5, -4])
    ax[idx].xaxis.set_major_formatter(sci_formatter)
    ax[idx].tick_params(axis='x', which='major', length=6)
    ax[idx].tick_params(axis='x', which='minor', length=3)
    ax[idx].tick_params(axis='both', labelsize=22)
add_log_minor_ticks(ax[0], (-7, -4), where='x')
add_log_minor_ticks(ax[1], (-8, -4), where='x')

fig.text(0.01, 0.995, 'a', fontsize=35, fontweight='bold', va='top')
fig.text(0.52, 0.995, 'b', fontsize=35, fontweight='bold', va='top')
fig.text(0.76, 0.995, 'c', fontsize=35, fontweight='bold', va='top')

fig.savefig(root/'fig_nc/pdf/figS4.pdf', dpi=300, bbox_inches='tight', transparent=True)
# %%
