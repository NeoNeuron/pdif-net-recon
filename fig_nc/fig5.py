# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import causal4.utils as c4u
from pathlib import Path
root_path = Path(__file__).parents[1]
from figrc import *

def get_conn_mat(df, key='connection'):
    conn_mat = np.zeros((int(df['pre_id'].max()+1),int(df['post_id'].max()+1)))
    conn_mat[df['pre_id'], df['post_id']] = df[key]
    return conn_mat
colors = [
    # '#335C8C', '#82b6db', '#cbe1ef', '#df1423', '#ca631c', '#5a3e16', '#f9ba80', 
    '#2D527C', '#9796C7', '#BEB0D5', '#EBC4CD', '#FDD5A8', '#F3AE8F', '#BE8076',
]
selected_nets = ['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz', 'Logistic', 'Rcon', 'Gaussian']
heatmap_kws = {'cbar': False, 'square': True}
# %%
keys = ['PTD-TE', 'STE', 'GLMCC', 'DDC', 'CCM', 'FDCCM', 'SCCM']
labels = ['TE', 'ste', 'glmcc_abs', 'ddc_abs', 'CCM', 'FDCCM', 'SCCM'] 
fig = plt.figure(figsize=(16,14))
for i, (subfolder, dfname) in enumerate(
    zip(['results/N10', 'results/N10_subnet_noisy'], ['recon_list.pkl', 'recon_list_noisy3.pkl'])):
    # subfolder = 'results/N10'
    # dfname = 'recon_list.pkl'
    causal_data = {}
    for key in keys:
        with open(root_path / subfolder / key / dfname, 'rb') as f:
            causal_data[key] = pkl.load(f)

    recon_data = {}
    for key, label in zip(keys, labels):
        recon_data[key] = {}
        for net in selected_nets:
            if net not in causal_data[key]:
                continue
            val = causal_data[key][net].copy()
            data_recon = c4u._reconstruction_analysis(val, label, 'connection', algorithm='EM', hist_type='linear')
            recon_data[key][net] = data_recon[0]
    ax = fig.subplots(8,8, gridspec_kw={
        'left': 0.05+i*0.5, 'right': 0.48+i*0.5, 'top': 0.95, 'bottom': 0.40,
    })
    # plot ground truth
    for axi, net in zip(ax[0], selected_nets):
        conn_mat = get_conn_mat(recon_data['PTD-TE'][net], key='connection')
        sns.heatmap(conn_mat, ax=axi, **heatmap_kws, cmap='Greens')
        if net == 'Rcon':
            axi.set_title('Rossler', fontweight='bold', fontsize=12)
        else:
            axi.set_title(net, fontweight='bold', fontsize=12)
    ax[0,0].set_ylabel('ground truth')

    for ax_row, key, label in zip(ax[1:], keys, labels):
        for axi, net in zip(ax_row, selected_nets):
            if net not in recon_data[key]:
                continue
            val = recon_data[key][net].copy()
            conn_mat = get_conn_mat(val, key='connection')
            recon_mat = get_conn_mat(val, key=f'recon-gauss-{label:s}')
            sns.heatmap(recon_mat, ax=axi, **heatmap_kws, cmap='Oranges')
            inconsistent_mask = conn_mat != recon_mat  # Find inconsistent blocks
            for y, x in zip(*np.where(inconsistent_mask)):  # Add transparent squares
                axi.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='#00BAFF', lw=1.5, alpha=1.0))
        ax_row[0].set_ylabel(key)
        # break

    for axi in ax.flatten():
        axi.set_xticks([])
        axi.set_yticks([])
        axi.spines['top'].set_visible(False)
        axi.spines['right'].set_visible(False)
        axi.spines['left'].set_visible(False)
        axi.spines['bottom'].set_visible(False)

    auc_list = {}
    for key, label in zip(keys, labels):
        auc_list[key] = pd.read_pickle(root_path / subfolder / key / dfname.replace('recon_list', 'auc_list'))[label]
        # print(key, auc_list[key])
    auc_df = pd.DataFrame(auc_list)
    auc_df = np.maximum(auc_df, 0.510)
    axb = fig.subplots(1,1, gridspec_kw={
        'left': 0.05+i*0.5, 'right': 0.48+i*0.5, 'top': 0.35, 'bottom': 0.24,
    })
    ax = auc_df.loc[['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz', 'Logistic', 'Rcon', 'Gaussian']].plot.bar(color=colors, width=0.7, ec='w', ax=axb)
    axb.set_xticklabels(['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz', 'Logistic', 'Rossler', 'Gaussian'],)
    axb.tick_params(axis='x', labelsize=12, rotation=0)
    axb.set_ylabel('AUC', fontsize=20)
    axb.set_ylim(0.5, 1.0)
    axb.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axb.legend(loc='lower left', bbox_to_anchor=(-0.05, 1.05), fontsize=10, ncols=7)
    # plt.tight_layout()
    sns.despine(ax=axb)

fig.text(x=0.02, y=0.97, s='a', ha='center', va='center', fontsize=24, fontweight='bold')
fig.text(x=0.52, y=0.97, s='b', ha='center', va='center', fontsize=24, fontweight='bold')
fig.text(x=0.02, y=0.20, s='c', ha='center', va='center', fontsize=24, fontweight='bold')
# sns.despine(offset=0, trim=True)
# plt.savefig(root_path / 'fig_nc/pdf' / f'comp_auc{noise_level:s}.pdf', transparent=True)

#%
noise_levels = ['', '_noisy1', '_noisy2', '_noisy3']
dfs = []
for noise_level in noise_levels:
    subfolder = 'results/N10_subnet' if noise_level == '' else 'results/N10_subnet_noisy'
    auc_list = {}
    dfname = f'auc_list{noise_level:s}.pkl'

    for key, label in zip(keys, labels):
        auc_list[key] = pd.read_pickle(root_path / subfolder / key / dfname)[label]
        # print(key, auc_list[key])
    auc_df = pd.DataFrame(auc_list)
    auc_df['noise_level'] = 0 if noise_level == '' else int(noise_level[-1])
    dfs.append(auc_df)
dfs = pd.concat(dfs)
df = dfs.reset_index().rename(columns={'index':'network'})

ax = fig.subplots(1,8, sharey=True, gridspec_kw={
    'left': 0.05, 'right': 0.98, 'top': 0.17, 'bottom': 0.05,
    })
for net, axi in zip(['HHEE', 'HHEI', 'HHconEE', 'HHconEI',
            'Lorenz', 'Logistic', 'Rcon', 'Gaussian'], ax.flatten()):
    for i, key in enumerate(keys):
        tmp = df[df['network'].eq(net)]
        lw = 1.5 if i == 0 else 2
        ms = 10 if i == 0 else 12
        zorder = 10 if i == 0 else None
        buff = np.maximum(tmp[key], 0.5)
        axi.plot(tmp['noise_level'], buff,
                  '-o', ms=ms, mec='w', mew=1.5, color=colors[i],
                  label=key, lw=lw, zorder=zorder)
    axi.set_xticks([0,1,2,3])
    axi.set_yticks([0.4, 0.6, 0.8, 1.0])
    if net == 'Rcon':
        axi.set_title('Rossler', fontweight='bold', fontsize=20)
    else:
        axi.set_title(net, fontweight='bold', fontsize=20)
[axi.set_xlabel('noise level', fontsize=18) for axi in ax]
ax[0].set_ylabel('AUC', fontsize=20)
# ax[0,-1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.00), fontsize=16)
[sns.despine(offset=1, trim=True, ax=axi) for axi in ax]

fig.savefig(root_path / 'fig_nc/pdf' / 'fig5.pdf', transparent=True)
#%%