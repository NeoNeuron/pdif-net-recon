# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import causal4.utils as c4u
from causal4.myplot import ReconstructionFigureGeneral
from pathlib import Path
root_path = Path(__file__).parents[1]
from figrc import *
import yaml
from sklearn.metrics import roc_auc_score

def get_conn_mat(df, key='connection'):
    conn_mat = np.zeros((int(df['pre_id'].max()+1),int(df['post_id'].max()+1)))
    conn_mat[df['pre_id'], df['post_id']] = df[key]
    return conn_mat
colors = [
    # '#335C8C', '#82b6db', '#cbe1ef', '#df1423', '#ca631c', '#5a3e16', '#f9ba80', 
    # '#2D527C', '#9796C7', '#BEB0D5', '#EBC4CD', '#FDD5A8', '#F3AE8F', '#BE8076',
    '#2D527C', '#ea3323',  '#ff8b00',  '#febb26',  '#1eb253',  '#017cf3', '#9c78fe',
]
with open(Path(__file__).resolve().parents[1] / 'scripts/benchmark/benchmark10_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

subnet_indices = np.load(root_path / 'benchmark/N100/subnet_indices.npy')[0]

save_path = root_path / 'results'
save_path.mkdir(parents=True, exist_ok=True)

net_keys = ['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz', 'Logistic', 'Rcon', 'RNN']
keys = ['PTD-TE', 'STE', 'GLMCC', 'DDC', 'CCM', 'FDCCM', 'SCCM']
labels = {'PTD-TE':'TE', 'STE': 'ste', 'GLMCC': 'glmcc_abs',
          'DDC': 'ddc_abs', 'CCM': 'ccm', 'FDCCM': 'ccm', 'SCCM': 'ccm'}
heatmap_kws = {'cbar': False, 'square': True}
buffer = []
for net_key in net_keys:
    for subnet_toggle, gfname in enumerate([
        lambda x: f'recon_df_noise_0_{x:s}_fullnet.pkl',
        lambda x: f'recon_df_noise_0.3_{x:s}_0.pkl']):
        for key in keys:
            if key == 'GLMCC':
                if net_key in ['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz']:
                    T = pm_causal_set[net_key]['T'] / 1e3
                else:
                    T = pm_causal_set[net_key]['T'] / 1e4
                dfname = gfname(f'{net_key:s}_T={T:.0f}')
            else:
                dfname = gfname(net_key)
            try:
                recon_df = pd.read_pickle(save_path / key / dfname)
                recon, fig_data = c4u._reconstruction_analysis(recon_df, labels[key], 'connection',
                                                               algorithm='EM', hist_type='linear')
                if key == 'PTD-TE' and net_key == 'Lorenz':
                    ReconstructionFigureGeneral(fig_data, causal_hist_with_gt=True)
                buffer.append({
                    'net': net_key, 'causal_measure': key, 'subnet_toggle': subnet_toggle,
                    # 'auc': roc_auc_score(recon_df['connection'], recon_df['log-'+labels[key]]),
                    'auc': roc_auc_score(recon_df['connection'], recon_df[labels[key]]),
                    'gt': get_conn_mat(recon, 'connection'),
                    'recon': get_conn_mat(recon, f'recon-gauss-{labels[key]:s}'),
                    'cpu_time': recon_df.attrs['cpu_time'],
                    'wall_time': recon_df.attrs['wall_time'],
                })
            except FileNotFoundError:
                print(f"File not found: {save_path / key / dfname}")
data = pd.DataFrame(buffer)
# %%
fig = plt.figure(figsize=(16,10))
for i in range(2):
    ax = fig.subplots(8,8, gridspec_kw={
        'left': 0.05+i*0.5, 'right': 0.48+i*0.5, 'top': 0.95, 'bottom': 0.28,
    })
    # plot ground truth
    for axi, net in zip(ax[0], net_keys):
        conn_mat = data[data['net'].eq(net)
                        * data['causal_measure'].eq('PTD-TE')
                        * data['subnet_toggle'].eq(i)
                        ]['gt'].values[0]
        if i == 1:
            conn_mat = conn_mat[subnet_indices][:, subnet_indices]
        print(net, conn_mat.mean(), conn_mat.shape)
        sns.heatmap(conn_mat, ax=axi, **heatmap_kws, cmap='Greens')
        if net == 'Rcon':
            axi.set_title('Rössler', fontweight='bold', fontsize=12)
        elif net == 'Gaussian':
            axi.set_title('LRNN', fontweight='bold', fontsize=12)
        else:
            axi.set_title(net, fontweight='bold', fontsize=12)
    ax[0,0].set_ylabel('ground\ntruth', fontsize=15)

    for ax_row, key in zip(ax[1:], keys):
        # if key not in ['PTD-TE', 'GLMCC']:
        #     continue
        for axi, net in zip(ax_row, net_keys):
            buff = data[data['net'].eq(net)
                        * data['causal_measure'].eq(key)
                        * data['subnet_toggle'].eq(i)
                        ]
            if len(buff) == 0:
                continue
            conn_mat = buff['gt'].values[0]
            recon_mat = buff['recon'].values[0]
            if i == 1:
                conn_mat = conn_mat[subnet_indices][:, subnet_indices]
                recon_mat = recon_mat[subnet_indices][:, subnet_indices]
            sns.heatmap(recon_mat, ax=axi, **heatmap_kws, cmap='Oranges')
            inconsistent_mask = conn_mat != recon_mat  # Find inconsistent blocks
            for y, x in zip(*np.where(inconsistent_mask)):  # Add transparent squares
                axi.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='#00BAFF', lw=1.5, alpha=1.0))
        ax_row[0].set_ylabel(key, rotation=90, fontsize=15)
        # break

    for axi in ax.flatten():
        axi.set_xticks([])
        axi.set_yticks([])
        axi.spines['top'].set_visible(False)
        axi.spines['right'].set_visible(False)
        axi.spines['left'].set_visible(False)
        axi.spines['bottom'].set_visible(False)

    auc_df = data[data['subnet_toggle'].eq(i)].copy()
    # auc_df['auc'] = auc_df.apply(lambda x: np.maximum(x['auc'], 0.510), axis=1)
    axb = fig.subplots(1,1, gridspec_kw={
        'left': 0.05+i*0.5, 'right': 0.48+i*0.5, 'top': 0.20, 'bottom': 0.05,
    })
    # ax = auc_df.loc[['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz', 'Logistic', 'Rcon', 'RNN']].plot.bar(color=colors, width=0.7, ec='w', ax=axb)
    sns.barplot(
        data=auc_df, x='net', y='auc', hue='causal_measure', ec='w',
        order=net_keys, palette=colors, ax=axb)
    axb.set_xticklabels(['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz', 'Logistic', 'Rössler', 'RNN'],)
    axb.tick_params(axis='x', labelsize=13, rotation=0)
    axb.set_ylabel('AUC', fontsize=20)
    axb.set_xlabel('')
    axb.set_ylim(0.3, 1.0)
    axb.set_yticks([0.4, 0.6, 0.8, 1.0])
    axb.legend(loc='lower left', bbox_to_anchor=(0.03, 1.05), fontsize=10, ncols=7, columnspacing=0.5)
    sns.despine(ax=axb)

fig.text(x=0.02, y=0.97, s='a', ha='center', va='center', fontsize=24, fontweight='bold')
fig.text(x=0.02, y=0.24, s='b', ha='center', va='center', fontsize=24, fontweight='bold')
fig.text(x=0.52, y=0.97, s='c', ha='center', va='center', fontsize=24, fontweight='bold')
fig.text(x=0.52, y=0.24, s='d', ha='center', va='center', fontsize=24, fontweight='bold')
fig.savefig(root_path / 'fig_nc/pdf' / f'fig6_new.pdf', transparent=True)

#%%