# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
import seaborn as sns
import pandas as pd
from pathlib import Path
root_path = Path(__file__).parents[1]
import yaml
from sklearn.metrics import roc_auc_score
from figrc import *

def get_conn_mat(df, key='connection'):
    conn_mat = np.zeros((int(df['pre_id'].max()+1),int(df['post_id'].max()+1)))
    conn_mat[df['pre_id'], df['post_id']] = df[key]
    return conn_mat
heatmap_kws = {'cbar': False, 'square': True}
with open(Path(__file__).resolve().parents[1] / 'scripts/benchmark/benchmark_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

save_path = root_path / 'results'
save_path.mkdir(parents=True, exist_ok=True)

net_keys = ['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz', 'Logistic', 'Rcon', 'RNN']
keys = ['PTD-TE', 'STE', 'GLMCC', 'DDC', 'CCM', 'FDCCM', 'SCCM']
labels = {'PTD-TE':'TE', 'STE': 'ste', 'GLMCC': 'glmcc_abs',
          'DDC': 'ddc_abs', 'CCM': 'ccm', 'FDCCM': 'ccm', 'SCCM': 'ccm'}
colors = {'PTD-TE':'#2D527C', 'STE':'#ea3323',  'GLMCC':'#ff8b00',  'DDC':'#febb26',  'CCM':'#1eb253',  'FDCCM':'#017cf3', 'SCCM':'#9c78fe'}
noise_levels = [0, 0.1, 0.2, 0.3, 0.4]
buffer = [] 
for net_key in net_keys:
    for key in keys:
        for noise_level in noise_levels:
            for shuffle_id in range(10):
                dfname = 'recon_df_noise'
                if noise_level == 0:
                    if key == 'GLMCC':
                        if net_key in ['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz']:
                            T = pm_causal_set[net_key]['T'] / 1e3
                        else:
                            T = pm_causal_set[net_key]['T'] / 1e4
                        dfname += f'_0_{net_key:s}_T={T:.0f}_{shuffle_id:d}.pkl'
                    else:
                        dfname += f'_0_{net_key:s}_{shuffle_id:d}.pkl'
                else:
                    if key == 'GLMCC':
                        if net_key in ['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz']:
                            T = pm_causal_set[net_key]['T'] / 1e3
                        else:
                            T = pm_causal_set[net_key]['T'] / 1e4
                        dfname += f'_{noise_level:.1f}_{net_key:s}_T={T:.0f}_{shuffle_id:d}.pkl'
                    else:
                        dfname += f'_{noise_level:.1f}_{net_key:s}_{shuffle_id:d}.pkl'
                try:
                    recon_df = pd.read_pickle(save_path / key / dfname)
                    buffer.append({
                        'net': net_key, 'causal_measure': key,
                        'shuffle_id': shuffle_id, 'noise': noise_level,
                        'auc': np.maximum(roc_auc_score(recon_df['connection'], recon_df[labels[key]]), 0.5),
                        'cpu_time': recon_df.attrs['cpu_time'],
                        'wall_time': recon_df.attrs['wall_time'],
                    })
                except FileNotFoundError:
                    print(f"File not found: {save_path / key / dfname}")
data = pd.DataFrame(buffer)
#%%
fig, axs = plt.subplots(2,4, figsize=(20,12), gridspec_kw={
    'left': 0.06, 'right': 0.88, 'top': 0.95, 'bottom': 0.35,
    'hspace': 0.5, 'wspace': 0.4,
    })
for net_key, ax in zip(net_keys, axs.flatten()):
    ax = sns.lineplot(
        data=data[data['net'].eq(net_key)], x='noise', y='auc',
        hue='causal_measure', palette=colors, ax=ax,
        style='causal_measure', markers=['o']*7, markersize=10,
        dashes=False, err_style='bars', errorbar='se',
        err_kws={'capsize': 3, 'lw':2, 'capthick':3}, legend=True if net_key=='RNN' else False, clip_on=False)
    zorder_map = {"PTD-TE": 6, "STE": 5, "GLMCC": 4, "DDC": 3, "CCM": 2, "FDCCM": 1, "SCCM": 0}
    # ax.set_yscale('log')
    for line, poly, label in zip(ax.lines, ax.collections, data[data['net'].eq(net_key)]['causal_measure'].unique()):
        line.set_zorder(zorder_map[label])
        poly.set_zorder(zorder_map[label])
    ax.set_title(net_key, fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(np.arange(5)*0.1)
    ax.set_ylim(0.48, 1.0)
    ax.set_yticks([0.5, 0.75, 1.0], ['0.5', '0.75', '1.0'])
    ax.set_xlabel('Noise level', fontsize=20)
    ax.set_ylabel('AUC', fontsize=20)
    sns.despine(ax=ax, trim=True, offset=5)
    ax.tick_params(axis='both', labelsize=16)
leg = axs[-1,-1].legend(loc=(1.1,0.8),fontsize=20)

for i, letter in enumerate('abcdefgh'):
    fig.text(x=-0.2, y=1.1, s=letter, ha='center', va='center', fontsize=26, fontweight='bold', transform=axs[i//4, i%4].transAxes)


ax = fig.subplots(1,1, gridspec_kw={
    'left': 0.06, 'right': 0.88, 'top': 0.26, 'bottom': 0.08,})
sns.barplot(data=data, x='net', y='cpu_time', hue='causal_measure', palette=colors, ax=ax,
            errorbar='se', capsize=0.4, legend=False, ec='w', lw=2)
ax.tick_params(axis='x', labelsize=18, rotation=0)
ax.set_ylabel('CPU time (seconds)', fontsize=18)
ax.set_xlabel('Noise level (in units of standard deviation)', fontsize=24)
ax.set_yscale('log')
ax.set_ylim(1e1, 1e6)
ax.tick_params(axis='y', labelsize=16)
fig.text(x=-0.041, y=1.2, s='i', ha='center', va='center', fontsize=26, fontweight='bold', transform=ax.transAxes)
fig.savefig(root_path / 'fig_nc/pdf' / f'fig7.pdf', transparent=True)
#%%