# %%
from figrc import *
import pandas as pd
import pickle as pkl
from causal4.utils import EM
from pathlib import Path
root_path = Path(__file__).parents[1]
import yaml
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score,
)

def get_conn_mat(df, key='connection'):
    conn_mat = np.zeros((int(df['pre_id'].max()+1),int(df['post_id'].max()+1)))
    conn_mat[df['pre_id'], df['post_id']] = df[key]
    return conn_mat
colors = [
    # '#335C8C', '#82b6db', '#cbe1ef', '#df1423', '#ca631c', '#5a3e16', '#f9ba80', 
    # '#2D527C', '#9796C7', '#BEB0D5', '#EBC4CD', '#FDD5A8', '#F3AE8F', '#BE8076',
    '#2D527C', '#ea3323',  '#ff8b00',  '#febb26',  '#1eb253',  '#017cf3', '#9c78fe',
]
with open(Path(__file__).resolve().parents[1] / 'scripts/benchmark/benchmark_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

subnet_indices = np.load(root_path / 'benchmark/N100/subnet_indices.npy')[0]

save_path = root_path / 'results'
save_path.mkdir(parents=True, exist_ok=True)

net_keys = ['HHEE', 'HHEI', 'HHconEE', 'HHconEI', 'Lorenz', 'Logistic', 'Rcon', 'RNN']
keys = ['PDIF', 'STE', 'GLMCC', 'DDC', 'CCM', 'FDCCM', 'SCCM']
labels = {'PDIF':'TE', 'STE': 'ste', 'GLMCC': 'glmcc_abs',
          'DDC': 'ddc_abs', 'CCM': 'ccm', 'FDCCM': 'ccm', 'SCCM': 'ccm'}
T_length = {
    'HHEE': 2e5, 'HHEI': 2e5, 'HHconEE': 1e5, 'HHconEI': 1e5,
    'Lorenz': 1e4, 'Logistic': 1e6,
    'Rcon': 1e5, 'RNN': 1e6,
}
noise_levels = [0, 0.1, 0.2, 0.3, 0.4]
heatmap_kws = {'cbar': False, 'square': True}

buffer = [] 
for net_key in net_keys:
    for key in keys:
        for noise_level in noise_levels:
            for shuffle_id in range(10):
                dfname = 'recon_df_noise'
                if noise_level == 0:
                    dfname += f'_0_{net_key:s}_T={T_length[net_key]:.2e}_{shuffle_id:d}.pkl'
                else:
                    dfname += f'_{noise_level:.1f}_{net_key:s}_T={T_length[net_key]:.2e}_{shuffle_id:d}.pkl'
                try:
                    recon_df = pd.read_pickle(save_path / key / dfname)
                    y_true = recon_df['connection'].to_numpy()
                    y_metric = recon_df[labels[key]].to_numpy()
                    _, th_, _, _ = EM(y_metric)
                    y_pred = (y_metric > th_).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else np.nan
                    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    fdr = fp / (tp + fp) if (tp + fp) > 0 else np.nan
                    f1 = (2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0 else np.nan)
                    buffer.append({
                        'net': net_key, 'causal_measure': key,
                        'shuffle_id': shuffle_id, 'noise': noise_level,
                        'accuracy': accuracy, 'precision': precision, 'recall': recall,
                        'false discovery rate': fdr, 'F1 score': f1,
                        'AUC': np.maximum(roc_auc_score(recon_df['connection'], recon_df[labels[key]]), 0.5),
                        'PR-AUC': np.maximum(average_precision_score(recon_df['connection'], recon_df[labels[key]]), 0.5),
                        'cpu_time': recon_df.attrs['cpu_time'],
                        'wall_time': recon_df.attrs['wall_time'],
                    })
                except FileNotFoundError:
                    print(f"File not found: {save_path / key / dfname}")
data = pd.DataFrame(buffer)
#%%
fig = plt.figure(figsize=(36,33))
axs = fig.subplots(7,8, gridspec_kw={
    'left': 0.04, 'right': 0.98, 'top': 0.92, 'bottom': 0.08,
    'hspace': 0.5, 'wspace': 0.4,
    })
ylims = [
    (0.5,1.0), (0.2, 1.0), (0.1, 1.0), (0.2, 1.0), (0.0, 0.9), (0.2, 1.0), (0.45, 1.0),]
for axi, ylim, metric_key in zip(axs, ylims, ['AUC', 'accuracy', 'precision', 'recall', 'false discovery rate', 'F1 score', 'PR-AUC']):
    for net_key, ax in zip(net_keys, axi):
        ax = sns.lineplot(
            data=data[data['net'].eq(net_key)], x='noise', y=metric_key,
            hue='causal_measure', palette=colors, ax=ax,
            style='causal_measure', markers=['o']*7, markersize=10,
            dashes=False, err_style='bars', errorbar='se',
            err_kws={'capsize': 3, 'lw':2, 'capthick':3}, legend=True if net_key=='HHEE' and metric_key=='AUC' else False, clip_on=False)
        zorder_map = {"PDIF": 6, "STE": 5, "GLMCC": 4, "DDC": 3, "CCM": 2, "FDCCM": 1, "SCCM": 0}
        # ax.set_yscale('log')
        for line, poly, label in zip(ax.lines, ax.collections, data[data['net'].eq(net_key)]['causal_measure'].unique()):
            line.set_zorder(zorder_map[label])
            poly.set_zorder(zorder_map[label])
        if metric_key == 'AUC':
            ax.set_title(net_key, fontsize=35, fontweight='bold', pad=15)
        ax.set_xticks(np.arange(5)*0.1)
        ax.set_ylim(ylim)
        # ax.set_yticks([0.5, 0.75, 1.0], ['0.5', '0.75', '1.0'])
        ax.set_xlabel(r'noise level ($\sigma$)', fontsize=20)
        ax.set_ylabel(metric_key, fontsize=26)
        sns.despine(ax=ax, trim=True, offset=5)
        ax.tick_params(axis='both', labelsize=16)
leg = axs[0,0].legend(loc=(2.2,1.25), fontsize=30, ncol=7, title='causal measure', title_fontsize=30, frameon=True)

for axi, letter in zip(axs[:,0], 'ABCDEFG'):
    fig.text(x=-0.3, y=1.10, s=letter, ha='center', va='center', fontsize=35, transform=axi.transAxes)

fig.savefig(root_path / 'fig_nc/pdf' / f'figR4-7.pdf', transparent=True)
#%%
# fig, ax = plt.subplots(1,1, gridspec_kw={
#     'left': 0.06, 'right': 0.98, 'top': 0.96, 'bottom': 0.10,}, figsize=(18,4))
# sns.barplot(data=data, x='net', y='wall_time', hue='causal_measure', palette=colors, ax=ax,
#             errorbar='se', capsize=0.4, legend=False, fill=True, saturation=1.0, err_kws={'lw':1.5, 'color':'#AAAAAA'})
# ax.tick_params(axis='x', labelsize=24, rotation=0)
# ax.set_ylabel('Wall time (seconds)', fontsize=24)
# ax.set_xlabel('', fontsize=24)
# ax.set_yscale('log')
# ax.set_ylim(1e0)
# ax.tick_params(axis='y', labelsize=16)
# fig.savefig(root_path / 'fig_nc/pdf' / f'figS5.pdf', transparent=True)


#%%