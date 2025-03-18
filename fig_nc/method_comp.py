# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl
import causal4.utils as c4u

from pathlib import Path
root_path = Path(__file__).parents[1]

def get_conn_mat(df, key='connection'):
    conn_mat = np.zeros((int(df['pre_id'].max()+1),int(df['post_id'].max()+1)))
    conn_mat[df['pre_id'], df['post_id']] = df[key]
    return conn_mat
# %%
with open(root_path / 'results' / 'N10' / 'DDC' / 'auc_list.pkl', 'rb') as f:
    auc_list = pkl.load(f)
# %%
list(auc_list.keys())
#%%
with open(root_path / 'results' / 'N10' / 'STE' / 'estimation_time.pkl', 'rb') as f:
    est_time = pkl.load(f)
est_time
# %%
# %%
subfolder = 'results/N10'
dfname = 'recon_list.pkl'
with open(root_path / subfolder / 'PTD-TE' / dfname, 'rb') as f:
    recon_list_ptdte = pkl.load(f)
with open(root_path / subfolder / 'DDC' / dfname, 'rb') as f:
    recon_list_ddc = pkl.load(f)
with open(root_path / subfolder / 'STE' / dfname, 'rb') as f:
    recon_list_ste = pkl.load(f)
with open(root_path / subfolder / 'GLMCC' / dfname, 'rb') as f:
    recon_list_glmcc = pkl.load(f)
with open(root_path / subfolder / 'CCM' / dfname, 'rb') as f:
    recon_list_ccm = pkl.load(f)
with open(root_path / subfolder / 'FDCCM' / dfname, 'rb') as f:
    recon_list_fdccm = pkl.load(f)
with open(root_path / subfolder / 'SCCM' / dfname, 'rb') as f:
    recon_list_sccm = pkl.load(f)
labels = ['TE', 'ste', 'glmcc_abs', 'ddc_abs', 'CCM', 'FDCCM', 'SCCM'] 
#%%
val = recon_list_ptdte['Lorenz']
data_recon = c4u._reconstruction_analysis(val, 'TE', 'connection', algorithm='EM', hist_type='linear')
plt.plot(data_recon[1]['edges'], data_recon[1]['hist_disconn'], label='dis-conn')
plt.plot(data_recon[1]['edges'], data_recon[1]['hist_conn'], label='conn')
plt.axvline(data_recon[1]['th_gauss'], ls='--', color='k')
#%%
data_recon[0]['TE']
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(data_recon[0]['TE'].to_numpy().reshape(-1,1))
#%%
plt.scatter(np.arange(len(data_recon[0]['TE'])), data_recon[0]['TE'], c=kmeans.labels_)
kmeans.cluster_centers_
# plt.axhline(-6.5, ls='--', color='k')
# plt.axhline(-5.45, ls='--', color='k')
# kmeans.labels_
#%%
#%%
# recon_dict = {}
# for i, key in enumerate(est_time.T.keys()):
#     recon_dict[key] = recon_list_ste[i]
#     print(i, key)

# with open(root_path / subfolder / 'STE' / dfname, 'wb') as f:
#     pkl.dump(recon_dict, f)
#%%
heatmap_kws = {'cbar': False, 'square': True, 'cmap':'Oranges'}
fig, ax = plt.subplots(8,10,figsize=(10,8))
# plot ground truth
for axi, (key, val) in zip(ax[0], recon_list_ptdte.items()):
    conn_mat = get_conn_mat(val, key='connection')
    sns.heatmap(conn_mat, ax=axi, **heatmap_kws)
    axi.set_title(key, fontweight='bold', fontsize=12)
ax[0,0].set_ylabel('GT')

for ax_row, recon_data, label in zip(ax[1:], [
    recon_list_ptdte, recon_list_ste, recon_list_glmcc, recon_list_ddc, recon_list_ccm,
    recon_list_fdccm, recon_list_sccm], labels):
    for axi, (key, val) in zip(ax_row, recon_data.items()):
        if val is None:
            continue
        data_recon = c4u._reconstruction_analysis(val, label, 'connection', algorithm='EM', hist_type='linear')
        conn_mat = get_conn_mat(data_recon[0], key='connection')
        recon_mat = get_conn_mat(data_recon[0], key=f'recon-gauss-{label:s}')
        sns.heatmap(recon_mat, ax=axi, **heatmap_kws)
        inconsistent_mask = conn_mat != recon_mat  # Find inconsistent blocks
        for y, x in zip(*np.where(inconsistent_mask)):  # Add transparent squares
            axi.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='#00BAFF', lw=1.5, alpha=1.0))
    ax_row[0].set_ylabel(label)

for axi in ax.flatten():
    axi.set_xticks([])
    axi.set_yticks([])
# %%
val = recon_list_glmcc['HHEE'].copy()
c4u.reconstruction_analysis_TE()
data_recon = c4u._reconstruction_analysis(val, 'glmcc_abs', 'connection', hist_type='linear')
buff = data_recon[1]
plt.plot(buff['edges'], buff['hist_disconn'])
plt.plot(buff['edges'], buff['hist_conn'])
plt.axvline(buff['th_gauss'], ls='--', color='k')
# %%
subfolder = 'results/N10_subnet'
noise_level = ''
dfname = f'auc_list{noise_level:s}.pkl'
auc_list = {}
keys = ['PTD-TE', 'STE', 'GLMCC', 'DDC', 'CCM', 'FDCCM', 'SCCM']
labels = ['TE', 'ste', 'glmcc_abs', 'ddc_abs', 'CCM', 'FDCCM', 'SCCM'] 

for key, label in zip(keys, labels):
    auc_list[key] = pd.read_pickle(root_path / subfolder / key / dfname)[label]
    # print(key, auc_list[key])
auc_df = pd.DataFrame(auc_list)
# %
colors = [
    '#335C8C', '#82b6db', '#cbe1ef', '#df1423', '#ca631c', '#5a3e16', '#f9ba80', 
]
ax = auc_df.plot.bar(color=colors, figsize=(12,3), width=0.7)
ax.set_xticklabels(auc_df.index, rotation=0, fontweight='bold')
ax.set_ylabel('AUC', fontsize=20)
ax.set_ylim(0.5, 1.0)
plt.tight_layout()
sns.despine(offset=10, trim=True)
plt.savefig(root_path / 'fig_nc/pdf' / f'comp_auc{noise_level:s}.pdf', transparent=True)

#%%
keys = ['PTD-TE', 'STE', 'GLMCC', 'DDC', 'CCM', 'FDCCM', 'SCCM']
labels = ['TE', 'ste', 'glmcc_abs', 'ddc_abs', 'CCM', 'FDCCM', 'SCCM'] 
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
#%%
fig, ax = plt.subplots(2,5, figsize=(15,6), sharex=True, sharey=True)
for net, axi in zip(['HHEE', 'HHII', 'HHEI', 'HHconEE', 'HHconII','HHconEI',
            'Lorenz', 'Lcon', 'Logistic', 'Gaussian'], ax.flatten()):
    for i, key in enumerate(keys):
        tmp = df[df['network'].eq(net)]
        lw = 2 if i == 0 else 1.5
        ms = 14 if i == 0 else 10
        axi.plot(tmp['noise_level'], tmp[key], '-o', ms=ms, mec='w', mew=0.5, color=colors[i], label=key, lw=lw)
    axi.set_xticks([0,1,2,3])
    axi.set_yticks([0.4, 0.6, 0.8, 1.0])
    axi.set_title(net, fontweight='bold', fontsize=20)
ax[0,0].legend()
sns.despine(offset=1, trim=True)
plt.tight_layout()
fig.savefig(root_path / 'fig_nc/pdf' / 'comp_auc_all.pdf', transparent=True)
#%%
colors = [
    '#335C8C', '#82b6db', '#cbe1ef', '#df1423', '#ca631c', '#5a3e16', '#f9ba80', 
]
ax = auc_df.plot.bar(color=colors, figsize=(12,3), width=0.7)
ax.set_xticklabels(auc_df.index, rotation=0, fontweight='bold')
ax.set_ylabel('AUC', fontsize=20)
ax.set_ylim(0.5, 1.0)
plt.tight_layout()
sns.despine(offset=10, trim=True)
plt.savefig(root_path / 'fig_nc/pdf' / f'comp_auc{noise_level:s}.pdf', transparent=True)
