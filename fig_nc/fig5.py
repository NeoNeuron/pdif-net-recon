# %%
from figrc import *
import pandas as pd
import pickle as pkl
import causal4.utils as c4u
from causal4.myplot import ReconstructionFigureGeneral
from pathlib import Path
root_path = Path(__file__).parents[1]
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
with open(Path(__file__).resolve().parents[1] / 'scripts/benchmark/benchmark_causal.yml', 'r') as yamlfile:
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
noise_levels = [0, 0.1, 0.2, 0.3, 0.4]
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
fig = plt.figure(figsize=(20,24))
for i in range(2):
    ax = fig.subplots(8,8, gridspec_kw={
        'left': 0.05+i*0.5, 'right': 0.48+i*0.5, 'top': 0.98, 'bottom': 0.55,
        'hspace': 0.1, 'wspace': 0.1,
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
            axi.set_title('Rössler', fontweight='bold', fontsize=16, pad=5)
        elif net == 'Gaussian':
            axi.set_title('LRNN', fontweight='bold', fontsize=16, pad=5)
        else:
            axi.set_title(net, fontweight='bold', fontsize=16, pad=5)
    ax[0,0].set_ylabel('ground\ntruth', rotation=0, fontsize=15, ha='center', va='center', labelpad=35)

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
        ax_row[0].set_ylabel(key if key != 'PTD-TE' else 'PDIF', rotation=0, fontsize=15, ha='center', va='center', labelpad=35)
        # break

    for axi in ax.flatten():
        axi.set_xticks([])
        axi.set_yticks([])
        axi.spines['top'].set_visible(False)
        axi.spines['right'].set_visible(False)
        axi.spines['left'].set_visible(False)
        axi.spines['bottom'].set_visible(False)

fig.text(x=0.02, y=0.99, s='a', ha='center', va='center', fontsize=24, fontweight='bold')
fig.text(x=0.52, y=0.99, s='b', ha='center', va='center', fontsize=24, fontweight='bold')

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
                        'net': net_key, 'causal_measure': key if key != 'PTD-TE' else 'PDIF',
                        'shuffle_id': shuffle_id, 'noise': noise_level,
                        'auc': np.maximum(roc_auc_score(recon_df['connection'], recon_df[labels[key]]), 0.5),
                        'cpu_time': recon_df.attrs['cpu_time'],
                        'wall_time': recon_df.attrs['wall_time'],
                    })
                except FileNotFoundError:
                    print(f"File not found: {save_path / key / dfname}")
data = pd.DataFrame(buffer)
#%%
axs = fig.subplots(2,4, gridspec_kw={
    'left': 0.06, 'right': 0.98, 'top': 0.50, 'bottom': 0.20,
    'hspace': 0.5, 'wspace': 0.4,
    })
for net_key, ax in zip(net_keys, axs.flatten()):
    ax = sns.lineplot(
        data=data[data['net'].eq(net_key)], x='noise', y='auc',
        hue='causal_measure', palette=colors, ax=ax,
        style='causal_measure', markers=['o']*7, markersize=10,
        dashes=False, err_style='bars', errorbar='se',
        err_kws={'capsize': 3, 'lw':2, 'capthick':3}, legend=True if net_key=='RNN' else False, clip_on=False)
    zorder_map = {"PDIF": 6, "STE": 5, "GLMCC": 4, "DDC": 3, "CCM": 2, "FDCCM": 1, "SCCM": 0}
    # ax.set_yscale('log')
    for line, poly, label in zip(ax.lines, ax.collections, data[data['net'].eq(net_key)]['causal_measure'].unique()):
        line.set_zorder(zorder_map[label])
        poly.set_zorder(zorder_map[label])
    ax.set_title(net_key, fontsize=25, fontweight='bold', pad=15)
    ax.set_xticks(np.arange(5)*0.1)
    ax.set_ylim(0.48, 1.0)
    ax.set_yticks([0.5, 0.75, 1.0], ['0.5', '0.75', '1.0'])
    ax.set_xlabel(r'noise level ($\sigma$)', fontsize=20)
    ax.set_ylabel('AUC', fontsize=20)
    sns.despine(ax=ax, trim=True, offset=5)
    ax.tick_params(axis='both', labelsize=16)
leg = axs[-1,-1].legend(loc=(0.5,-1.4),fontsize=20)

for i, letter in enumerate('cdefghij'):
    fig.text(x=-0.2, y=1.1, s=letter, ha='center', va='center', fontsize=26, fontweight='bold', transform=axs[i//4, i%4].transAxes)


ax = fig.subplots(1,1, gridspec_kw={
    'left': 0.06, 'right': 0.88, 'top': 0.16, 'bottom': 0.04,})
sns.barplot(data=data, x='net', y='cpu_time', hue='causal_measure', palette=colors, ax=ax,
            errorbar='se', capsize=0.4, legend=False, fill=True, saturation=1.0, err_kws={'lw':1.5, 'color':'#AAAAAA'})
ax.tick_params(axis='x', labelsize=22, rotation=0)
ax.set_ylabel('CPU time (seconds)', fontsize=18)
ax.set_xlabel('', fontsize=24)
ax.set_yscale('log')
ax.set_ylim(1e1, 1e6)
ax.tick_params(axis='y', labelsize=16)
fig.text(x=-0.041, y=1.1, s='k', ha='center', va='center', fontsize=26, fontweight='bold', transform=ax.transAxes)
fig.savefig(root_path / 'fig_nc/pdf' / f'fig5.pdf', transparent=True)
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