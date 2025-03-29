# statistical analysis for the causal reconstruction results for all Visual Coding data 
#%%
import pickle
# import h5py
import numpy as np
import pandas as pd
# from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
plt.rcParams['font.size']=15
plt.rcParams['axes.labelsize']=15
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
from pathlib import Path
root_path = Path(__file__).parents[1]

import warnings
warnings.filterwarnings('ignore')
#%%
session_id=[]
    #   'consistency':[],
    #   'consistency_binary':[],
    #   'inconsistent_ratio':[],
average_auc = []
for out_dir in (root_path/ 'visualcoding').iterdir():
    if not out_dir.is_dir():
        continue
    session_id.append(int(out_dir.stem))
    units = pd.read_pickle(out_dir / f"units.pkl")
    n_unit = len(units)
    # with open(out_dir/'allen-data-ref=5-gap=250-sfx=0-K=1_5-bin=1.00-delay=0.00.pkl', 'rb') as f:
    with open(out_dir/'allen-data-ref=5-gap=250-sfx=0-K=1_5-bin=1.00.pkl', 'rb') as f:
        data_fig_all = pickle.load(f)

    # df['consistency'].append(data_fig_all['consistency']['TE'].min())
    # df['consistency_binary'].append(data_fig_all['consistency_binary']['TE'].min())
    # df['inconsistent_ratio'].append(data_fig_all['drifting_gratings']['hist_inconsist']['TE'].sum()*(np.diff(data_fig_all['drifting_gratings']['edges']['TE'])[0]))
    average_auc.append(
        {'drifting_gratings': data_fig_all['drifting_gratings']['auc_gauss']['TE'],
         'static_gratings': data_fig_all['static_gratings']['auc_gauss']['TE'],
         'natural_scenes': data_fig_all['natural_scenes']['auc_gauss']['TE'],
         'natural_movie': data_fig_all['natural_movie']['auc_gauss']['TE'],
         })

for out_dir in (root_path / 'visualbehavior').iterdir():
    if not out_dir.is_dir():
        continue
    session_id.append(int(out_dir.stem))
    units = pd.read_pickle(out_dir / f"units.pkl")
    n_unit = len(units)
    # with open(out_dir/'allen-data-ref=5-gap=250-sfx=250-K=1_5-bin=1.00-delay=0.00.pkl', 'rb') as f:
    with open(out_dir/'allen-data-ref=5-gap=250-sfx=250-K=1_5-bin=1.00.pkl', 'rb') as f:
        data_fig_all = pickle.load(f)

    # df['consistency'].append(data_fig_all['consistency']['TE'][1,0])
    # df['consistency_binary'].append(data_fig_all['consistency_binary']['TE'][1,0])
    # df['inconsistent_ratio'].append(data_fig_all['active']['hist_inconsist']['TE'].sum()*(np.diff(data_fig_all['active']['edges']['TE'])[0]))
    average_auc.append(
        {'active': data_fig_all['active']['auc_gauss']['TE'],
         'passive': data_fig_all['passive']['auc_gauss']['TE']})
    
df_auc = pd.DataFrame(average_auc, index=session_id)
# df.set_index('session_id', inplace=True)
df_auc.head()
# %%
fig, ax = plt.subplots(2,3,figsize=(10,5))
for label, axi in zip(df_auc.columns, ax.flatten()):
    sns.histplot(df_auc[label], ax=axi, bins=20, binrange=(0.5,1.0), kde=True, ec='w')
    axi.set_xlabel('AUC')
    axi.set_ylabel('session counts')
    axi.set_xlim(0.5,1)
    axi.set_xticks([0.5,0.6, 0.7, 0.8, 0.9,1])
        # arr_image = plt.imread('../'+stimulus_names_plot[i].split('-')[0]+'.png', format='png')
    arr_image = plt.imread(root_path / 'figure' / (label+'.png'), format='png')
    axins = axi.inset_axes([0.05, 0.5, 0.5, 0.4], transform=axi.transAxes)

    axins.imshow(arr_image)
    axins.axis('off')
    if '_' in label:
        axi.set_title(label.replace('_', ' '))
    else:
        axi.set_title(label+' behavior')
plt.tight_layout()
for label, axi in zip('abcdef', ax.flatten()):
    axi.text(-0.08, 1.4, label, transform=axi.transAxes, fontsize=25, va='top', ha='right', weight='bold')

fig.savefig(root_path / 'fig_nc/pdf' / 'fig5.pdf')
# sns.histplot(df_auc, x='consistency', ax=ax[0], bins=10)
# sns.histplot(df_auc, x='average_auc', ax=ax[1], bins=10)
# ax[0].set_xlabel('Min. Consistency')
# ax[1].set_xlabel('AUC')
# ax[1].set_xlim(None,1)

# ax.set_ylabel('values')
# ax.set_xticklabels(['consistency', 'consistency(bin)', 'inconsistency\nratio', 'auc(TE)'], rotation=0)
# ax.set_ylim(0,1)
# ax.grid()
# sns.boxplot(df, x=2, y=)