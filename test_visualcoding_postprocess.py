# statistical analysis for the causal reconstruction results for all Visual Coding data 
#%%
import pickle
# import h5py
import pandas as pd
# from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
plt.rcParams['font.size']=16
plt.rcParams['axes.labelsize']=16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
# set True to postprocess the shuffle-ablation results instead of the real-data results
shuffle_toggle = True
fig_suffix = "ref=5-gap=250-sfx=0-K=1_5-bin=1.00-delay=0.00"
if shuffle_toggle:
    fig_suffix += "-shuffle"
#%%
df = {'session_id':[],
      'consistency':[],
    #   'consistency_binary':[],
    #   'inconsistent_ratio':[],
      'average_auc':[],
      }
for out_dir in Path('./visualcoding/').iterdir():
    if not out_dir.is_dir():
        continue
    session_id = int(out_dir.stem)
    df['session_id'].append(session_id)
    units = pd.read_pickle(out_dir / f"units.pkl")
    n_unit = len(units)
    # with open(out_dir/f'allen-data-{fig_suffix}-delay=0.00.pkl', 'rb') as f:
    with open(out_dir/f'allen-data-{fig_suffix}.pkl', 'rb') as f:
        data_fig_all = pickle.load(f)

    df['consistency'].append(data_fig_all['consistency']['TE'].min())
    # df['consistency_binary'].append(data_fig_all['consistency_binary']['TE'].min())
    # df['inconsistent_ratio'].append(data_fig_all['drifting_gratings']['hist_inconsist']['TE'].sum()*(np.diff(data_fig_all['drifting_gratings']['edges']['TE'])[0]))
    df['average_auc'].append(
        (data_fig_all['drifting_gratings']['auc_gauss']['TE']
         + data_fig_all['static_gratings']['auc_gauss']['TE']
         + data_fig_all['natural_scenes']['auc_gauss']['TE']
         + data_fig_all['natural_movie']['auc_gauss']['TE']
         )/4)
    
df = pd.DataFrame(df)
df.set_index('session_id', inplace=True)
df.head()
# %%
fig, ax = plt.subplots(2,1,figsize=(4,7))
sns.histplot(df, x='consistency', ax=ax[0], bins=10)
sns.histplot(df, x='average_auc', ax=ax[1], bins=10)
ax[0].set_xlabel('Min. Consistency')
ax[1].set_xlabel('AUC')
ax[1].set_xlim(None,1)
plt.tight_layout()
#%%
fig, ax = plt.subplots(1,1,figsize=(4,3))
sns.histplot(df, x='average_auc', ax=ax, bins=10)
ax.set_xlabel('AUC', fontsize=16)
ax.set_xlim(0.5,1)
ax.set_xticks([0.5,0.75,1])
plt.tight_layout()
# %%
fig, ax = plt.subplots(figsize=(3,6))
for i, col in enumerate(df.columns):
    sns.violinplot(df, x=i, y=col, ax=ax, cut=0)
ax.set_ylabel('values')
ax.set_xticklabels(['Consistency', 'AUC'], rotation=0)
ax.set_ylim(0,1)
ax.grid()
# %%
print(df.iloc[df['consistency'].argmax()])
print(df.iloc[df['average_auc'].argmax()])
# %%