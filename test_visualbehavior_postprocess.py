# statistical analysis for the causal reconstruction results for all Visual Behavior data 
#%%
import pickle
# import h5py
import numpy as np
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
key_map = {'TE': 'TE', 'MI': 'sum(MI)', 'CC': 'sum(CC2)', 'GC': 'GC'}
#%%
df = {'session_id':[],
      'consistency':[],
      'consistency_binary':[],
      'inconsistent_ratio':[],
      'average_auc':[],
      }
for out_dir in Path('./visualbehavior/').iterdir():
    if not out_dir.is_dir():
        continue
    session_id = int(out_dir.stem)
    df['session_id'].append(session_id)
    units = pd.read_pickle(out_dir / f"units.pkl")
    n_unit = len(units)
    with open(out_dir/'allen_data.pkl', 'rb') as f:
        data_fig_all = pickle.load(f)

    df['consistency'].append(data_fig_all['consistency']['TE'][1,0])
    df['consistency_binary'].append(data_fig_all['consistency_binary']['TE'][1,0])
    df['inconsistent_ratio'].append(data_fig_all['active']['hist_inconsist']['TE'].sum()*(np.diff(data_fig_all['active']['edges']['TE'])[0]))
    df['average_auc'].append(
        (data_fig_all['active']['auc_gauss']['TE']
         + data_fig_all['passive']['auc_gauss']['TE'])/2)
    
df = pd.DataFrame(df)
df.set_index('session_id', inplace=True)
df.head()
# %%
fig, ax = plt.subplots(figsize=(12,6))
for i, col in enumerate(df.columns):
    sns.violinplot(df, x=i, y=col, ax=ax, cut=0)
ax.set_ylabel('values')
ax.set_xticklabels(['consistency', 'consistency(bin)', 'inconsistency\nratio', 'auc'], rotation=0)
ax.set_ylim(0,1)
ax.grid()
# sns.boxplot(df, x=2, y=)
# %%
df.iloc[df['consistency'].argmax()]
# %%
df.iloc[df['average_auc'].argmax()]

# %%
