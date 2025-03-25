# %%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import smite
import causal4.myplot as myplot
import time

import yaml
import matplotlib.pyplot as plt
plt.rcParams['font.size']=16
import seaborn as sns
import pickle as pkl

def get_vfname(fname: str, sfx:str=''):
    if key == 'Gaussian':
        fname = fname.replace('th=0.020','')
    if fname.startswith('Lp') or fname.startswith('Lcon') or fname.startswith('Rcon'):
        fname = fname + '_x'
    else:
        fname = fname + '_voltage'
    return fname + sfx + '.npy'

#%%
yml_names = ['benchmark10_causal.yml',
             'benchmark10_subnet_causal.yml',
             'benchmark10_subnet_causal.yml',
             'benchmark10_subnet_causal.yml',
             'benchmark10_subnet_causal.yml',
             'benchmark10_subnet_causal.yml',
            ]
sfxs = ['', '', '_noisy1', '_noisy2', '_noisy3', '_noisy4']
save_folder_names = [
    'N10',
    'N10_subnet',
    'N10_subnet_noisy',
    'N10_subnet_noisy',
    'N10_subnet_noisy',
    'N10_subnet_noisy',
    ]
m = 5
regen = False
for yml_name, sfx, sfname in zip(yml_names, sfxs, save_folder_names):
    
    with open(yml_name, 'r') as yamlfile:
        pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
    for key in pm_causal_set.keys():
        pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

    save_path = root_path / 'results' / sfname / 'STE'
    save_path.mkdir(parents=True, exist_ok=True)

    if (save_path / f'auc_list{sfx:s}.pkl').exists():
        auc_list = pd.read_pickle(save_path / f'auc_list{sfx:s}.pkl').T.to_dict()
    else:
        auc_list = {}
    if (save_path / f'recon_list{sfx:s}.pkl').exists():
        with open(save_path / f'recon_list{sfx:s}.pkl', 'rb') as f:
            recon_list = pkl.load(f)
    else:
        recon_list = {}
    if (save_path / f'estimation_time{sfx:s}.pkl').exists():
        estimation_time = pd.read_pickle(save_path / f'estimation_time{sfx:s}.pkl').T.to_dict()
    else:
        estimation_time = {}

    #! Calculate STE
    # load data
    mask = (1 - np.eye(10)).astype(bool)
    for key, val in pm_causal_set.items():
        if key in recon_list and not regen:
            print(f'{key} already exists')
            continue
        N = val['N']
        vol_fname = val['path'] / get_vfname(val['spk_fname'], sfx=sfx)
        # vol_data = np.load(vol_fname, mmap_mode='r')
        vol_data = np.load(vol_fname)
        dt = vol_data[1,0]-vol_data[0,0]
        stride = int(val['dt'] / dt)
        t0_wall = time.time()
        t0_cpu = time.process_time()
        ste = smite.symbolic_transfer_entropy_matrix(vol_data[0::stride,1:], m=m, n_jobs=10)
        t1_cpu = time.process_time()
        t1_wall = time.time()
        conn_fname = val['path'] / val['conn_file']
        xx, yy = np.meshgrid(np.arange(N), np.arange(N))
        if conn_fname.suffix == '.dat': 
            conn = np.fromfile(conn_fname, dtype=float).reshape(N,N).T
        elif conn_fname.suffix == '.npy':
            conn = np.load(conn_fname).T
        else:
            raise ValueError('Unknown file format')
        df_list = []
        recon_df = pd.DataFrame({'pre_id': xx[mask],
                           'post_id': yy[mask],
                           'ste': ste[mask],
                           'log-ste': np.log10(ste[mask]),
                           'connection': conn[mask]})
        recon_list[key] = recon_df
        # column (row) index representing pre_id (post_id)
        auc ={'ste': roc_auc_score(recon_df['connection'], recon_df['ste']),
              'log-ste': roc_auc_score(recon_df['connection'], recon_df['log-ste']),}
        auc_list[key] = auc
        estimation_time[key] = {'wall':t1_wall-t0_wall, 'cpu':t1_cpu-t0_cpu}
        print(key, dt, f"{t1_wall-t0_wall:.2f}", auc)
    
    auc_df = pd.DataFrame(auc_list).T
    auc_df.to_pickle(save_path / f'auc_list{sfx:s}.pkl')
    with open(save_path / f'recon_list{sfx:s}.pkl', 'wb') as f:
        pkl.dump(recon_list, f)
    time_df = pd.DataFrame(estimation_time).T
    time_df.to_pickle(save_path / f'estimation_time{sfx:s}.pkl')
# %%
