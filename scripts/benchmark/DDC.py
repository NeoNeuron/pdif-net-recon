# %%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
# from sklearn.metrics import roc_auc_score, roc_curve
from causal4.ddc import DDC, c_sensitivity, DDC_long
import time

import yaml
from utils import get_vfname
#%%
with open(Path(__file__).resolve().parent / 'benchmark_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']
# sfxs = ['', '', '_noisy1', '_noisy2', '_noisy3', '_noisy4']
# save_folder_names = [
#     'N10',
#     'N10_subnet',
#     'N10_subnet_noisy',
#     'N10_subnet_noisy',
#     'N10_subnet_noisy',
#     'N10_subnet_noisy',
#     ]
rerun=False
# for yml_name, sfx, sfname in zip(yml_names, sfxs, save_folder_names):

save_path = root_path / 'results/DDC'
save_path.mkdir(parents=True, exist_ok=True)

indices = np.load(root_path / 'benchmark' / 'N100' / 'subnet_indices.npy')

#! Calculate DDC
# mask = (1 - np.eye(10)).astype(bool)
for key, val in pm_causal_set.items():
    # if key in recon_list and not rerun:
    #     print(f'{key} already exists')
    #     continue
    N = val['N']
    vol_fname = val['path'] / get_vfname(val['spk_fname'])
    vol_data = np.load(vol_fname, mmap_mode='r')
    dt = vol_data[1,0]-vol_data[0,0]
    # L = int(vol_data.shape[0]/4)
    conn_fname = val['path'] / val['conn_file']
    conn = np.load(conn_fname.with_suffix('.npy'))
        
    for i in range(indices.shape[0]):
        t0_wall = time.time()
        t0_cpu = time.process_time()
        # ddc = DDC(vol_data[:,1:].T, dt)
        ddc = DDC_long(vol_fname, N=N, indices=indices[i], n_blocks=20)
        t1_cpu = time.process_time()
        t1_wall = time.time()
        xx, yy = np.meshgrid(indices[i], indices[i], indexing='ij')
        recon_df = pd.DataFrame({
            'pre_id': xx.flatten(),
            'post_id': yy.flatten(),
            'ddc': ddc.T.flatten(),
            'ddc_abs': np.abs(ddc.T.flatten()),
            'log-ddc_abs': np.log10(np.abs(ddc.flatten())),
            'connection': conn[xx.flatten(), yy.flatten()]})
        # recon_list[key] = recon_df
        # column (row) index representing pre_id (post_id)
        # auc ={'ddc':roc_auc_score(recon_df['connection'], recon_df['ddc']),
        #         'ddc_abs':roc_auc_score(recon_df['connection'], recon_df['ddc_abs'])}
        # auc_list[key] = auc
        # estimation_time[key] = {'wall':t1_wall-t0_wall, 'cpu':t1_cpu-t0_cpu}
        # print(key, dt, f"{t1_wall-t0_wall:.2f}", auc)
        wall_time, cpu_time = t1_wall-t0_wall, t1_cpu-t0_cpu
        recon_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        recon_df.dropna(inplace=True)

        recon_df.attrs['wall_time'] = wall_time
        recon_df.attrs['cpu_time']  = cpu_time
        recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}_{i:d}.pkl')

        # auc_df = pd.DataFrame(auc_list).T
        # auc_df.to_pickle(save_path / f'auc_list{sfx:s}.pkl')
        # with open(save_path / f'recon_list{sfx:s}.pkl', 'wb') as f:
        #     pkl.dump(recon_list, f)
        # time_df = pd.DataFrame(estimation_time).T
        # time_df.to_pickle(save_path / f'estimation_time{sfx:s}.pkl')
# %%

