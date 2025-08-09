# %%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import smite
import yaml
from utils import get_vfname

with open(Path(__file__).resolve().parent / 'benchmark_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

#%%
dts = [0.2]*4 + [0.01, 0.02] + [1, 1] + [0.01]
sigmas = [4]*4 + [20, 16, 0.1, 0.01, 20]
for dt, sigma in zip(dts, sigmas):
    print(sigma * np.sqrt(dt))
#%%
save_path = root_path / 'results/STE'
save_path.mkdir(parents=True, exist_ok=True)

noise_level = np.arange(4)
m = 5
regen = True
sfxs = [''] #['_s0', '_s1', '_s2', '_s3']
indices = np.load(root_path / 'benchmark' / 'N100' / 'subnet_indices.npy')

# for sfx in zip(sfxs):

#! Calculate STE
# load data
# mask = (1 - np.eye(4)).astype(bool)
for key, val in pm_causal_set.items():
    # if key in recon_list and not regen:
        # print(f'{key} already exists')
        # continue
    # N = 4 #val['N']
    vol_fname = val['path'] / get_vfname(val['spk_fname'])
    vol_data = np.load(vol_fname, mmap_mode='r')
    dt = vol_data[1,0]-vol_data[0,0]
    stride = int(val['dt'] / dt)
    conn_fname = val['path'] / val['conn_file']
    conn = np.load(conn_fname.with_suffix('.npy'))

    for i in range(indices.shape[0]):
        vol_data_ = vol_data[::stride, indices[i]+1]
        ste, cpu_time, wall_time = smite.symbolic_transfer_entropy_matrix(
            vol_data_, m=m, runtime_collect=True)
        xx, yy = np.meshgrid(indices[i], indices[i], indexing='ij')
        # df_list = []
        recon_df = pd.DataFrame({
            'pre_id': xx.flatten(),
            'post_id': yy.flatten(),
            'ste': ste.T.flatten(),
            'log-ste': np.log10(ste.T.flatten()),
            'connection': conn[xx.flatten(), yy.flatten()]})
        recon_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        recon_df.dropna(inplace=True)

        recon_df.attrs['wall_time'] = wall_time
        recon_df.attrs['cpu_time']  = cpu_time
        recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}_{i:d}.pkl')

    # column (row) index representing pre_id (post_id)
    # auc ={'ste': roc_auc_score(recon_df['connection'], recon_df['ste']),
    #       'log-ste': roc_auc_score(recon_df['connection'], recon_df['log-ste']),}
    # auc_list[key] = auc
    # print(f"STE calculation for {key} took {cpu_time:.2f} seconds CPU time",
    #     f" and {wall_time:.2f} seconds wall time.")
# auc_df = pd.DataFrame(auc_list).T
# auc_df.to_pickle(save_path / f'auc_list{sfx:s}.pkl')
# with open(save_path / f'recon_list{sfx:s}.pkl', 'wb') as f:
#     pkl.dump(recon_list, f)
# time_df = pd.DataFrame(estimation_time).T
# time_df.to_pickle(save_path / f'estimation_time{sfx:s}.pkl')
# %%
