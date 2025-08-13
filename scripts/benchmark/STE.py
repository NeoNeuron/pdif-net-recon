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

save_path = root_path / 'results/STE'
save_path.mkdir(parents=True, exist_ok=True)

m = 5
regen = True
indices = np.load(root_path / 'benchmark' / 'N100' / 'subnet_indices.npy')

#! Calculate STE
# for key, val in pm_causal_set.items():
def core_function(key, val, shuffle_id, noise_level=None):
    vol_fname = val['path'] / get_vfname(val['spk_fname'])
    vol_data = np.load(vol_fname, mmap_mode='r')
    dt = vol_data[1,0]-vol_data[0,0]
    stride = int(val['dt'] / dt)
    conn_fname = val['path'] / val['conn_file']
    conn = np.load(conn_fname.with_suffix('.npy'))

    if noise_level is None:
        preprocessing = lambda x: x
    else:
        sigma = vol_data[:10000, :11].flatten().std()*noise_level
        preprocessing = lambda x: x+np.random.randn(*x.shape)*sigma

    vol_data_ = preprocessing(vol_data[::stride, indices[shuffle_id]+1])
    ste, cpu_time, wall_time = smite.symbolic_transfer_entropy_matrix(
        vol_data_, m=m, runtime_collect=True)
    xx, yy = np.meshgrid(indices[shuffle_id], indices[shuffle_id], indexing='ij')
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
    if noise_level is not None:
        recon_df.to_pickle(save_path / f'recon_df_noise_{noise_level:.1f}_{key:s}_{shuffle_id:d}.pkl')
    else:
        recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}_{shuffle_id:d}.pkl')

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--key', type=str)
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--noise_level', type=float, default=None)
args = parser.parse_args()

core_function(args.key, pm_causal_set[args.key], args.idx, args.noise_level)
#%%

# for noise_level in [0.1, 0.2, 0.3, 0.4]:
#     for key, val in pm_causal_set.items():
#         for i in range(indices.shape[0]):
#             core_function(key, val, noise_level, i)
#%%