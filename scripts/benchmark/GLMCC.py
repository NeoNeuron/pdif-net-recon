# %%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd

import yaml
from utils import get_spk_fname
from glmcc.Est_Data import Est_Data

with open(Path(__file__).resolve().parent / 'benchmark_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']
indices = np.load(root_path / 'benchmark' / 'N100' / 'subnet_indices.npy')

# Load binarization thresholds and refs from YAML file
with open(root_path / 'scripts/benchmark' / 'binarization.yaml', 'r') as binarization_file:
    binarization_cfg = yaml.load(binarization_file, Loader=yaml.FullLoader)
refs = binarization_cfg.get('refractory', {})
thresholds = binarization_cfg.get('threshold', {})
dt = binarization_cfg.get('dt',{})

regen=True
# for yml_name, sfx, sfname in zip(yml_names, sfxs, save_folder_names):

save_path = root_path / 'results' / 'GLMCC'
save_path.mkdir(parents=True, exist_ok=True)

def core_function(key, val, shuffle_id, noise_level=None):
    #! Calculate GLMCC
    N = val['N']
    if noise_level is None:
        spk_fname = get_spk_fname(val['spk_fname'])
    else:
        spk_fname = get_spk_fname(
            val['spk_fname'], f'_noisy{noise_level:.1f}', th=thresholds[key], ref=refs[key])
        # val['spk_fname'], sfx=sfx, th=thresholds[key], ref=refs[key])
    conn_fname = val['path'] / val['conn_file']
    conn = np.load(conn_fname)
    if key in ['HHEE', 'HHEI', 'HHconEE', 'HHconEI']:
        T = val['T'] / 1e4
    elif key in ['Gaussian', 'Rcon', 'Logistic', 'RNN', 'Lorenz']:
        T = val['T'] / 1e6
    W, cpu_time, wall_time = Est_Data(
        val['path'], spk_fname, N=N, T=T, indices=indices[shuffle_id],
        outfile_sfx=f'{shuffle_id:d}', DELTA=dt[key], WIN=dt[key]*50,
        n_jobs=shuffle_id.shape[1])
    try:
        GLMCC = np.load(val['path'] / f"W_GLM_{T:.0f}-{spk_fname:s}_{shuffle_id:d}.npy")
        xx, yy = np.meshgrid(indices[shuffle_id], indices[shuffle_id], indexing='ij')
        GLMCC[GLMCC==0] = 1e-12
        recon_df = pd.DataFrame({
            'pre_id': xx.flatten(),
            'post_id': yy.flatten(),
            'glmcc': GLMCC.flatten(),
            'glmcc_abs': np.abs(GLMCC.flatten()),
            'log-glmcc_abs': np.log10(np.abs(GLMCC.flatten())),
            'connection': conn[xx.flatten(), yy.flatten()]})

        recon_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        recon_df.dropna(inplace=True)

        recon_df.attrs['wall_time'] = wall_time
        recon_df.attrs['cpu_time']  = cpu_time
        if noise_level is None:
            recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}_T={T:.0f}_{shuffle_id:d}.pkl')
        else:
            recon_df.to_pickle(save_path / f'recon_df_noise_{noise_level:.1f}_{key:s}_T={T:.0f}_{shuffle_id:d}.pkl')
        # print(recon_df.head(100))

    except Exception as e:
        print(e)
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