#%%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
import crossmap_indices as cm
import yaml
from utils import get_vfname
import pandas as pd

def run_CCM(ccm_type:str, data:np.ndarray, tau:int):
    if ccm_type == 'CCM':
        method = cm.ClassicalCCM(
            emb_params=(5, 1),
            min_T=1000,
            n_samples=500,
            n_T=30,
            rho_tol=0.025,
            exp_fit=True
        )
        cmat = method.network_inference(data[::tau])
    elif ccm_type == 'FDCCM':
        method = cm.FrequencyCCM(
            emb_params=(5, tau),
            min_T=1000,
            n_samples=500,
            n_T=30,
            rho_tol=0.025,
            exp_fit=True,
            truncate_tol=1e-2
        )
        cmat = method.network_inference(data)
    elif ccm_type == 'SCCM':
        method = cm.SymbolicCCM(
            emb_params=(5, 1),
            min_T=1000,
            n_samples=500,
            n_T=30,
            rho_tol=0.025,
            exp_fit=True
        )
        cmat = method.network_inference(data[::tau])
    else:
        raise ValueError(f"Unknown CCM type: {ccm_type}")
    return cmat, method.cpu_time, method.wall_time


with open(Path(__file__).resolve().parent / 'benchmark_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

indices = np.load(root_path / 'benchmark' / 'N100' / 'subnet_indices.npy')

L_dict = {
    'HHconEE': int(2e6),  # total 2e8
    'HHconEI': int(2e6),  # total 2e8
    'HHEE':    int(1e6),  # total 5e7
    'HHEI':    int(1e6),  # total 5e7
    'Lorenz':  int(1e6),  # total 1e8
    'Logistic':int(1e6),  # total 1e8
    'Rcon':    int(1e7),  # total 1e9
    'RNN':     int(1e6),  # total 1e8
}


# 为部分方法进行下采样 (CCM 和 SCCM) / 作为FDCCM的必要参数
tau_dict = {
    'HHconEE': 30,
    'HHconEI': 30,
    'HHEE':    10,
    'HHEI':    10,
    'Lorenz':  10,
    'Logistic': 1,
    'Rcon':   100,
    'RNN':      2,
}


def core_function(key, val, shuffle_id, ccm_type, noise_level=None):

    L = L_dict[key]  # 读取的样本量
    tau = tau_dict[key]

    conn_fname = val['path'] / val['conn_file']
    conn = np.load(conn_fname)

    data = np.load(val['path'] / get_vfname(val['spk_fname']), mmap_mode='r')[:L, 1+indices[shuffle_id]]
    if noise_level is None:
        preprocessing = lambda x: x
    else:
        sigma = data[:10000].flatten().std()*noise_level
        preprocessing = lambda x: x+np.random.randn(*x.shape)*sigma

    save_path = root_path / 'results' / ccm_type
    save_path.mkdir(parents=True, exist_ok=True)

    data = preprocessing(data)
    cmat, cpu_time, wall_time = run_CCM(ccm_type, data, tau)

    xx, yy = np.meshgrid(indices[shuffle_id], indices[shuffle_id], indexing='ij')
    recon_df = pd.DataFrame({
        'pre_id': xx.flatten(),
        'post_id': yy.flatten(),
        'ccm': cmat.flatten(),
        'log-ccm': np.log10(cmat.flatten()),
        'connection': conn[xx.flatten(), yy.flatten()]})

    recon_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    recon_df.dropna(inplace=True)

    recon_df.attrs['wall_time'] = wall_time
    recon_df.attrs['cpu_time']  = cpu_time
    if noise_level is None:
        recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}_{shuffle_id:d}.pkl')
    else:
        recon_df.to_pickle(save_path / f'recon_df_noise_{noise_level:.1f}_{key:s}_{shuffle_id:d}.pkl')

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ccm', type=str, choices=['CCM', 'FDCCM', 'SCCM'], default='CCM')
parser.add_argument('--key', type=str)
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--noise_level', type=float, default=None)
args = parser.parse_args()

core_function(args.key, pm_causal_set[args.key], args.idx, args.ccm, args.noise_level)
#%%

# ccm_types = ['CCM', 'FDCCM', 'SCCM']

# for noise_level in [0.1, 0.2, 0.3, 0.4]:
#     for key, val in pm_causal_set.items():
#         for shuffle_id in range(indices.shape[0]):
#             for ccm_type in ccm_types:
#                 core_function(key, val, shuffle_id, ccm_type, noise_level)
