# %%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from causal4.Causality import CausalityEstimator
from causal4.utils import match_features
import causal4.myplot as myplot
import time
import pickle as pkl

import yaml
import matplotlib.pyplot as plt
plt.rcParams['font.size']=16
import seaborn as sns
import subprocess

def get_vfname(fname: str, sfx:str=None):
    if key == 'Gaussian':
        fname = fname.replace('th=0.020','')
    if fname.startswith('Lp') or fname.startswith('Lcon'):
        fname = fname + '_x'
    else:
        fname = fname + '_voltage'
    if sfx is not None:
        fname += sfx
    return fname + '.npy'

def get_spk_fname(fname: str, sfx:str=None, th: float=None, ref: float=None):
    if sfx is None:
        return fname
    else:
        if key == 'Gaussian':
            fname = fname.replace('th=0.020','')
        fname += sfx + f'_th={th:.2f}ref={ref:.2f}'
        return fname

def run_shell_command(command: str):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with error: {result.stderr}")
    return result.stdout

# Example usage:
# output = run_shell_command("ls -l")
# print(output)
#%%
yml_names = ['benchmark10_causal.yml',
             'benchmark10_subnet_causal.yml',
             'benchmark10_subnet_causal.yml',
             'benchmark10_subnet_causal.yml',
             'benchmark10_subnet_causal.yml',
             'benchmark10_subnet_causal.yml',
            ]
sfxs = [None, None, '_noisy1', '_noisy2', '_noisy3', '_noisy4']
save_folder_names = [
    'N10',
    'N10_subnet',
    'N10_subnet_noisy',
    'N10_subnet_noisy',
    'N10_subnet_noisy',
    'N10_subnet_noisy',
    ]
# Load binarization thresholds and refs from YAML file
with open(root_path / 'scripts/benchmark' / 'binarization.yaml', 'r') as binarization_file:
    binarization_cfg = yaml.load(binarization_file, Loader=yaml.FullLoader)
refs = binarization_cfg.get('refractory', {})
thresholds = binarization_cfg.get('threshold', {})

for yml_name, sfx, sfname in zip(yml_names, sfxs, save_folder_names):
    
    with open(yml_name, 'r') as yamlfile:
        pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
    for key in pm_causal_set.keys():
        pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

    recon_list = {}
    auc_list = {}
    estimation_time = {}
    #! Calculate PTD-TE
    for key, val in pm_causal_set.items():
        val = val.copy()
        val['spk_fname'] = get_spk_fname(
            val['spk_fname'], sfx=sfx, th=thresholds[key], ref=refs[key])
        estimator = CausalityEstimator(**val, n_thread=10)
        _, text = estimator._run_estimation(regen=True, verbose=False, return_log=True)
        # print(text.splitlines()[-1].split())
        wall_time = float(text.splitlines()[-1].split()[3])
        cpu_time = float(text.splitlines()[-1].split()[7])
        print(key, f"Elapsed cpu time: {cpu_time:.2f} s")
        print(key, f"Elapsed time: {wall_time:.2f} s")
        data = estimator.fetch_data(new_run=True)
        data = data[['pre_id', 'post_id', 'TE', 'Delta_p']]
        # Plot distribution of TE values in log-scale
        recon_df = match_features(
            data, N=val['N'], conn_file=val['path']/val['conn_file'])

        recon_list[key] = recon_df
        # column (row) index representing pre_id (post_id)
        auc =[roc_auc_score(recon_df['connection'], recon_df['TE']),]
        auc_list[key] = auc
        estimation_time[key] = [wall_time, cpu_time]
        print(key, f"{wall_time:.2f}", auc)
    
    save_path = root_path / 'results' / sfname / 'PTD-TE'
    save_path.mkdir(parents=True, exist_ok=True)
    sfx = '' if sfx is None else sfx
    auc_df = pd.DataFrame(auc_list, index=['TE']).T
    auc_df.to_pickle(save_path / f'auc_list{sfx:s}.pkl')
    with open(save_path / f'recon_list{sfx:s}.pkl', 'wb') as f:
        pkl.dump(recon_list, f)
    time_df = pd.DataFrame(estimation_time, index=['wall', 'cpu']).T
    time_df.to_pickle(save_path / f'estimation_time{sfx:s}.pkl')
# %%