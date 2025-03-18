# %%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
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
        return fname + '_spike_train.npy'
    else:
        if key == 'Gaussian':
            fname = fname.replace('th=0.020','')
        fname += sfx + f'_th={th:.2f}ref={ref:.2f}'
        return fname + '_spike_train.npy'

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

    #! Calculate GLMCC
    # load data
    recon_list = {}
    auc_list = {}
    estimation_time = {}
    mask = (1 - np.eye(10)).astype(bool)
    for i, (key, val) in enumerate(pm_causal_set.items()):
        # print(key)
        if key in ['Logistic',]:
            recon_list[key] = None
            auc_list[key] = np.nan
            estimation_time[key] = [np.nan, np.nan]
            continue
        N = val['N']
        spk_fname = get_spk_fname(
            val['spk_fname'], sfx=sfx, th=thresholds[key], ref=refs[key])
        cml = 'python -m glmcc.Est_Data ' \
            + str(val['path']) + ' ' + spk_fname \
            + ' ' + str(N) + ' ' + str(val['T']/1e3) \
            + ' sim GLM'
        # t0_wall = time.time()
        # t0_cpu = time.process_time()
        # outputs = run_shell_command(cml)
        # t1_cpu = time.time()
        # t1_wall = time.time()

        # wall_time = t1_wall-t0_wall
        # cpu_time = t1_cpu-t0_cpu
        save_path = root_path / 'results' / sfname / 'GLMCC'
        if key in ['Lorenz', 'Lcon']:
            if sfx is None:
                with open(save_path / (key + '_old.txt'), 'r') as f:
                    tmp = f.readlines()
            else:
                with open(save_path / (key + '_old' + sfx[-1] + '.txt'), 'r') as f:
                    tmp = f.readlines()
        else:
            if sfx is None:
                with open(save_path / (key + '.txt'), 'r') as f:
                    tmp = f.readlines()
            else:
                with open(save_path / (key + sfx[-1] + '.txt'), 'r') as f:
                    tmp = f.readlines()
        wall_time = float(tmp[-1].split(' ')[-2])
        cpu_time = float(tmp[-2].split(' ')[-2])

        # print(key, wall_time, cpu_time)
        
        try:
            if key == 'Gaussian':
                val['T'] /= 10
            if key in ['Lorenz', 'Lcon']:
                GLMCC = np.loadtxt(
                    val['path'] / f"W_GLM_{val['T']/1e3:.0f}_old-{spk_fname.replace('npy', 'csv'):s}",
                    delimiter=',').T
            else:
                GLMCC = np.loadtxt(
                    val['path'] / f"W_GLM_{val['T']/1e3:.0f}-{spk_fname.replace('npy', 'csv'):s}",
                    delimiter=',').T
            conn_fname = val['path'] / val['conn_file']
            xx, yy = np.meshgrid(np.arange(N), np.arange(N))
            if conn_fname.suffix == '.dat': 
                conn = np.fromfile(conn_fname, dtype=float).reshape(N,N).T
            elif conn_fname.suffix == '.npy':
                conn = np.load(conn_fname).T
            else:
                raise ValueError('Unknown file format')
            df_list = []
            GLMCC[GLMCC==0] = 1e-12
            glmcc_df = pd.DataFrame({'pre_id': xx[mask],
                                'post_id': yy[mask],
                                'glmcc': GLMCC[mask],
                                'glmcc_abs': np.abs(GLMCC[mask]),
                                'log-glmcc_abs': np.log(np.abs(GLMCC[mask])),
                                'connection': conn[mask]})
            recon_list[key] = glmcc_df
            # column (row) index representing pre_id (post_id)
            auc =[roc_auc_score(glmcc_df['connection'], glmcc_df['glmcc']),
                  roc_auc_score(glmcc_df['connection'], glmcc_df['glmcc_abs']),]
            auc_list[key] = auc
            estimation_time[key] = [wall_time, cpu_time]
            print(key, f"{wall_time:.2f}", auc)
        except Exception as e:
            print(e)
            recon_list[key] = None
            auc_list[key] = np.nan
            estimation_time[key] = [np.nan, np.nan]
    
    save_path = root_path / 'results' / sfname / 'GLMCC'
    save_path.mkdir(parents=True, exist_ok=True)
    sfx = '' if sfx is None else sfx
    auc_df = pd.DataFrame(auc_list, index=['glmcc', 'glmcc_abs']).T
    auc_df.to_pickle(save_path / f'auc_list{sfx:s}.pkl')
    with open(save_path / f'recon_list{sfx:s}.pkl', 'wb') as f:
        pkl.dump(recon_list, f)
    time_df = pd.DataFrame(estimation_time, index=['wall', 'cpu']).T
    time_df.to_pickle(save_path / f'estimation_time{sfx:s}.pkl')
# %%