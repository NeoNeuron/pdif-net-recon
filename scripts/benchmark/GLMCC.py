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
# yml_names = ['benchmark10_causal.yml',
#              'benchmark10_subnet_causal.yml',
#              'benchmark10_subnet_causal.yml',
#              'benchmark10_subnet_causal.yml',
#              'benchmark10_subnet_causal.yml',
#              'benchmark10_subnet_causal.yml',
#             ]
# sfxs = ['', '', '_noisy1', '_noisy2', '_noisy3', '_noisy4']
# save_folder_names = [
#     'N10',
#     'N10_subnet',
#     'N10_subnet_noisy',
#     'N10_subnet_noisy',
#     'N10_subnet_noisy',
#     'N10_subnet_noisy',
#     ]
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

for key, val in pm_causal_set.items():
    key = 'Logistic'
    val = pm_causal_set[key]
    #! Calculate GLMCC
    # load data
    # mask = (1 - np.eye(10)).astype(bool)
        # if key in recon_list and not regen:
        #     print(f'{key} already exists')
        #     continue
    N = val['N']
    spk_fname = get_spk_fname(val['spk_fname'])
        # val['spk_fname'], sfx=sfx, th=thresholds[key], ref=refs[key])
    conn_fname = val['path'] / val['conn_file']
    conn = np.load(conn_fname)
    # T = 10
    T = val['T']
    if key in ['Gaussian', 'Rcon', 'Logistic', 'RNN']:
        val['T'] /= 10
    for i in range(indices.shape[0]):
        W, cpu_time, wall_time = Est_Data(
            val['path'], spk_fname, N=N, T=T, indices=indices[i],
            outfile_sfx=f'{i:d}', DELTA=dt[key], WIN=dt[key]*50)
        try:
            GLMCC = np.load(val['path'] / f"W_GLM_{T:.0f}-{spk_fname:s}_{i:d}.npy")
            xx, yy = np.meshgrid(indices[i], indices[i], indexing='ij')
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
            recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}_{i:d}.pkl')
            # column (row) index representing pre_id (post_id)
            # auc ={'glmcc':roc_auc_score(glmcc_df['connection'], glmcc_df['glmcc']),
            #       'glmcc_abs':roc_auc_score(glmcc_df['connection'], glmcc_df['glmcc_abs']),}
            # auc_list[key] = auc
            # estimation_time[key] = {'wall':wall_time, 'cpu':cpu_time}
            # print(key, f"{wall_time:.2f}", auc)
        except Exception as e:
            print(e)
            # recon_list[key] = None
            # auc_list[key] = {'glmcc': np.nan, 'glmcc_abs': np.nan}
            # estimation_time[key] = {'wall': np.nan, 'cpu': np.nan}
        # auc_df = pd.DataFrame(auc_list).T
        # auc_df.to_pickle(save_path / f'auc_list{sfx:s}.pkl')
        # with open(save_path / f'recon_list{sfx:s}.pkl', 'wb') as f:
        #     pkl.dump(recon_list, f)
        # time_df = pd.DataFrame(estimation_time).T
        # time_df.to_pickle(save_path / f'estimation_time{sfx:s}.pkl')
# %%