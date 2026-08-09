# %%
import numpy as np
import pandas as pd
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import yaml
from utils import get_spk_fname
from glmcc.Est_Data import Est_Data

# Load binarization thresholds and refs from YAML file
with open(root_path / 'scripts/benchmark' / 'binarization.yaml', 'r') as binarization_file:
    binarization_cfg = yaml.load(binarization_file, Loader=yaml.FullLoader)
refs = binarization_cfg.get('refractory', {})
thresholds = binarization_cfg.get('threshold', {})
dt = binarization_cfg.get('dt',{})

regen=True

def core_function(key, val, shuffle_id:int=None, noise_level=None, T:float=None):
    #! Calculate GLMCC
    if noise_level is not None and np.abs(noise_level) < 1e-6:
        noise_level = None  # 0.0 is functionally identical to None (no noise added)
    N = val['N']
    T_in_sec = val['T'] / 1e3 if T is None else T / 1e3 # convert from ms to s
    if noise_level is None:
        spk_fname = get_spk_fname(val['spk_fname'])
    else:
        spk_fname = get_spk_fname(
            val['spk_fname'], f'_noisy{noise_level:.1f}', th=thresholds[key], ref=refs[key])
        # val['spk_fname'], sfx=sfx, th=thresholds[key], ref=refs[key])
    conn_fname = val['path'] / val['conn_file']
    conn = np.load(conn_fname)

    if shuffle_id is None:
        indices = np.arange(N, dtype=int)
        print(f"[INFO]: Estimating GLMCC for {key} with T={T_in_sec:.0f} s, noise_level={noise_level} ...")
    else:
        indices = np.load(val['path'].parent / 'subnet_indices.npy')[shuffle_id]
        print(f"[INFO]: Estimating GLMCC for {key} with T={T_in_sec:.0f} s, noise_level={noise_level}, shuffle_id={shuffle_id}...")

    W, cpu_time, wall_time = Est_Data(
        val['path'], spk_fname, N=N, T=T_in_sec, indices=indices,
        outfile_sfx=None if shuffle_id is None else f'{shuffle_id:d}',
        DELTA=dt[key], WIN=dt[key]*50,
        n_jobs=indices.shape[0])
    try:
        if shuffle_id is None:
            GLMCC = np.load(val['path'] / f"W_GLM_{T_in_sec:.0f}-{spk_fname:s}.npy")
        else:
            GLMCC = np.load(val['path'] / f"W_GLM_{T_in_sec:.0f}-{spk_fname:s}_{shuffle_id:d}.npy")
        xx, yy = np.meshgrid(indices, indices, indexing='ij')
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

        save_path = val['path'].parents[2] / 'results' / 'GLMCC'
        save_path.mkdir(parents=True, exist_ok=True)
        t_tag = '' if T is None else f'_T={T:.2e}'

        if shuffle_id is None:
            recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}{t_tag:s}_fullnet.pkl')
        else:
            if noise_level is None:
                recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}{t_tag:s}_{shuffle_id:d}.pkl')
            else:
                recon_df.to_pickle(save_path / f'recon_df_noise_{noise_level:.1f}_{key:s}{t_tag:s}_{shuffle_id:d}.pkl')

    except Exception as e:
        print(e)
# %%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str)
    parser.add_argument('--idx', type=int, default=None)
    parser.add_argument('--noise_level', type=float, default=None)
    parser.add_argument('--T', type=float, default=None,
        help='Duration (ms, same units as the config T field) of data to use for '
             'GLMCC estimation. Defaults to the full T from --cfg-file.')
    parser.add_argument('--cfg-file', dest='cfg_file', type=str, default='benchmark_causal.yml')
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent / args.cfg_file, 'r') as yamlfile:
        pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
    for key in pm_causal_set.keys():
        pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

    core_function(args.key, pm_causal_set[args.key], args.idx, args.noise_level, args.T)
#%%

# for noise_level in [0.1, 0.2, 0.3, 0.4]:
#     for key, val in pm_causal_set.items():
#         for i in range(indices.shape[0]):
#             core_function(key, val, noise_level, i)

#%%