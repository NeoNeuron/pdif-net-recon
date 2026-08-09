# %%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
from causal4.Causality import CausalityEstimator
from causal4.utils import match_features
import yaml
from utils import get_spk_fname
import multiprocessing
num_cpus = multiprocessing.cpu_count()

# Load binarization thresholds and refs from YAML file
with open(root_path / 'scripts/benchmark' / 'binarization.yaml', 'r') as binarization_file:
    binarization_cfg = yaml.load(binarization_file, Loader=yaml.FullLoader)
refs = binarization_cfg.get('refractory', {})
thresholds = binarization_cfg.get('threshold', {})
#%%
regen=True

#! Calculate PTD-TE
def core_function(key, val, shuffle_id:int=None, noise_level=None, T:float=None):
    val = dict(val)
    if T is not None:
        val['T'] = T
    if noise_level is not None:
        noisy_spk_fname = get_spk_fname(
            val['spk_fname'], f'_noisy{noise_level:.1f}', th=thresholds[key], ref=refs[key])
    # create mask file 
    if shuffle_id is None:
        indices = np.arange(val['N'], dtype=int)
        mask_fname = None
    else:
        indices = np.load(val['path'].parent / 'subnet_indices.npy')[shuffle_id]
        mask_fname = val['path'] / f'mask_{shuffle_id:d}.npy'
        if not mask_fname.exists():
            xx, yy = np.meshgrid(indices, indices, indexing='ij')
            mask_buff = np.vstack([xx.flatten(), yy.flatten()])
            np.save(mask_fname, mask_buff)

    n_thread = min(int(indices.shape[0]*(indices.shape[0]-1)), num_cpus)
    estimator = CausalityEstimator(**val, n_thread=n_thread, mask_file=mask_fname)
    if noise_level is not None:
        estimator.spk_fname = noisy_spk_fname
    _, text = estimator._run_estimation(regen=regen, verbose=False, return_log=True)
    wall_time = float(text.splitlines()[-1].split()[3])
    cpu_time = float(text.splitlines()[-1].split()[7])
    print(key, f"Elapsed cpu time: {cpu_time:.2f} s")
    print(key, f"Elapsed time: {wall_time:.2f} s")
    data = estimator.fetch_data(new_run=True)
    data = data[['pre_id', 'post_id', 'TE', 'Delta_p']]
    # Plot distribution of TE values in log-scale
    recon_df = match_features(
        data, N=val['N'], conn_file=val['path']/val['conn_file'])

    recon_df.attrs['wall_time'] = wall_time
    recon_df.attrs['cpu_time']  = cpu_time
    save_path = val['path'].parents[2] / 'results' / 'PTD-TE'
    save_path.mkdir(parents=True, exist_ok=True)
    t_tag = '' if T is None else f'_T={T:.2e}'
    if shuffle_id is None:
        recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}{t_tag:s}_fullnet.pkl')
    else:
        if noise_level is not None:
            recon_df.to_pickle(save_path / f'recon_df_noise_{noise_level:.1f}_{key:s}{t_tag:s}_{shuffle_id:d}.pkl')
        else:
            recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}{t_tag:s}_{shuffle_id:d}.pkl')

#%%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str)
    parser.add_argument('--idx', type=int, default=None)
    parser.add_argument('--noise_level', type=float, default=None)
    parser.add_argument('--T', type=float, default=None,
        help='Duration (ms, same units as the config T field) of data to use for '
             'causality estimation. Defaults to the full T from --cfg-file.')
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
# %%