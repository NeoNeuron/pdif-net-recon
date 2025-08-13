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

with open(Path(__file__).resolve().parent / 'benchmark_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

# Load binarization thresholds and refs from YAML file
with open(root_path / 'scripts/benchmark' / 'binarization.yaml', 'r') as binarization_file:
    binarization_cfg = yaml.load(binarization_file, Loader=yaml.FullLoader)
refs = binarization_cfg.get('refractory', {})
thresholds = binarization_cfg.get('threshold', {})
#%%
regen=True
save_path = root_path / 'results/PTD-TE'
save_path.mkdir(parents=True, exist_ok=True)

indices = np.load(root_path / 'benchmark' / 'N100' / 'subnet_indices.npy')
#! Calculate PTD-TE
def core_function(key, val, shuffle_id, noise_level=None):
    if noise_level is not None:
        noisy_spk_fname = get_spk_fname(
            val['spk_fname'], f'_noisy{noise_level:.1f}', th=thresholds[key], ref=refs[key])
    # create mask file 
    mask_fname = val['path'] / f'mask_{shuffle_id:d}.npy'
    if not mask_fname.exists():
        xx, yy = np.meshgrid(indices[shuffle_id], indices[shuffle_id], indexing='ij')
        mask_buff = np.vstack([xx.flatten(), yy.flatten()])
        np.save(mask_fname, mask_buff)

    n_thread = min(int(indices.shape[1]*(indices.shape[1]-1)), num_cpus)
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
    print(recon_df.attrs)
    if noise_level is not None:
        recon_df.to_pickle(save_path / f'recon_df_noise_{noise_level:.1f}_{key:s}_{shuffle_id:d}.pkl')
    else:
        recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}_{shuffle_id:d}.pkl')

#%%
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
# %%