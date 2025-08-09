# %%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
from causal4.Causality import CausalityEstimator
from causal4.utils import match_features
import yaml
from utils import get_vfname, get_spk_fname
import multiprocessing
num_cpus = multiprocessing.cpu_count()
#%%

with open(Path(__file__).resolve().parent / 'benchmark_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']
#%%
# sfxs = ['', '', '_noisy1', '_noisy2', '_noisy3', '_noisy4']
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
# spk_noisy_fname = c4u.binarize(
#     pm['path']/get_vfname(pm['spk_fname'], sfx=f'_noisy{i:d}'),
#     N=int(pm['N']), threshold=th, T=pm['T'], verbose=True,
#     ref=ref, force_regen=False, sfx=f'noisy{i:d}',
#     preprocess_fn=lambda x: x+np.random.randn(*x.shape)*sigma,)
#! Calculate PTD-TE
#%%
for key, val in pm_causal_set.items():
    val = val.copy()
    # vol_std = np.load(get_vfname(val['spk_fname']), mmap_mode='r')[:10000, 1:].std()
    for i in range(indices.shape[0]):
        noisy_vfname = get_vfname(val['spk_fname'], '_n3')
        # for key, val in pm_causal_set.items():
            # if key in recon_list and not regen:
            #     print(f'{key} already exists')
            #     continue
        # sfx = ''
        
        #     val['spk_fname'], sfx=sfx, th=thresholds[key], ref=refs[key])
        # create mask file 
        mask_fname = val['path'] / f'mask_{i:d}.npy'
        if not mask_fname.exists():
            xx, yy = np.meshgrid(indices[i], indices[i], indexing='ij')
            mask_buff = np.vstack([xx.flatten(), yy.flatten()])
            np.save(mask_fname, mask_buff)

        n_thread = min(int(indices.shape[1]*(indices.shape[1]-1)), num_cpus)
        estimator = CausalityEstimator(**val, n_thread=n_thread, mask_file=mask_fname)
        _, text = estimator._run_estimation(regen=regen, verbose=False, return_log=True)
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

        recon_df.attrs['wall_time'] = wall_time
        recon_df.attrs['cpu_time']  = cpu_time
        print(recon_df.attrs)
        recon_df.to_pickle(save_path / f'recon_df_noise_0_{key:s}_{i:d}.pkl')
    
# %%