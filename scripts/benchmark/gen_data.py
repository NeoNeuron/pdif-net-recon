#%%
import yaml
import numpy as np

from pathlib import Path
root_path = Path(__file__).parents[2] 
from multiprocessing import Pool
from pdif.io import chunked_voltage_operation
from utils import run_simulation, get_vfname
#%% [markdown]
# # Part 1: 10-neuron network
#%% load config files for data generation and causal inference
with open(root_path/'scripts/benchmark/benchmark10.yml', 'r') as yamlfile:
    pm_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
with open(root_path/'scripts/benchmark/benchmark10_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)

for key in pm_set.keys():
    pm_set[key]['record_path'] = root_path / pm_set[key]['record_path']
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']
# %%
run_simulation(pm_set['RNN'])
import sys
sys.exit(0)
#%% generate time series for benchmarks
pool = Pool(len(pm_set))
results = [pool.apply_async(run_simulation, args=(val,)) for val in pm_set.values()]
pool.close()
pool.join()
#%% [markdown]
# # Part 2: 100-neuron network
#%% load config files for data generation and causal inference
with open(root_path/'scripts/benchmark/benchmark100.yml', 'r') as yamlfile:
    pm_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
with open(root_path/'scripts/benchmark/benchmark100_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)

for key in pm_set.keys():
    pm_set[key]['record_path'] = root_path / pm_set[key]['record_path']
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']
#%% generate time series for benchmarks
pool = Pool(len(pm_set))
results = [pool.apply_async(run_simulation, args=(val,)) for val in pm_set.values()]
pool.close()
pool.join()

#%%
# convert *.dat to *.npy
def identity(x):
    return x
for key in pm_causal_set.keys():
    print(key)
    pm = pm_causal_set[key]
    datadir = Path(str(pm['path']))
    vol_fname = (datadir / get_vfname(pm['spk_fname'])).with_suffix('.dat')
    chunked_voltage_operation(
        identity, vol_fname,
        vol_fname.with_suffix('.npy'),
        N=10, num_chunks=20)
#%% [markdown]
# # Part 3: generate subnetwork seed
#%%
np.random.seed(42)
n_trials = 10
indices = np.zeros((n_trials, 10), dtype=int)
for i in range(n_trials):
    indices[i] = np.sort(np.random.choice(100, 10, replace=False))
np.save(root_path / 'benchmark' / 'N100' / 'subnet_indices.npy', indices)
print(indices)
#%%