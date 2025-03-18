#%%
import yaml
import numpy as np
import matplotlib.pyplot as plt
import causal4.utils as c4u

from pathlib import Path
root_path = Path(__file__).parents[2] 
from multiprocessing import Pool
from causal4.io import chunked_npy_operation, chunked_voltage_operation
from utils import run_simulation, get_vol_fname
#%% [markdown]
# # Part 1: 10-neuron network
#%% load config files for data generation and causal inference
with open('benchmark10.yml', 'r') as yamlfile:
    pm_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
with open('benchmark10_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)

for key in pm_set.keys():
    pm_set[key]['record_path'] = root_path / pm_set[key]['record_path']
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']
#%% generate time series for benchmarks
pool = Pool(len(pm_set))
results = [pool.apply_async(run_simulation, args=(val,)) for key, val in pm_set.items()]
pool.close()
pool.join()
#%% [markdown]
# # Part 2: 100-neuron network
#%% load config files for data generation and causal inference
with open('benchmark100.yml', 'r') as yamlfile:
    pm_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
with open('benchmark100_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)

for key in pm_set.keys():
    pm_set[key]['record_path'] = root_path / pm_set[key]['record_path']
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']
#%% generate time series for benchmarks
pool = Pool(len(pm_set))
results = [pool.apply_async(run_simulation, args=(val,)) for key, val in pm_set.items()]
pool.close()
pool.join()

#%%
# select voltage time series from 10 neurons for state-space reconstruction methods
selected_ids = np.arange(45,55)
for key in pm_causal_set.keys():
    pm = pm_causal_set[key]
    fname_old = pm['path']/(get_vol_fname(pm['spk_fname'], key)+'.dat')
    fname_new=Path(str(fname_old).replace('N100', 'N10_subnet')).with_suffix('.npy')
    fname_new.parent.mkdir(parents=True, exist_ok=True)
    N = pm['N']
    # truncate the voltage time series to the middle 10 neurons
    chunked_voltage_operation(
        lambda x: np.hstack((x[:,0:1], x[:,selected_ids+1])), # select neuron 45-55
        fname_old, fname_new, N=N, N_out=10, num_chunks=20)
    # truncate the connectivity matrix 
    fname_old = fname_old.with_name(pm['conn_file'])
    fname_new = fname_new.with_name(pm['conn_file']).with_suffix('.npy')
    conn_mat = np.fromfile(fname_old, dtype=float).reshape(N,N)
    np.save(fname_new, conn_mat[selected_ids][:,selected_ids].astype(int))
    with open(fname_old.with_name('config.yml'), 'r') as yamlfile:
        config_pm = yaml.load(yamlfile, Loader=yaml.FullLoader)
    if 'NE' in config_pm:
        config_pm['NE'] = int(config_pm['NE']//10)
        config_pm['NI'] = int(config_pm['NI']//10)
    else:
        config_pm['N'] = int(config_pm['N']//10)
    config_pm['record_path'] = config_pm['record_path'].replace('N100', 'N10_subnet')
    with open(fname_new.with_name('config.yml'), 'w') as yamlfile:
        yaml.dump(config_pm, yamlfile)
    print(pm['dt'], pm['T'])
#%% select spike train for 10 neurons
for key in pm_causal_set.keys():
    pm = pm_causal_set[key]
    fname_old = pm['path']/(pm['spk_fname']+'_spike_train.dat')
    fname_new=Path(str(fname_old).replace('N100', 'N10_subnet')).with_suffix('.npy')
    print(fname_old.exists(), fname_old.name)
    print(fname_new.exists(), fname_new.name)
    fname_new.parent.mkdir(parents=True, exist_ok=True)
    spk_data_old = np.fromfile(fname_old, dtype=float).reshape(-1,2)
    mask = (spk_data_old[:,1] >= 45)*(spk_data_old[:,1] < 55)
    print(spk_data_old.shape)
    print(spk_data_old[mask].shape)
    spk_data_new = spk_data_old[mask]
    spk_data_new[:,1] -= 45
    np.save(fname_new, spk_data_new)

#%% add noise to the data
dts = [0.2]*6 + [0.01, 0.02] + [1, 1]
sigmas = [4]*6 + [20, 16, 0.1, 0.01]
def add_noise(voltage, sigma):
    voltage[:,1:] += np.random.randn(voltage.shape[0], voltage.shape[1]-1)*sigma
    return voltage
for key, sigma, dt in zip(pm_causal_set.keys(), sigmas, dts):
    print(key)
    pm = pm_causal_set[key]
    datadir = Path(str(pm['path']).replace('N100', 'N10_subnet'))
    vol_fname = datadir / (get_vol_fname(pm['spk_fname'], key) + '.npy')
    sigma_e = sigma * np.sqrt(dt)
    for i in range(1,5):
        print(sigma_e/4*i)
        chunked_npy_operation(
            add_noise, vol_fname,
            vol_fname.with_stem(vol_fname.stem+f'_noisy{i:d}').with_suffix('.npy'),
            num_chunks=20, sigma=sigma_e/4*i)
#%%
# convert *.dat to *.npy
def identity(x):
    return x
for key in pm_causal_set.keys():
    print(key)
    pm = pm_causal_set[key]
    datadir = Path(str(pm['path']))
    vol_fname = datadir / (get_vol_fname(pm['spk_fname'], key) + '.dat')
    chunked_voltage_operation(
        identity, vol_fname,
        vol_fname.with_suffix('.npy'),
        N=10, num_chunks=20)
#%% [markdown]
# # Part 3: 10-neuron subnetworks
#%%
with open('benchmark10_subnet_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)

for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']
#%% visualization of noisy voltage time series
import seaborn as sns
for key in pm_causal_set.keys():
    pm = pm_causal_set[key]
    print(key)
    fname = get_vol_fname(pm['spk_fname'], key)
    vol_clean = np.load(pm['path'] / (fname + '.npy'), mmap_mode='r')
    vol_noisy = np.load(pm['path'] / (fname + '_noisy.npy'), mmap_mode='r')
    dt = vol_clean[1,0] - vol_clean[0,0]
    print(dt)
    Tn = 80 if key.startswith('L') else 400
    _Tn = int(Tn/dt)
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(vol_clean[:_Tn,0], vol_clean[:_Tn,1], zorder=5, lw=2, label='clean')
    ax.plot(vol_noisy[:_Tn,0], vol_noisy[:_Tn,1], label='noisy')
    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.legend(fontsize=18)
    plt.tight_layout()
    (root_path/'figures/N10_subnet').mkdir(parents=True, exist_ok=True)
    sns.despine()
    fig.savefig(root_path/'figures/N10_subnet' / (key+'_vol_demo.png'))

    Tn = 800 if key.startswith('L') else 4000
    _Tn = int(Tn/dt)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(vol_clean[:_Tn,1], vol_clean[1:1+_Tn,1], 'o', zorder=5, ms=0.5)
    ax.plot(vol_noisy[:_Tn,1], vol_noisy[1:1+_Tn,1], 'o', ms=0.5)
    ax.plot([], [], 'o', c='C0', ms=5, label='clean')
    ax.plot([], [], 'o', c='C1', ms=5, label='noisy')
    ax.set_xlabel(r'$X_t$', fontsize=18)
    ax.set_ylabel(r'$X_{t+1}$', fontsize=18)
    ax.legend(fontsize=18)
    sns.despine()
    plt.tight_layout()
    fig.savefig(root_path/'figures/N10_subnet' / (key+'_return_map.png'))
# %% binarization noisy subetworks
refs = [3.0]*6+[0.5]*2+[0.0, 3.0]
thresholds = [-50]*6 + [10, 10, 0.9, 0.02]
for (key, pm), ref, th in zip(pm_causal_set.items(), refs, thresholds):
    # xrange = (0,1000)
    # spk_raw = c4u.load_spike_data(dataset_dir/(pm['spk_fname']+'_spike_train.dat'),
    #                     xrange=xrange, verbose=True)
    spk_noisy_fname = c4u.binarize(
        pm['path']/(get_vol_fname(pm['spk_fname'],key)+'_noisy.npy'),
        N=int(pm['N']), threshold=th, T=pm['T'], verbose=True,
        ref=ref, force_regen=False, sfx=None)#'noisy1')
    print(spk_noisy_fname.name)
#%%
