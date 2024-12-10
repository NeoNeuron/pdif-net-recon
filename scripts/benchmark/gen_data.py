#%%
import yaml
import numpy as np
import matplotlib.pyplot as plt
import causal4.utils as c4u
import shutil

from pathlib import Path, PosixPath
root_path = Path(__file__).parents[2] 
from subprocess import call
from multiprocessing import Pool
from causal4.io import chunked_npy_operation, chunked_voltage_operation

def arg_wrapper(pm_dict:dict):
    wrapped_list = []
    for key, val in pm_dict.items():
        if isinstance(val, tuple):
            wrapped_list.append(f'--{key}='+' '.join([str(i) for i in val]))
            f'--{key}={val[0]}:{val[1]}'
        elif isinstance(val, PosixPath):
            wrapped_list.append(f'--{key}={str(val)}/')
        else:
            wrapped_list.append(f'--{key}={val}')
    return wrapped_list

def run_simulation(pm_simulation:dict):
    _pm = pm_simulation.copy()
    simulator = _pm.pop('simulator')
    simulator = str(root_path / simulator)
    if '.py' in simulator:
        call(['python', simulator] + arg_wrapper(_pm))
    else:
        call([simulator] + arg_wrapper(_pm))
    # dump yaml config to pm_simulation['record_path']
    with open(_pm['record_path'] + 'config.yml', 'w') as yamlfile:
        yaml.dump(pm_simulation, yamlfile)

def get_vol_fname(fname:str, key:str):
    if key in ['Lorenz', 'Lcon']:
        return fname + '_x'
    elif key == 'Gaussian':
        return fname.replace('th=0.020', '')+'_voltage'
    else:
        return fname + '_voltage'

def joyplot_voltage(data, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    for i in range(data.shape[1]-1):
        time = data[:,0]
        vol = data[:,i+1]
        vol = (vol - vol.min())/(vol.max()-vol.min())*0.9+i
        ax.plot(time, vol, **kwargs)
    ax.set_yticks(np.arange(data.shape[1])+.5, range(1,data.shape[1]+1))
    return ax

#%% load config files
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
#%% generate subset of data for causal inference
keys = ['HHEE', 'HHII', 'HHEI',
        'HHconEE', 'HHconII', 'HHconEI',
        'Lorenz', 'Lcon',
        'Logistic', 'Gaussian']
#%%
# select voltage time series from 10 neurons for state-space reconstruction methods
for key in keys:
    pm = pm_causal_set[key]
    fname_old = pm['path']/(get_vol_fname(pm['spk_fname'], key)+'.dat')
    fname_new=Path(str(fname_old).replace('N100', 'N10_subnet')).with_suffix('.npy')
    fname_new.parent.mkdir(parents=True, exist_ok=True)
    N = pm['N']
    if N == 100:
        # truncate the voltage time series to the middle 10 neurons
        chunked_voltage_operation(
            lambda x: np.hstack((x[:,0:1], x[:,46:56])), # select neuron 45-55
            fname_old, fname_new, N=N, N_out=10, num_chunks=20)
        # truncate the connectivity matrix 
        fname_old = fname_old.with_name(pm['conn_file'])
        fname_new = fname_new.with_name(pm['conn_file']).with_suffix('.npy')
        conn_mat = np.fromfile(fname_old, dtype=float).reshape(N,N)
        np.save(fname_new, conn_mat[45:55,45:55].astype(int))
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
    # else:
    #     chunked_voltage_operation(lambda x:x, fname_old, fname_new, N=N, num_chunks=20)
    print(pm['dt'], pm['T'])
#%% select spike train for 10 neurons
for key in keys:
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
#%% move files
for key in keys:
    pm = pm_causal_set[key]
    new_folder = root_path / 'share' / key
    new_folder.mkdir(parents=True, exist_ok=True)
    fname = pm['path']+pm['spk_fname']
    fname_old = pm['path']/(get_vol_fname(pm['spk_fname'], key)+'.dat')
    fname_new = pm['path']/(get_vol_fname(pm['spk_fname'], key)+'_short.npy')
    shutil.copy(fname_old, new_folder)
    shutil.copy(fname_new, new_folder)
    shutil.copy(Path(pm['path'])/'config.yml', new_folder)
    
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
    vol_fname = datadir / (get_vol_fname(pm['spk_fname'], key) + '.dat')
    sigma_e = sigma * np.sqrt(dt)
    for i in range(3):
        print(sigma_e/4*(i+1))
        chunked_voltage_operation(
            add_noise, vol_fname,
            vol_fname.with_stem(vol_fname.stem+f'_noisy_{i+1:d}').with_suffix('.npy'),
            N=10, num_chunks=20, sigma=sigma_e/4*(i+1))
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
#%%
sigmas = [2,2,2,2,2,2,2,2,0.1,0.01]
old_path = root_path/'share/N10/'
new_path = root_path/'share/N10_with_noise/'
# keys = ['HHconEE', 'HHconII', 'HHconEI',]
for key, sigma in zip(pm_causal_set.keys(), sigmas):
    pm = pm_causal_set[key]
    v_dir = old_path/(get_vol_fname(pm['spk_fname'], key)+'_short.npy')
    voltage = np.load(v_dir)
    Tn = 1000
    plt.figure(figsize=(12,3))
    plt.plot(voltage[:Tn,0], voltage[:Tn,1])
    voltage[:,1:] += np.random.randn(voltage.shape[0], voltage.shape[1]-1)*sigma
    plt.plot(voltage[:Tn,0], voltage[:Tn,1])
    (new_path/key).mkdir(parents=True, exist_ok=True)
    np.save(new_path/key/v_dir.name, voltage)
    shutil.copy(old_path/key/'config.yml', new_path/key)
# %% binarization noisy sub-network
refs = [3.0]*6+[0.5]*2+[0.0, 3.0]
thresholds = [-50]*6 + [10, 10, 0.9, 0.02]
for (key, pm), ref, th in zip(pm_causal_set.items(), refs, thresholds):
    dataset_dir = Path(str(pm['path']).replace('N100', 'N10_subnet'))
    # xrange = (0,1000)
    # spk_raw = c4u.load_spike_data(dataset_dir/(pm['spk_fname']+'_spike_train.dat'),
    #                     xrange=xrange, verbose=True)
    spk_noisy_fname = c4u.binarize(
        dataset_dir/(get_vol_fname(pm['spk_fname'],key)+'_noisy.npy'),
        N=int(pm['N']/10), threshold=th, T=pm['T'], verbose=True,
        ref=ref, force_regen=True)
    print(spk_noisy_fname)
#%%
