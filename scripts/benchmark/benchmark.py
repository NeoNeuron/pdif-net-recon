#%%
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import shutil
import causal4.Causality as Causality
import causal4.myplot as mplt
import causal4.utils as c4u
from pathlib import Path
root_path = Path(__file__).parents[2]

with open('benchmark100.yml', 'r') as yamlfile:
    pm_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
with open('benchmark100_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)

#%%
refs=[3.0]*6+[0.5]*2+[0.0, 3.0]
thresholds = [-50, -50, -50, -50, -50, -50, 10, 10, 0.9, 0.02]

for th, ref, (key, val) in zip(thresholds, refs, pm_causal_set.items()):
    val = val.copy()
    val['path'] = root_path / val['path']
    if key == 'Gaussian':
        val['spk_fname'] = val['spk_fname'].replace('th=0.020','')
    val['spk_fname'] = val['spk_fname']+ f'th={th:.2f}ref={ref:.2f}'
    val['N'] = 10
    estimator = Causality.CausalityEstimator(
        **val, n_thread=int(os.cpu_count()/4))
    # estimator.delay=16
    # estimator.get_optimal_delay(np.arange(20))
    # estimator._run_estimation(regen=True, verbose=True)
    # estimator.order = (1,1)
    data = estimator.fetch_data(new_run=True)
    data.drop(columns=['TE(l=5)'], inplace=True)
    # break

    # % Plot distribution of TE values in log-scale
    data_matched = c4u.match_features(
        data, N=val['N'], conn_file=f'/data/kchen/causal4-dev/scripts/generate_benchmark_dataset/N10_subnet/conn_mat_{key:s}.npy')
    data_recon, fig_data = c4u.reconstruction_analysis_TE(
        data_matched, nbins=40, hist_range=None, algorithm='EM')
    print(key, np.round(fig_data['acc_gauss']['TE'],2))

    try:
        fig = mplt.reconstruction_illustration_TE(fig_data)
        fig.savefig(val['path']/f"causal_recon_{key}.pdf", transparent=True)
    except:
        print(f"Failed to save {key}")
        # continue
    # break

#%%
from scipy.optimize import curve_fit
def exp_decay(t, tau, C):
    return np.exp(-t / tau) + C
idx = 100
timescale = []
decays = []
for th, ref, (key, val) in zip(thresholds, refs, pm_causal_set.items()):
    val = val.copy()
    val['path'] = root_path / val['path']
    # xx, yy = np.meshgrid(np.arange(45, 55), np.arange(45, 55))
    # np.save(val['path']/'mask_file.npy', np.vstack((xx.flatten(), yy.flatten())).astype(float))
    if key == 'Gaussian':
        val['spk_fname'] = val['spk_fname'].replace('th=0.020','')
    val['spk_fname'] = val['spk_fname']+ f'th={th:.2f}ref={ref:.2f}'
    val['N'] = 10
    tmax = 1e4*val['dt']
    spike_data = c4u.load_spike_data(
        val['path']/(val['spk_fname']+'_spike_train.dat'), 
        xrange=(0, tmax))
    spike_bool = c4u.spk2bin(spike_data[spike_data[:,1]==0], dt=val['dt'], tmax=tmax)
    print(key, spike_bool.shape)
    acf = c4u.ACF(spike_bool, nlags=100, plot=True)
    popt, pcov = curve_fit(exp_decay, np.arange(acf.shape[0]), acf, p0=(1, 0))
    print(popt)
    timescale.append(popt[0])
#%%
    # val['mask_file'] = 'mask_file.npy'
    estimator = Causality.CausalityEstimator(
        **val, n_thread=int(os.cpu_count()/4))
    # estimator.delay=16
    # estimator.get_optimal_delay(np.arange(20))
    # print(estimator.delay)
    # estimator._run_estimation(regen=True, verbose=True)
    # estimator.delay=1
    # estimator._run_estimation(regen=True, verbose=True)
    # %
    # estimator.order = (1,1)
    data = estimator.fetch_data(new_run=True)
    data.drop(columns=['TE(l=5)'], inplace=True)
    # break

    # % Plot distribution of TE values in log-scale
    data_matched = c4u.match_features(
        data, N=val['N'], conn_file=f'/data/kchen/causal4-dev/scripts/generate_benchmark_dataset/N10_subnet/conn_mat_{key:s}.npy')
    data_recon, fig_data = c4u.reconstruction_analysis_TE(
        data_matched, nbins=40, hist_range=None, algorithm='EM')
    print(key, np.round(fig_data['acc_gauss']['TE'],2))

    try:
        fig = mplt.reconstruction_illustration_TE(fig_data)
        fig.savefig(val['path']/f"causal_recon_{key}.pdf", transparent=True)
    except:
        print(f"Failed to save {key}")
    
#%%
fig, ax = plt.subplots(figsize=(10,3))
key = 'Gaussian'

ref=0.0
xrange=(0,100)
key_idx = keys.index(key)
val = pm_causal_set[key].copy()
val['path'] = root_path / val['path']
raster = mplt.plot_raster(
    val['path']/(val['spk_fname']+'_spike_train.dat'),
    xrange=xrange, return_spk=True, ax=ax, mew=5, alpha=1);
if key == 'Gaussian':
    val['spk_fname'] = val['spk_fname'].replace('th=0.020','')
val['spk_fname'] = val['spk_fname'] + f'th={thresholds[key_idx]:.2f}ref={ref:.2f}'
raster = mplt.plot_raster(
    val['path']/(val['spk_fname']+'_spike_train.dat'),
    xrange=xrange, return_spk=True, ax=ax);

#%%
key = 'HHEE'
val = pm_causal_set[key].copy()
val['path'] = root_path / val['path']
val['mask_file'] = 'mask_file.npy'
print(val.pop('T'))
# val['spk_fname'] = val['spk_fname'] + f'th={th:.2f}ref={ref:.2f}'
estimator = Causality.CausalityEstimator(
    **val, n_thread=int(os.cpu_count()/4))
# estimator.delay=16
# estimator.get_optimal_delay(np.arange(20)*0.5)
# print(estimator.delay)
# estimator.order = (1,8)
estimator._run_estimation(regen=True, verbose=True)
data = estimator.fetch_data(new_run=True)
# data.drop(columns=['TE(l=5)'], inplace=True)
# print(data.head(20))
print(estimator.causality_fname())

# % Plot distribution of TE values in log-scale
data_matched = c4u.match_features(
    data, N=val['N'], conn_file=val['path']/val['conn_file'])
data_recon, fig_data = c4u.reconstruction_analysis_TE(
    data_matched, nbins=20, hist_range=None, algorithm='EM')
print(key, fig_data['acc_gauss']['TE'])
fig = mplt.reconstruction_illustration_TE(fig_data)

#%%
tmp = np.fromfile('/data/kchen/causal4-dev/benchmark/N10/HHEE/HHp=0.25s=0.020f=0.080u=0.150_spike_train.dat',
                  dtype=float).reshape(-1,2)
np.save('/data/kchen/causal4-dev/benchmark/N10/HHEE/HHp=0.25s=0.020f=0.080u=0.150_spike_train.npy', tmp)
#%%
tmp = np.load('/data/kchen/causal4-dev/benchmark/N10/HHEE/HHp=0.25s=0.020f=0.080u=0.150_spike_train.npy')
plt.plot(tmp[:1000,0], tmp[:1000,1], '|')

#%%
xrange = (0,1000)
key = 'HHconEE'
# keys = ['HHconEE', 'HHconII', 'HHconEI', 'Lorenz', 'Lcon', 'Logistic', 'Gaussian']
pm = pm_causal_set[key].copy()
dataset_dir = root_path/pm['path']
# th = -46
ref = 3.0
thresholds = np.arange(-60, -39.9, 0.1)
for th in thresholds:
    spk_noisy_fname = c4u.binarize(
        dataset_dir/(pm['spk_fname']+'_voltage_short.npy'),
        N=10, threshold=th, T=pm['T'], verbose=True,
        ref=ref, force_regen=True)
    print(spk_noisy_fname)
#%%
# thresholds = np.arange(-50, -40.0, 0.1)
# thresholds = [-50,]
thresholds = [-50, -48, -46]
pm = pm_causal_set[key].copy()
plt.figure(figsize=(10,5))
tmp_list = []
counter = 0
for th in thresholds:
    tmp = np.fromfile(
        dataset_dir/(pm['spk_fname'] + f'th={th:.2f}ref={ref:.2f}_spike_train.dat'),
        dtype=float).reshape(-1,2)
    tmp[:,0] += counter
    counter += 1e5
    tmp_list.append(tmp)
    plt.plot(tmp[:,0], tmp[:,1], '|')
plt.xlim(0,1000)
# plt.xlim(100000,101000)
spk_data = np.vstack(tmp_list)
# spk_data = spk_data[spk_data[:,0].argsort()]
np.save(dataset_dir/(pm['spk_fname']+f'_multi_th_spike_train.npy'), spk_data)
#%%
pm = pm_causal_set[key].copy()
pm['path'] = root_path / pm['path']
pm['DT'] = 1e4
pm['T'] = pm['T']/100 * len(thresholds)+5e5
pm['order'] = (1,5)
pm['dt'] = 0.5
pm['delay'] = 3.0
# th = -52
# pm['spk_fname'] = pm['spk_fname'] + f'th={th:.2f}ref={ref:.2f}'
pm['spk_fname'] = pm['spk_fname'] + f'_multi_th'
estimator = Causality.CausalityEstimator(
    **pm, n_thread=int(os.cpu_count()/4))
# estimator.delay=16
# estimator.get_optimal_delay(np.arange(20)*0.5)
# print(estimator.delay)
# estimator.order = (1,8)
estimator._run_estimation(regen=True, verbose=True)
data = estimator.fetch_data(new_run=True)
# data.drop(columns=['TE(l=5)'], inplace=True)
# print(data.head(20))
print(estimator.causality_fname())

# % Plot distribution of TE values in log-scale
data_matched = c4u.match_features(
    data, N=pm['N'], conn_file=pm['path']/pm['conn_file'])
data_recon, fig_data = c4u.reconstruction_analysis_TE(
    data_matched, nbins=100, hist_range=None, algorithm='EM')
print(key, fig_data['acc_gauss']['TE'])
fig = mplt.reconstruction_illustration_TE(fig_data)
# %%
