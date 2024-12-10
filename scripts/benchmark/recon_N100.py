#%%
import os
import yaml
import numpy as np
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
# refs=[3.0]*6+[0.5]*2+[0.0, 3.0]
# thresholds = [-50, -50, -50, -50, -50, -50, 10, 10, 0.9, 0.02]

for key, val in pm_causal_set.items():
    val = val.copy()
    val['path'] = root_path / val['path']
    # xx, yy = np.meshgrid(np.arange(45, 55), np.arange(45, 55))
    # np.save(val['path']/'mask_file.npy', np.vstack((xx.flatten(), yy.flatten())).astype(float))
    val['N'] = 100
    # val['mask_file'] = 'mask_file.npy'
    estimator = Causality.CausalityEstimator(
        **val, n_thread=int(os.cpu_count()/4))
    # estimator.get_optimal_delay(np.arange(20))
    # print(estimator.delay)
    estimator._run_estimation(regen=True, verbose=True)
    data = estimator.fetch_data(new_run=True)
    data.drop(columns=['TE(l=5)'], inplace=True)

    # % Plot distribution of TE values in log-scale
    data_matched = c4u.match_features(
        data, N=val['N'], conn_file=val['path']/val['conn_file'])
    data_recon, fig_data = c4u.reconstruction_analysis_TE(
        data_matched, nbins=40, hist_range=None, algorithm='EM')
    print(key, np.round(fig_data['acc_gauss']['TE'],2))

    try:
        fig = mplt.reconstruction_illustration_TE(fig_data)
        fig.savefig(val['path']/f"causal_recon_full_net{key}.pdf", transparent=True)
    except:
        print(f"Failed to save {key}")
        continue
    
#%%
val = pm_causal_set['Gaussian'].copy()
val['path'] = root_path / val['path']
estimator = Causality.CausalityEstimator(**val, n_thread=int(os.cpu_count()/4))
# estimator.get_optimal_delay(np.arange(20))
# estimator._run_estimation(regen=True, verbose=True)
data = estimator.fetch_data(new_run=True)
data.head()
#%%
# Plot distribution of TE values in log-scale
data_matched = c4u.match_features(
    data, N=val['N'], conn_file=val['path']/val['conn_file'])
data_recon, fig_data = c4u.reconstruction_analysis_TE(
    data_matched, nbins=50, hist_range=None, algorithm='EM')
print(np.round(fig_data['acc_gauss']['TE'],2))
fig = mplt.ReconstructionFigureTE(fig_data)
# fig.savefig(val['path']/f"causal_recon_full_net{key}.pdf", transparent=True)
#%%
import matplotlib.pyplot as plt
xrange = (1000, 2000)
voltage = c4u.fetch_voltage(val['path'] / (val['spk_fname'].replace('th=0.020', '') + '_voltage.dat'), N = val['N'], voltage_range=xrange)
spk_data = c4u.load_spike_data(val['path'] / (val['spk_fname'] + '_spike_train.dat'), xrange=xrange)
fig, ax = plt.subplots(2,1, figsize=(8,3), sharex=True)
ax[0].plot(voltage[:,0], voltage[:,1])
ax[0].plot(voltage[:,0], voltage[:,32])
ax[0].axhline(0.02, color='r', ls='--')
ax[0].set_ylabel('node\nactivity')
ax[1].plot(spk_data[:,0], spk_data[:,1], '|', ms=3)
ax[1].set_xlabel('time (ms)')
ax[1].set_ylabel('neuron\nIDs')
ax[1].set_xlim(xrange)
ax[1].set_ylim(0,100)