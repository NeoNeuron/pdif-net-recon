#%%
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import causal4.Causality as Causality
import causal4.myplot as mplt
import causal4.utils as c4u
import time

from pathlib import Path
root_path = Path(__file__).parents[2]

with open('benchmark10_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

for key, val in pm_causal_set.items():
    val = val.copy()
    val['spk_fname'] = val['spk_fname']
    estimator = Causality.CausalityEstimator(
        **val, n_thread=10)
    _, text = estimator._run_estimation(regen=True, verbose=False, return_log=True)
    wall_time = float(text.splitlines()[-1].split()[3])
    cpu_time = float(text.splitlines()[-1].split()[7])
    print(key, f"Elapsed cpu time: {cpu_time:.2f} s")
    print(key, f"Elapsed time: {wall_time:.2f} s")
    data = estimator.fetch_data(new_run=True)

    # Plot distribution of TE values in log-scale
    data_matched = c4u.match_features(
        data, N=val['N'], conn_file=val['path']/val['conn_file'])
    data_recon, fig_data = c4u.reconstruction_analysis_TE(
        data_matched, nbins=40, hist_range=None, algorithm='EM')
    print(key, np.round(fig_data['acc_gauss']['TE'], 2))

    try:
        fig = mplt.reconstruction_illustration_TE(fig_data)
        plt.tight_layout()
        figname = f"causal_recon_{key}.pdf"
        fig.savefig(root_path/'figures/N10'/figname, transparent=True)
    except:
        print(f"Failed to save {key}")
        continue
    
#%%