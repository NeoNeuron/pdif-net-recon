#%%
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import causal4.Causality as Causality
import causal4.myplot as mplt
import causal4.utils as c4u

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
        **val, n_thread=int(os.cpu_count()/4))
    # estimator._run_estimation(regen=True, verbose=True)
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