#%%
import os
import yaml
import numpy as np
import causal4.Causality as Causality
import causal4.myplot as mplt
import causal4.utils as c4u
from pathlib import Path
root_path = Path(__file__).parents[2]

with open('benchmark10_subnet_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

#%%
refs=[3.0]*6+[0.5]*2+[0.0, 3.0]
thresholds = [-50, -50, -50, -50, -50, -50, 10, 10, 0.9, 0.02]

for th, ref, (key, val) in zip(thresholds, refs, pm_causal_set.items()):
    val = val.copy()
    if key == 'Gaussian':
        val['spk_fname'] = val['spk_fname'].replace('th=0.020','')
    val['spk_fname'] = val['spk_fname']+ f'_th={th:.2f}ref={ref:.2f}'
    estimator = Causality.CausalityEstimator(
        **val, n_thread=int(os.cpu_count()/4))
    # estimator.delay=16
    # estimator.get_optimal_delay(np.arange(20))
    # estimator._run_estimation(regen=True, verbose=True)
    # estimator.order = (1,1)
    data = estimator.fetch_data(new_run=True)

    # % Plot distribution of TE values in log-scale
    data_matched = c4u.match_features(
        data, N=val['N'], conn_file=val['path']/val['conn_file'])
    data_recon, fig_data = c4u.reconstruction_analysis_TE(
        data_matched, nbins=40, hist_range=None, algorithm='EM')
    print(key, np.round(fig_data['acc_gauss']['TE'],2))

    try:
        fig = mplt.reconstruction_illustration_TE(fig_data)
        fig.savefig(val['path']/f"causal_recon_{key}.pdf", transparent=True)
    except:
        print(f"Failed to save {key}")
        continue

#%%