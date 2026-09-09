# %%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
import yaml
import matplotlib.pyplot as plt
plt.rcParams['font.size']=16
import seaborn as sns
import pickle as pkl
from utils import get_vfname, get_spk_fname

with open(Path(__file__).resolve().parent / 'benchmark_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

#%%
n_subfigures = len(pm_causal_set)
fig, ax = plt.subplots(2, n_subfigures//2, figsize=(16, 8), gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
for axi, (key, val) in zip(ax.flatten(), pm_causal_set.items()):
    tmp = np.load(val['path'] / get_vfname(val['spk_fname']), mmap_mode='r')
    tmp = tmp[:100000, 1:11].flatten()
    counts, bins = np.histogram(tmp, bins=100, density=True)
    axi.plot(bins[1:], counts)
    axi.fill_between(bins[1:], counts, alpha=0.3)
    axi.set_title(key, fontweight='bold')
    th = bins[np.nonzero(np.cumsum(counts)*(bins[1]-bins[0]) > 0.9)[0][0]]
    axi.axvline(th, color='r', linestyle='--', label='90% line')
    axi.axvline(tmp.mean(), color='g', linestyle='--', label='mean')
    axi.axvline(tmp.mean()+tmp.std(), color='b', linestyle='--', label='std')
    axi.axvline(tmp.mean()-tmp.std(), color='b', linestyle='--')
    axi.set_ylim(0)
    print(key, ':', tmp.std())
ax[0,0].legend()
plt.tight_layout()
plt.savefig('PDF_level_0.9.pdf', bbox_inches='tight')
#%%
from multiprocessing import Pool
import gc
from pdif.utils import binarize
with open(root_path / 'scripts/benchmark' / 'binarization.yaml', 'r') as binarization_file:
    binarization_cfg = yaml.load(binarization_file, Loader=yaml.FullLoader)
refs = binarization_cfg.get('refractory', {})
thresholds = binarization_cfg.get('threshold', {})
dt= binarization_cfg.get('dt', 0.01)

def binarize_with_noise(key, val, threshold):
    binarize(val['path']/get_vfname(val['spk_fname']),
        N=int(val['N']), threshold=threshold, T=val['T'],
        verbose=False, ref=refs[key], force_regen=False,
    )
    gc.collect()

thresholds = np.arange(-0.5, 0.51, 0.05)
pool = Pool(processes=len(pm_causal_set), maxtasksperchild=1)
results = [pool.apply_async(binarize_with_noise, args=('RNN', pm_causal_set['RNN'], threshold))
           for threshold in thresholds]
pool.close()
pool.join()
# %%
thresholds = np.arange(-20, 20.1, 2.5)
pool = Pool(processes=len(pm_causal_set), maxtasksperchild=1)
results = [pool.apply_async(binarize_with_noise, args=('Rcon', pm_causal_set['Rcon'], threshold))
           for threshold in thresholds]
pool.close()
pool.join()
#%%