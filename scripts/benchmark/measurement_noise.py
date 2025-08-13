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
    axi.set_title(key, fontweight='bold')
    th = bins[np.nonzero(np.cumsum(counts)*(bins[1]-bins[0]) > 0.9)[0][0]]
    axi.axvline(th, color='r', linestyle='--', label='90% line')
    axi.axvline(tmp.mean(), color='g', linestyle='--', label='mean')
    axi.axvline(tmp.mean()+tmp.std(), color='b', linestyle='--', label='std')
    axi.axvline(tmp.mean()-tmp.std(), color='b', linestyle='--')
    print(key, ':', tmp.std())
ax[0,0].legend()
plt.tight_layout()
plt.savefig('PDF_level_0.9.pdf', bbox_inches='tight')
#%%
sigma_dict = {'HHEE': 4, 'HHEI': 4, 'HHconEE': 4, 'HHconEI': 4,
    'Lorenz': 20, 'Lcon': 16, 'Logistic': 0.1, 'RNN': 0.1, 'Rcon': 20}
for key, val in pm_causal_set.items():
    tmp = np.load(val['path'] / get_vfname(val['spk_fname']), mmap_mode='r')
    dt = tmp[1,0] - tmp[0,0]
    std = tmp[:100000, 1:11].flatten().std()
    sigma = sigma_dict[key]
    print(f'{key}: {sigma * np.sqrt(dt):.2f} a.u., std: {std:.2f} a.u., dt: {dt:.2f} s')
# %%
n_subfigures = len(pm_causal_set)
fig, ax = plt.subplots(n_subfigures//2, 2, figsize=(8, 16), gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
for axi, (key, val) in zip(ax.flatten(), pm_causal_set.items()):
    tmp = np.load(val['path'] / get_vfname(val['spk_fname']), mmap_mode='r')[:10000, :11]
    tmp_std = tmp[:,1:].flatten().std()*.2
    counts, bins = np.histogram(tmp, bins=100, density=True)
    axi.plot(tmp[:,0], tmp[:,1], label='data')
    axi.plot(tmp[:,0], tmp[:,1]+np.random.randn(tmp.shape[0])*tmp_std, alpha=0.7, label='noise')
    axi.set_title(key, fontweight='bold')
    axi.set_xlim(0, 100)
    print(key, ':', tmp_std)
# ax[0,0].legend()
plt.tight_layout()
plt.savefig('measurement_noise_std02.pdf', bbox_inches='tight')
# %%
from multiprocessing import Pool
import gc
from causal4.utils import binarize
with open(root_path / 'scripts/benchmark' / 'binarization.yaml', 'r') as binarization_file:
    binarization_cfg = yaml.load(binarization_file, Loader=yaml.FullLoader)
refs = binarization_cfg.get('refractory', {})
thresholds = binarization_cfg.get('threshold', {})
dt= binarization_cfg.get('dt', 0.01)

def binarize_with_noise(key, val, noise_intensity):
    tmp = np.load(val['path'] / get_vfname(val['spk_fname']), mmap_mode='r')[:10000, :11]
    sigma = tmp[:,1:].flatten().std()*noise_intensity
    binarize(val['path']/get_vfname(val['spk_fname']),
        N=int(val['N']), threshold=thresholds[key], T=val['T'],
        verbose=False, ref=refs[key], force_regen=True, sfx=f'noisy{noise_intensity:.1f}',
        preprocess_fn=lambda x: x+np.random.randn(*x.shape)*sigma,)
    gc.collect()

noise_intensities = [0.1, 0.2, 0.3, 0.4]
pool = Pool(processes=len(pm_causal_set), maxtasksperchild=1)
results = [pool.apply_async(binarize_with_noise, args=(key, val, noise_intensity))
           for key, val in pm_causal_set.items()
           for noise_intensity in noise_intensities]
pool.close()
pool.join()
# %%
