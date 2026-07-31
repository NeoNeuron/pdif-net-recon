# %%
# packages for plotting
from figrc import *
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
from causal4.Causality import CausalityEstimator
from causal4.utils import binarize, match_features, reconstruction_analysis_TE
from matplotlib.ticker import FuncFormatter
@FuncFormatter
def sci_formatter(x, pos):
    return r'$10^{%d}$'%x
import pickle
from causal4.downsample import downsample

# %% path for your time series data
path = Path(root/'data/HH100')
T = 1e7         # length of time series, unit ms
N = 100
dt = 0.5        # time step for descritization unit ms
delay = 3
order = (1,1)

ratios = np.array([0.2, 0.25, 0.3, 0.5, 0.7, 0.9])
uu = np.array([0.15, 0.15, 0.15, 0.08, 0.06, 0.04])
results = []
for i, (s, u) in enumerate(zip(ratios, uu)):
    if u < 0.15:
        ra=1.0
        _fname  = f'HHp={s:.2f}s=0.020f=0.080u={u:.3f}_spike_train.dat'
        _ofname  = f'HHp={s:.2f}s=0.020f=0.080u={u:.3f}_ra={ra:.2f}_spike_train.dat'
        downsample(_fname, _ofname, path, ra)
        spk_fname = Path(path/_ofname)
    else:
        spk_fname = Path(path/f'HHp={s:.2f}s=0.020f=0.080u={u:.3f}_spike_train.dat')
    estimator = CausalityEstimator(
        path, spk_fname, N, delay=delay, T=T, dt=dt,
        n_thread=128, order=order,
    )
    # fetch computed PTD-TE values
    # estimator._run_estimation(delay=0, regen=True)
    data = estimator.fetch_data(new_run=True)

    # Compare with ground truth connectivity
    data_matched = match_features(data, N, path/f'connect_matrix-p={s:.3f}.npy')
    # ! important note: the 'connect_matrix-p=0.250.npy' file should be a binary adjacency matrix with shape (N, N), with W_{ij} representing the connection from node i to node j.
    # Reconstruction based on PTE-TE values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=60, hist_range=(-12, -4), algorithm='EM')
    print(df_fig['auc_svm']['TE'])
    results.append(df_fig)

# %%

fig, ax = plt.subplots(3,6,figsize=(32,15), gridspec_kw={'wspace':0.55, 'hspace':0.6})
for i, (s, u, res, axi) in enumerate(zip(ratios, uu, results, ax.T)):
    conn = np.load(path/f'connect_matrix-p={s:.3f}.npy')
    plot_conn_matrix(axi[0], conn)
    axi[0].set_title(f'p={s*100:.0f}%', fontsize=22, pad=10)
    if u < 0.15:
        ra=1.0
        _fname  = f'HHp={s:.2f}s=0.020f=0.080u={u:.3f}_spike_train.dat'
        spk_all = np.fromfile(path / _fname, dtype=float, count=20000).reshape(-1, 2)
        _ofname  = f'HHp={s:.2f}s=0.020f=0.080u={u:.3f}_ra={ra:.2f}_spike_train.dat'
        spk_fname = Path(path/_ofname)
    else:
        spk_all=None
        spk_fname = Path(path/f'HHp={s:.2f}s=0.020f=0.080u={u:.3f}_spike_train.dat')
    spk = np.fromfile(spk_fname, dtype=float, count=10000).reshape(-1, 2)
    plot_spike_train(axi[1], N, spk, spk_all)
    axi[1].set_title(f'mean firing rate: {spk.shape[0] / (spk[:, 0].max() - spk[:, 0].min())*1000/N:.2f} Hz \n' +
                     fr'($\nu$={u:.2f})', fontsize=22, pad=0)
    axi[1].set_xlim(500,900)
    tmp = res.loc['TE']
    plot_pdif_hist(tmp, axi[2])

for tag, axi in zip('ABC', ax[:,0]):
    fig.text(-0.5, 1.18, tag, fontsize=35, va='top', transform=axi.transAxes)

fig.savefig(root/'fig_nc/pdf'/'figR4-2.pdf', transparent=True, bbox_inches='tight')

#%%
ratios = np.array([0.3, 0.35, 0.4, 0.5, 0.6])
results_common_Poisson = []
results_common_Poisson_ra = []
ra=1.0
for i, r in enumerate(ratios):
    _fname = f'P{r:.2f}HHp=0.25s=0.020f=0.080u=0.150_spike_train.dat'
    _ofname = f'P{r:.2f}HHp=0.25s=0.020f=0.080u=0.150_ra={ra:.2f}_spike_train.dat'
    downsample(_fname, _ofname, path, ra)
    spk_fname = path / _fname
    spk_fname_ra = path / _ofname

    estimator = CausalityEstimator(
        path, spk_fname, N, delay=delay, T=T, dt=dt,
        n_thread=128, order=order,
    )
    # fetch computed PTD-TE values
    data = estimator.fetch_data(new_run=True)

    # Compare with ground truth connectivity
    data_matched = match_features(data, N, path/f'connect_matrix-p=0.250.npy')
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=60, hist_range=None, algorithm='EM')
    print(df_fig['auc_svm']['TE'])
    results_common_Poisson.append(df_fig)

    estimator = CausalityEstimator(
        path, spk_fname_ra, N, delay=delay, T=T, dt=dt,
        n_thread=128, order=order,
    )
    # fetch computed PTD-TE values
    data = estimator.fetch_data(new_run=True)

    # Compare with ground truth connectivity
    data_matched = match_features(data, N, path/f'connect_matrix-p=0.250.npy')
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=60, hist_range=None, algorithm='EM')
    print(df_fig['auc_svm']['TE'])
    results_common_Poisson_ra.append(df_fig)

#%%
fig, ax = plt.subplots(3,len(ratios),figsize=(25,15), gridspec_kw={'wspace':0.5, 'hspace':0.5})
for i, (r, res, res_ra, axi) in enumerate(zip(ratios, results_common_Poisson, results_common_Poisson_ra, ax.T)):
    _fname = f'P{r:.2f}HHp=0.25s=0.020f=0.080u=0.150_spike_train.dat'
    _ofname = f'P{r:.2f}HHp=0.25s=0.020f=0.080u=0.150_ra={ra:.2f}_spike_train.dat'
    spk_all = np.fromfile(path / _fname, dtype=float, count=10000).reshape(-1, 2)
    spk = np.fromfile(path / _ofname, dtype=float, count=10000).reshape(-1, 2)
    plot_spike_train(axi[0], N, spk, spk_all)
    axi[0].set_title(f'mean firing rate: {spk.shape[0] / (spk[:, 0].max() - spk[:, 0].min())*1000/N:.2f} Hz \n (ratio={r*100:.0f}%)', fontsize=22, pad=10)
    tmp = res.loc['TE']
    plot_pdif_hist(tmp, axi[1])
    tmp = res_ra.loc['TE']
    plot_pdif_hist(tmp, axi[2])

for tag, axi in zip('ABC', ax[:,0]):
    fig.text(-0.4, 1.18, tag, fontsize=35, va='top', transform=axi.transAxes)

fig.savefig(root/'fig_nc/pdf'/'figR4-3.pdf', transparent=True, bbox_inches='tight')

#%%
strengths = np.array([0.02, 0.03, 0.04, 0.05])
uu = np.array([0.15, 0.08, 0.05, 0.03])
results_vary_strength = []
results_vary_strength_ra = []
ra=1.0
for s, u in zip(strengths, uu):
    spk_fname = Path(path/f'HHp=0.25s={s:.3f}f=0.080u={u:.3f}_spike_train.dat')
    _fname = f'HHp=0.25s={s:.3f}f=0.080u={u:.3f}_spike_train.dat'
    _ofname = f'HHp=0.25s={s:.3f}f=0.080u={u:.3f}_ra={ra:.2f}_spike_train.dat'
    downsample(_fname, _ofname, path, ra)
    spk_fname = path / _fname
    spk_fname_ra = path / _ofname
    estimator = CausalityEstimator(
        path, spk_fname, N, delay=delay, T=T, dt=dt,
        n_thread=128, order=order,
    )
    # fetch computed PTD-TE values
    # estimator._run_estimation(delay=0, regen=True)
    data = estimator.fetch_data(new_run=True)

    # Compare with ground truth connectivity
    data_matched = match_features(data, N, path/f'connect_matrix-p=0.250.npy')
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=60, hist_range=None, algorithm='EM')
    print(df_fig['auc_svm']['TE'])
    results_vary_strength.append(df_fig)

    estimator = CausalityEstimator(
        path, spk_fname_ra, N, delay=delay, T=T, dt=dt,
        n_thread=128, order=order,
    )
    # fetch computed PTD-TE values
    # estimator._run_estimation(delay=0, regen=True)
    data = estimator.fetch_data(new_run=True)

    # Compare with ground truth connectivity
    data_matched = match_features(data, N, path/f'connect_matrix-p=0.250.npy')
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=60, hist_range=None, algorithm='EM')
    print(df_fig['auc_svm']['TE'])
    results_vary_strength_ra.append(df_fig)

#%%
fig, ax = plt.subplots(3,len(strengths),figsize=(20,15), gridspec_kw={'wspace':0.4, 'hspace':0.4})
for i, (s, u, res, res_ra, axi) in enumerate(zip(strengths, uu, results_vary_strength, results_vary_strength_ra, ax.T)):
    _fname = f'HHp=0.25s={s:.3f}f=0.080u={u:.3f}_spike_train.dat'
    _ofname = f'HHp=0.25s={s:.3f}f=0.080u={u:.3f}_ra={ra:.2f}_spike_train.dat'
    spk_all = np.fromfile(path / _fname, dtype=float, count=10000).reshape(-1, 2)
    spk = np.fromfile(path / _ofname, dtype=float, count=10000).reshape(-1, 2)
    plot_spike_train(axi[0], N, spk, spk_all)
    axi[0].set_title(f'mean firing rate: {spk.shape[0] / (spk[:, 0].max() - spk[:, 0].min())*1000/N:.2f} Hz \n' + fr'($\nu$={u:.2f})', fontsize=22, pad=10)
    axi[0].set_xlim(500,900)
    tmp = res.loc['TE']
    plot_pdif_hist(tmp, axi[1])
    tmp = res_ra.loc['TE']
    plot_pdif_hist(tmp, axi[2])

for tag, axi in zip('ABC', ax[:,0]):
    fig.text(-0.4, 1.18, tag, fontsize=35, va='top', transform=axi.transAxes)

fig.savefig(root/'fig_nc/pdf'/'figR4-4.pdf', transparent=True, bbox_inches='tight')

#%%
