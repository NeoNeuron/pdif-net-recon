# %%
# packages for plotting
from figrc import *
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
from causal4.Causality import CausalityEstimator
from causal4.utils import binarize, match_features, reconstruction_analysis_TE
from matplotlib.ticker import FuncFormatter, MaxNLocator
@FuncFormatter
def sci_formatter(x, pos):
    return r'$10^{%d}$'%x
import pickle
from causal4.downsample import downsample
# %% path for your time series data
path = Path(root/'data/connnet')
T = 1e7         # length of time series, unit ms
dt = 0.5        # time step for descritization unit ms
delay = 3
order = (1,1)

Nlist = np.array([178, 152, 609], dtype=int)
results = []

for i, N in enumerate(Nlist):
    if i == 1:
        ra = 1.0
        # spk_fname = Path(path/f'HHp=0.{i+1:d}0s=0.020f=0.090u=0.060_spike_train.dat')
        _fname  = f'HHp=0.{i+1:d}0s=0.020f=0.080u=0.080_spike_train.dat'
        _ofname  = f'HHp=0.{i+1:d}0s=0.020f=0.080u=0.080_ra={ra:.2f}_spike_train.dat'
        downsample(_fname, _ofname, path, ra)
        spk_fname = Path(path/_ofname)
    else:
        spk_fname = Path(path/f'HHp=0.{i+1:d}0s=0.020f=0.080u=0.150_spike_train.dat')
    estimator = CausalityEstimator(
        path, spk_fname, N, delay=delay, T=T, dt=dt,
        n_thread=128, order=order,
    )
    # Search of optimal delay parameter
    # optimal_m = estimator.get_optimal_delay(np.arange(20)*0.2, mode=0)
    # print(optimal_m)
    # fetch computed PTD-TE values
    # estimator._run_estimation(delay=3, regen=True)
    data = estimator.fetch_data(new_run=True)

    # Compare with ground truth connectivity
    data_matched = match_features(data, N, path/f'connect_matrix-p=0.{i+1:d}00.npy')
    # ! important note: the 'connect_matrix-p=0.250.npy' file should be a binary adjacency matrix with shape (N, N), with W_{ij} representing the connection from node i to node j.
    # Reconstruction based on PTE-TE values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=60, hist_range=(-12, -4), algorithm='EM')
    print(df_fig['auc_svm']['TE'])
    results.append(df_fig)

# %%
fig, ax = plt.subplots(3,3,figsize=(18,15), gridspec_kw={'wspace':0.4, 'hspace':0.4})
for i, (N, result, axi) in enumerate(zip(Nlist, results, ax.T)):
    conn = np.load(path/f'connect_matrix-p=0.{i+1:d}00.npy')
    plot_conn_matrix(axi[0], conn, title=f'Connectivity: {N} neurons')
    # axi[0].tick_params(axis='both', labelsize=16)
    if i == 1:
        _fname  = f'HHp=0.{i+1:d}0s=0.020f=0.080u=0.080_spike_train.dat'
        spk_all = np.fromfile(path / _fname, dtype=float, count=20000).reshape(-1, 2)
        _ofname  = f'HHp=0.{i+1:d}0s=0.020f=0.080u=0.080_ra={ra:.2f}_spike_train.dat'
        spk = np.fromfile(path / _ofname, dtype=float, count=20000).reshape(-1, 2)
    else:
        spk_all = None
        spk = np.fromfile(path/f'HHp=0.{i+1:d}0s=0.020f=0.080u=0.150_spike_train.dat',
                          dtype=float, count=20000).reshape(-1, 2)
    plot_spike_train(axi[1], N, spk, spk_all=spk_all, title=f'mean firing rate: {spk.shape[0] / (spk[:, 0].max() - spk[:, 0].min())*1000/N:.2f} Hz')
    axi[1].set_xlim(600,800)
    tmp = result.loc['TE']
    plot_pdif_hist(tmp, axi[2])

for tag, axi in zip('ABCDEFGHI', ax.T.flatten()):
    fig.text(-0.25, 1.18, tag, fontsize=35, va='top', transform=axi.transAxes)

fig.savefig(root/'fig_nc/pdf'/'figR4-1.pdf', transparent=True, bbox_inches='tight')

#%% preprocess real connectome data
path = Path(root/'data/connnet')
fnames = ['adjacency_drosophilaCx_pre_post.npy',
          'adjacency_larva_pre_post.npy',
          'adjacency_zebrafish_pre_post.npy']
for fname in fnames:
    tmp = np.load(path / fname)
    print(tmp[:10,:10])
    tmp = tmp > 0
    # print(tmp.dtype)
    plt.figure()
    plt.imshow(tmp.astype(float), cmap='viridis', aspect='auto')
    plt.colorbar()
    np.save(path / fname, tmp.astype(float)*0.02)
# %%
