# %%
# packages for plotting
from figrc import *
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
from matplotlib.ticker import MaxNLocator
import pickle
# %% path for your time series data
path = Path(root/'data/connnet')
T = 1e7         # length of time series, unit ms
dt = 0.5        # time step for descritization unit ms
delay = 3
order = (1,1)

Nlist = np.array([178, 152, 609], dtype=int)
results = []

ra = 1.0
for i, N in enumerate(Nlist):
    if i == 1:
        _fname = f'HHp=0.{i+1:d}0s=0.020f=0.080u=0.080_spike_train.dat'
    else:
        _fname = f'HHp=0.{i+1:d}0s=0.020f=0.080u=0.150_spike_train.dat'
    spk_fname, _ = maybe_downsample(path, _fname, ra, i == 1)

    # Compare with ground truth connectivity
    # ! important note: the 'connect_matrix-p=0.250.npy' file should be a binary adjacency matrix with shape (N, N), with W_{ij} representing the connection from node i to node j.
    # Reconstruction based on PTE-TE values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p=0.{i+1:d}00.npy', T, dt=dt, delay=delay,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=(-12, -4), algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'], df_fig['acc_gauss']['TE'])
    results.append(df_fig)

# %%
fig, ax = plt.subplots(3,3,figsize=(18,15), gridspec_kw={'wspace':0.4, 'hspace':0.4})
for i, (N, result, axi) in enumerate(zip(Nlist, results, ax.T)):
    conn = np.load(path/f'connect_matrix-p=0.{i+1:d}00.npy')
    plot_conn_matrix(axi[0], conn, title=f'Connectivity: {N} neurons')
    # axi[0].tick_params(axis='both', labelsize=16)
    if i == 1:
        _fname = f'HHp=0.{i+1:d}0s=0.020f=0.080u=0.080_spike_train.dat'
    else:
        _fname = f'HHp=0.{i+1:d}0s=0.020f=0.080u=0.150_spike_train.dat'
    spk_fname, raw_fname = maybe_downsample(path, _fname, ra, i == 1)
    spk_all = np.fromfile(raw_fname, dtype=float, count=20000).reshape(-1, 2) if raw_fname is not None else None
    spk = np.fromfile(spk_fname, dtype=float, count=20000).reshape(-1, 2)
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
