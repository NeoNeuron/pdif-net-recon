# %%
# packages for plotting
from figrc import *
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# %%
def figure_balance_saved(path, pfx, ax, spike_blank_ms=4.0):
    """Example-neuron E/I input currents from an EXISTING run_balanced_EINet.py
    output. Pure plotting: only reads .dat files already on disk, never calls
    simHH (contrast figure_balance, which simulates its own trace).

    run_balanced_EINet's default `--record-currents` run does not save
    voltage.dat (only `--record-v` does), so there is no membrane trace to
    threshold a spike out of. Spikes are instead blanked directly from the
    recorded spike train: +-spike_blank_ms around each spike time of the
    example cell. This cell's own recorded I_E/I_I already carry its action
    potential's driving-force swing (I_E=-(g_f+g_se)(v-V_E) etc. use this
    cell's own v), which decays over ~3-4 ms here -- checked empirically
    against the raw trace -- hence the wider default than the 2 ms voltage-
    based dilation figure_balance uses. Net current also omits the leak term
    for the same reason (I_L = G_L*(E_L-v) needs v).
    """
    ie = np.fromfile(path / (pfx + "_IE.dat"), dtype=float)
    ie = ie[: (ie.size // (N + 1)) * (N + 1)].reshape(-1, N + 1)
    ii = np.fromfile(path / (pfx + "_II.dat"), dtype=float)
    ii = ii[: (ii.size // (N + 1)) * (N + 1)].reshape(-1, N + 1)
    n = min(len(ie), len(ii))
    ie, ii = ie[:n], ii[:n]
    t = ie[:, 0]
    t_range = (float(t[0]), float(t[-1]))
    # spk = load_spikes(data_dir, t_range)
    spk_data = np.fromfile(path / (pfx + "_spike_train.dat"), dtype=float, count=100000).reshape(-1, 2)
    NE, NI = 320, 80
    spk = spk_data[(spk_data[:, 0] >= t_range[0]) * (spk_data[:, 0] <= t_range[1])]

    # example neuron: the excitatory cell whose spike count is closest to the
    # E-population mean, same convention as figure_balance
    counts = np.bincount(spk[:, 1].astype(int), minlength=N)
    cell = int(np.argmin(np.abs(counts[:NE] - counts[:NE].mean())))
    I_E, I_I = ie[:, 1 + cell], ii[:, 1 + cell]
    cell_spk = spk[spk[:, 1] == cell, 0]

    dt = float(t[1] - t[0])
    w = int(round(spike_blank_ms / dt))
    mask = np.ones(n, dtype=bool)
    for i in np.searchsorted(t, cell_spk):
        mask[max(0, i - w):min(n, i + w + 1)] = False
    I_E, I_I = np.where(mask, I_E, np.nan), np.where(mask, I_I, np.nan)
    net = I_E + I_I

    ax.plot(t, I_E, lw=0.8, color="#c0392b", label=r"$I_E$")
    ax.plot(t, I_I, lw=0.8, color="#2471a3", label=r"$I_I$")
    ax.plot(t, net, lw=0.8, color="k", alpha=0.75, label=r"$I_E{+}I_I$")
    ax.axhline(np.nanmean(I_E), color="#c0392b", ls="--", lw=0.7)
    ax.axhline(np.nanmean(I_I), color="#2471a3", ls="--", lw=0.7)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel(r"current ($\mu$A/cm$^2$)")
    ax.legend(loc="upper right", ncol=1, fontsize=12, framealpha=0.9)
    # ax.set_title(f"E neuron #{cell}, from {path} (spikes blanked "
    #              f"+-{spike_blank_ms:g} ms, no leak term -- no voltage.dat "
    #              f"in this run)", fontsize=10)
    return ax
# %%
T = 5e6         # length of time series, unit ms
N = 400           # total number of nodes in the network
Ne=320
Ni=80
# ref = 5.0       # refractory period in thresholding to generate surrogated spikes, unit ms
# th = 10         # threshold for generating surrogated spikes
dt = 0.5        # time step for descritization unit ms
path = Path(root/'data/EINet/balanced')
spk_fname = Path(root/'data/EINet/balanced/HHp=0.25s=0.020s=0.020f=0.420u=0.500_spike_train.dat')

delay = 1.5
order = (1,1)
dt = 0.5
estimator = CausalityEstimator(
    path, spk_fname, N, delay=delay, T=T, dt=dt,
    n_thread=128, order=order,
)
# fetch computed PTD-TE values
# estimator._run_estimation(delay=1.5, regen=True)
# Compare with ground truth connectivity
data = estimator.fetch_data(delay=1.5, new_run=True)
data_matched = match_features(data, N=Ne, conn_file=path/'connect_matrix-p=0.250.npy', Ni=Ni)
#%%
# ! important note: the 'connect_matrix-p=0.250.npy' file should be a binary adjacency matrix with shape (N, N), with W_{ij} representing the connection from node i to node j.
# Reconstruction based on PTE-TE values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
df_recon, df_fig = reconstruction_analysis_TE(
    data_matched, nbins=60, hist_range=(-10,-4), fit_p0=(0.5, -5.5, -4.6, 1e-1, 1e-1), algorithm='EM')
print(df_fig['auc_svm']['TE'])

# %%
# Search of optimal delay parameter
fig, ax = plt.subplots(1,3,figsize=(18,4.5), gridspec_kw={'wspace':0.5})

spk_data = np.fromfile(spk_fname, dtype=float, offset=8*8000, count=10000).reshape(-1, 2)
masks = [(spk_data[:, 0] > 1000) * (spk_data[:, 1] < 320),
         (spk_data[:, 0] > 1000) * (spk_data[:, 1] >= 320)]
for m, c, label in zip(masks, ("#c0392b", "#2471a3"), ('E', 'I')):
    ax[0].plot(spk_data[m, 0], spk_data[m, 1], '|', lw=0.5, color=c, alpha=0.5, label=label)
ax[0].set_xlabel('time (ms)', fontsize=26)
ax[0].set_ylabel('node index', fontsize=26)
ax[0].set_xlim(1000, 1300)
ax[0].set_ylim(0, 400)
ax[0].legend(loc='upper right', fontsize=20, framealpha=0.9)

ax[1] = figure_balance_saved(path, 'HHp=0.25s=0.020s=0.020f=0.420u=0.500', ax[1], spike_blank_ms=4.0)
ax[1].set_xlim(1000, 1300)
ax[1].set_ylim(-100, 100)
ax[1].set_xlabel('time (ms)', fontsize=26)
tmp = df_fig.loc['TE']
plot_pdif_hist(tmp, ax[2])
ax[2].set_xlim(-10, -4)

for i, tag in enumerate('ABC'):
    fig.text(0.06+i*0.30, 0.98, tag, fontsize=35, va='top')

fig.savefig(root/'fig_nc/pdf'/'figR1-2.pdf', transparent=True, bbox_inches='tight')
# %%
