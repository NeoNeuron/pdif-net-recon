# %%
# packages for plotting
from figrc import *
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
import pickle
from causal4.utils import binarize

def heatmap(xx, yy, data, ax, vmin=0.5, vmax=1.0, pad=0.005, width=0.01,
            cbar_label='AUC', xlabel='time (ms)', y_label=r'$\mu$ (kHz)', xscale='log'):
    fig = ax.get_figure()
    im = ax.pcolormesh(xx, yy, data, vmin=vmin, vmax=vmax)
    pos = ax.get_position()  # save main axes position BEFORE adding colorbar
    # manually place colorbar axes: [left, bottom, width, height] in figure coords
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
    cbar = fig.colorbar(im, cax=cax, label=cbar_label)
    ax.set_position(pos) 
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel(y_label, fontsize=26)
    ax.set_xscale(xscale)
    ax.set_xlim(xx[0], xx[-1])
    cbar.ax.tick_params(labelsize=18)
    return ax, cbar

# %% path for your time series data
path = Path(root/'data/HH100')
T = 1e7         # length of time series, unit ms
N = 100
dt = 0.5        # time step for descritization unit ms
delay = 3
order = (1,1)

kk = np.arange(1,16).astype(int)
print(kk)
#%%
results_scan_k = []
_fname = f'HHp=0.25s=0.020f=0.080u=0.150_spike_train.dat'
spk_fname, _ = maybe_downsample(path, _fname, 1.0, False)
for i, k in enumerate(kk):
    # Compare with ground truth connectivity
    # Reconstruction based on PDIF values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p=0.250.npy', T, dt=dt, delay=delay,
        order=(k, order[1]), n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=(-12, -4), algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results_scan_k.append(df_fig)

#%%
results_scan_l = []
ll = np.arange(1,16).astype(int)
_fname = f'HHp=0.25s=0.020f=0.080u=0.150_spike_train.dat'
spk_fname, _ = maybe_downsample(path, _fname, 1.0, False)
for i, l in enumerate(ll):
    # Compare with ground truth connectivity
    # Reconstruction based on PDIF values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p=0.250.npy', T, dt=dt, delay=delay,
        order=(order[0], l), n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=(-12, -4), algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results_scan_l.append(df_fig)

# %%
results_scan_dt = []
dtdt = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0])
_fname = f'HHp=0.25s=0.020f=0.080u=0.150_spike_train.dat'
spk_fname, _ = maybe_downsample(path, _fname, 1.0, False)
buff = np.fromfile(spk_fname, dtype=float, count=10000).reshape(-1, 2)
mfr = buff.shape[0] / (buff[:, 0].max() - buff[:, 0].min())*1000/N
event_rate = mfr * dtdt / 1e3
print('event rate:', event_rate)
for i, dt_val in enumerate(dtdt):
    # Compare with ground truth connectivity
    # Reconstruction based on PDIF values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p=0.250.npy', T, dt=dt_val, delay=delay,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=None, algorithm='EM'),
    )
    print(df_fig['acc_gauss']['TE'])
    results_scan_dt.append(df_fig)

#%%
results_scan_delay = []
delays = np.arange(16)
_fname = f'HHp=0.25s=0.020f=0.080u=0.150_spike_train.dat'
spk_fname, _ = maybe_downsample(path, _fname, 1.0, False)
for i, delay_val in enumerate(delays):
    # Compare with ground truth connectivity
    # Reconstruction based on PDIF values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p=0.250.npy', T, dt=dt, delay=delay_val,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=(-12, -2), algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results_scan_delay.append(df_fig)
    try:
        plt.figure()
        plot_pdif_hist(df_fig.loc['TE'], plt.gca())
    except:
        print(1)
#%%
results_scan_delay_l5 = []
delays = np.arange(16)
_fname = f'HHp=0.25s=0.020f=0.080u=0.150_spike_train.dat'
spk_fname, _ = maybe_downsample(path, _fname, 1.0, False)
for i, delay_val in enumerate(delays):
    # Compare with ground truth connectivity
    # Reconstruction based on PDIF values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p=0.250.npy', T, dt=dt, delay=delay_val,
        order=(1,5), n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=(-12, -2), algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results_scan_delay_l5.append(df_fig)
    try:
        plt.figure()
        plot_pdif_hist(df_fig.loc['TE'], plt.gca())
    except:
        print(1)
#%%
results_scan_threshold = []
thresholds = np.arange(-60, -14, 5)
print(thresholds)
vol_fname = f'HHp=0.25s=0.020f=0.080u=0.150_voltage.dat'
for i, th in enumerate(thresholds):
    # load the voltage time series
    spk_fname = binarize(path/vol_fname,
        N=N, threshold=th, T=T, verbose=False,
        ref=10, force_regen=False)
    print(spk_fname)

    # Compare with ground truth connectivity
    # Reconstruction based on PDIF values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p=0.250.npy', T, dt=dt, delay=delay,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=(-12, -4), algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results_scan_threshold.append(df_fig)
    try:
        plt.figure()
        plot_pdif_hist(df_fig.loc['TE'], plt.gca())
    except:
        print(1)
#%%
u = 0.4
_fname = f'HHp=0.25s=0.020f=0.080u={u:.3f}_spike_train.dat'
spk_fname = path/_fname
buff = np.fromfile(spk_fname, dtype=float, count=10000).reshape(-1, 2)
plt.plot(buff[:, 0], buff[:, 1], '|', ms=2)
#%%
results_scan_u = []
uu = np.array([0.05, 0.10, 0.2, 0.25, 0.3])
TT = np.array([1e5, 5e5, 1e6, 5e6, 1e7])
mfr = []
for i, u in enumerate(uu):
    _fname = f'HHp=0.25s=0.020f=0.080u={u:.3f}_spike_train.dat'
    spk_fname, raw_fname = maybe_downsample(path, _fname, 1.0, u>0.25)
    if raw_fname is None:
        raw_fname = spk_fname
    buff = np.fromfile(raw_fname, dtype=float, count=10000).reshape(-1, 2)
    mfr.append(buff.shape[0] / (buff[:, 0].max() - buff[:, 0].min())*1000/N)

    buff = []
    for T_val in TT:
        # Reconstruction based on PDIF values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
        df_recon, df_fig = run_reconstruction_TE(
            path, spk_fname, N, path/f'connect_matrix-p=0.250.npy', T_val, dt=dt, delay=delay,
            order=order, n_thread=128,
            recon_kwargs=dict(nbins=60, hist_range=(-12, -4), algorithm='EM'),
        )
        print(df_fig['auc_svm']['TE'])
        buff.append(df_fig)
    results_scan_u.append(buff)
mfr = np.asarray(mfr)
# %%
scan_specs = [
    ('k', kk, results_scan_k, r'$k$'),
    ('l', ll, results_scan_l, r'$l$'),
    ('dt', dtdt, results_scan_dt, r'$\Delta t$ (ms)'),
    # ('delay', delays, results_scan_delay, r'$\tau$ (ms)'),
    ('delay_l5', delays, results_scan_delay_l5, r'$\tau$ (ms)'),
    ('threshold', thresholds, results_scan_threshold, 'threshold (mV)'),
    ('u', mfr, [item[-1] for item in results_scan_u], 'mean firing rate (Hz)'),
]

fig, axes = plt.subplots(2, 4, figsize=(26, 10), gridspec_kw={'wspace':0.7, 'hspace':0.4})
axes = axes.ravel()
for ax, (name, params, results, xlabel) in zip(axes, scan_specs):
    auc_vals = np.array([res['auc_svm']['TE'] for res in results])
    acc_vals = np.array([res['acc_gauss']['TE'] for res in results])

    ax.plot(params, auc_vals, 'o-', color=ORANGE, clip_on=False, ms=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel('AUC', fontsize=26, color=ORANGE)
    ax.tick_params(axis='y', labelcolor=ORANGE)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(params, acc_vals, 'o-', color=GREEN, clip_on=False, ms=8)
    ax2.set_ylim(0.5, 1.0)
    ax2.set_ylabel('accuracy', fontsize=26, color=GREEN)
    ax2.tick_params(axis='y', labelcolor=GREEN)
    ax2.spines['left'].set_color(ORANGE)
    ax2.spines['right'].set_color(GREEN)
    ax2.spines['right'].set_visible(True)
    if name == 'dt':
        ax.set_xscale('log')
        ax.set_xlim(params[0], params[-1])
        ax_top = ax.twiny()
        ax_top.spines['top'].set_visible(True)
        ax_top.set_xscale('log')
        ax_top.set_xlim(event_rate[0], event_rate[-1])
        # ax_top.set_xticks(params[::2])
        # ax_top.set_xticklabels([f'{er:.2g}' for er in event_rate[::2]], fontsize=14)
        ax_top.set_xlabel(r'event rate [$p(y_n^{(1)}=1)$]', fontsize=26, y=1.4)
        ax_top.tick_params(axis='x', top=True, bottom=False)

    auc_vals = np.array([[res['auc_svm']['TE'] for res in results] for results in results_scan_u])
    acc_vals = np.array([[res['acc_gauss']['TE'] for res in results] for results in results_scan_u])

ax_auc, ax_acc = axes[-2:]
_, cbar = heatmap(TT, uu, auc_vals, ax_auc)
_, cbar = heatmap(TT, uu, acc_vals, ax_acc, cbar_label='accuracy')
ax_auc.set_rasterized(True)
ax_acc.set_rasterized(True)

axes[0].set_xlim(0,15)
axes[1].set_xlim(0,15)
axes[3].set_xlim(0,15)
axes[5].set_xlim(0,40)

for tag, axi in zip('ABCDEFGH', axes.flatten()):
    fig.text(-0.3, 1.18, tag, fontsize=35, va='top', transform=axi.transAxes)

fig.savefig(root/'fig_nc/pdf'/'figR4-5.pdf', transparent=True, bbox_inches='tight')

#%%