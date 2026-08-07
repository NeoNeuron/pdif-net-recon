# %%
# packages for plotting
from figrc import *
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
import pickle

# %% path for your time series data
path = Path(root/'data/HH100')
T = 1e7         # length of time series, unit ms
N = 100
dt = 0.5        # time step for descritization unit ms
delay = 3
order = (1,1)

ratios = np.array([0.2, 0.25, 0.3, 0.5, 0.7, 0.9])
uu = np.array([0.15, 0.15, 0.15, 0.08, 0.06, 0.04])
results_ds, results = [], []
for i, (s, u) in enumerate(zip(ratios, uu)):
    ra = 1.0
    _fname = f'HHp={s:.2f}s=0.020f=0.080u={u:.3f}_spike_train.dat'
    spk_fname, _ = maybe_downsample(path, _fname, ra, u < 0.15)
    spk_fname_all, _ = maybe_downsample(path, _fname, ra, False)

    # Compare with ground truth connectivity
    # Reconstruction based on PDIF values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p={s:.3f}.npy', T, dt=dt, delay=delay,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=(-12, -4), algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results_ds.append(df_fig)

    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname_all, N, path/f'connect_matrix-p={s:.3f}.npy', T, dt=dt, delay=delay,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=(-12, -4), algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results.append(df_fig)

# %%

fig = plt.figure(figsize=(32, 19))
gs = fig.add_gridspec(4, 6, wspace=0.55, hspace=0.6)
ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(6)] for r in range(3)])
ax_confounder = fig.add_subplot(gs[3, :2])
ax_chain = fig.add_subplot(gs[3, 2:4])
ax_auc = fig.add_subplot(gs[3, 4])
ax_acc = fig.add_subplot(gs[3, 5])

confounder_counts_list, chain_counts_list = [], []
auc_list, acc_list = [], []
auc_list_all, acc_list_all = [], []
for i, (s, u, res, res_all, axi) in enumerate(zip(ratios, uu, results_ds, results, ax.T)):
    conn = np.load(path/f'connect_matrix-p={s:.3f}.npy')
    plot_conn_matrix(axi[0], conn)
    axi[0].set_title(f'p={s*100:.0f}%', fontsize=22, pad=10)
    confounder_counts, chain_counts = motif_path_counts(conn)
    confounder_counts_list.append(confounder_counts)
    chain_counts_list.append(chain_counts)

    ra = 1.0
    _fname = f'HHp={s:.2f}s=0.020f=0.080u={u:.3f}_spike_train.dat'
    spk_fname, raw_fname = maybe_downsample(path, _fname, ra, u < 0.15)
    if raw_fname is not None:
        spk_all = np.fromfile(raw_fname, dtype=float, count=20000).reshape(-1, 2)
    else:
        spk_all = None
    spk = np.fromfile(spk_fname, dtype=float, count=10000).reshape(-1, 2)
    auc_list.append(res['auc_svm']['TE'])
    acc_list.append(res['acc_gauss']['TE'])
    auc_list_all.append(res_all['auc_svm']['TE'])
    acc_list_all.append(res_all['acc_gauss']['TE'])
    plot_spike_train(axi[1], N, spk, spk_all)
    axi[1].set_title(f'mean firing rate: {spk.shape[0] / (spk[:, 0].max() - spk[:, 0].min())*1000/N:.2f} Hz \n' +
                     fr'($\nu$={u:.2f})', fontsize=22, pad=0)
    axi[1].set_xlim(500,900)
    tmp = res.loc['TE']
    plot_pdif_hist(tmp, axi[2])

xtick_labels = [f'{s*100:.0f}%' for s in ratios]
positions = np.arange(len(ratios))

parts = ax_confounder.violinplot(confounder_counts_list,
                                 positions=positions, 
                                 showmeans=True,
                                 showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor(GREEN)
    pc.set_edgecolor(GREEN)
    pc.set_alpha(0.6)
for partname in ('cbars','cmins','cmaxes','cmeans'):
    vp = parts[partname]
    vp.set_edgecolor(GREEN)
    vp.set_linewidth(1)
ax_confounder.set_xticks(positions)
ax_confounder.set_xticklabels(xtick_labels)
# ax_confounder.set_yscale('symlog')
ax_confounder.set_xlabel('connection density', fontsize=26)
ax_confounder.set_ylabel('# confounder per\nunconnected pair', fontsize=22)

parts = ax_chain.violinplot(chain_counts_list, positions=positions, showmeans=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor(GREEN)
    pc.set_edgecolor(GREEN)
    pc.set_alpha(0.6)
for partname in ('cbars','cmins','cmaxes','cmeans'):
    vp = parts[partname]
    vp.set_edgecolor(GREEN)
    vp.set_linewidth(1)
ax_chain.set_xticks(positions)
ax_chain.set_xticklabels(xtick_labels)
# ax_chain.set_yscale('symlog')
ax_chain.set_xlabel('connection density', fontsize=26)
ax_chain.set_ylabel('# chain per\nunconnected pair', fontsize=22)

ax_auc.plot(ratios*100, auc_list, 'o-', color='C0', clip_on=False, ms=10, label='raw')
ax_auc.plot(ratios*100, auc_list_all, 'o-', color='C1', clip_on=False, ms=8, label='downsampled')
ax_auc.set_ylim(0.5, 1.0)
ax_auc.set_xlabel('connection density (%)', fontsize=26)
ax_auc.set_ylabel('AUC', fontsize=26)
ax_auc.legend(fontsize=16, loc='lower left')
ax_auc.grid(color='gray', alpha=0.3, linestyle='--')

ax_acc.plot(ratios*100, acc_list, 'o-', color='C0', clip_on=False, ms=10, label='raw')
ax_acc.plot(ratios*100, acc_list_all, 'o-', color='C1', clip_on=False, ms=8, label='downsampled')
ax_acc.set_ylim(0.5, 1.0)
ax_acc.set_xlabel('connection density (%)', fontsize=26)
ax_acc.set_ylabel('accuracy', fontsize=26)
ax_acc.legend(fontsize=16, loc='lower left')
ax_acc.grid(color='gray', alpha=0.3, linestyle='--')

for tag, axi in zip('ABC', ax[:,0]):
    fig.text(-0.5, 1.18, tag, fontsize=35, va='top', transform=axi.transAxes)
fig.text(-0.20, 1.18, 'D', fontsize=35, va='top', transform=ax_confounder.transAxes)
fig.text(-0.20, 1.18, 'E', fontsize=35, va='top', transform=ax_chain.transAxes)
fig.text(-0.40, 1.18, 'F', fontsize=35, va='top', transform=ax_auc.transAxes)
fig.text(-0.40, 1.18, 'G', fontsize=35, va='top', transform=ax_acc.transAxes)

fig.savefig(root/'fig_nc/pdf'/'figR4-2.pdf', transparent=True, bbox_inches='tight')

#%%
ratios = np.array([0.3, 0.35, 0.4, 0.5, 0.6])
results_common_Poisson = []
results_common_Poisson_ra = []
ra=0.2
for i, r in enumerate(ratios):
    _fname = f'P{r:.2f}HHp=0.25s=0.020f=0.080u=0.150_spike_train.dat'
    spk_fname_ra, spk_fname = maybe_downsample(path, _fname, ra, True)

    # Compare with ground truth connectivity
    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p=0.250.npy', T, dt=dt, delay=delay,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=None, algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results_common_Poisson.append(df_fig)

    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname_ra, N, path/f'connect_matrix-p=0.250.npy', T, dt=dt, delay=delay,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=(-8,-4), algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results_common_Poisson_ra.append(df_fig)

#%%
fig = plt.figure(figsize=(32, 20))
gs = fig.add_gridspec(3, len(ratios),
    left=0.05, right=0.95, top=0.95, bottom=0.35,
    wspace=0.5, hspace=0.5)
ax = np.array([fig.add_subplot(g) for g in gs]).reshape(3, len(ratios))
gs = fig.add_gridspec(1, 3,
    left=0.05, right=0.95, top=0.23, bottom=0.05,
    wspace=0.3, hspace=0.5)
ax_auc = fig.add_subplot(gs[0,0])
ax_prauc = fig.add_subplot(gs[0,1])
ax_acc = fig.add_subplot(gs[0,2])

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

auc_raw = [res['auc_svm']['TE'] for res in results_common_Poisson]
auc_ra = [res['auc_svm']['TE'] for res in results_common_Poisson_ra]
prauc_raw = [res['pr_auc_gt']['TE'] for res in results_common_Poisson]
prauc_ra = [res['pr_auc_gt']['TE'] for res in results_common_Poisson_ra]
acc_raw = [res['acc_gauss']['TE'] for res in results_common_Poisson]
acc_ra = [res['acc_gauss']['TE'] for res in results_common_Poisson_ra]
for axi, raw_, downsampled_ in zip([ax_auc, ax_prauc, ax_acc], [auc_raw, prauc_raw, acc_raw], [auc_ra, prauc_ra, acc_ra]):
    axi.plot(ratios, raw_, 'o-', label='raw', clip_on=False, ms=15)
    axi.plot(ratios, downsampled_, 'o-', label='downsampled', clip_on=False, ms=12)
    axi.set_ylim(0.5, 1.0)
    axi.set_xlabel('shared Poisson drive ratio', fontsize=26)
    axi.legend(fontsize=26)
    axi.grid(color='gray', alpha=0.3, linestyle='--')
ax_auc.set_ylabel('AUC', fontsize=26)
ax_prauc.set_ylabel('PR-AUC', fontsize=26)
ax_acc.set_ylabel('accuracy', fontsize=26)

for tag, axi in zip('ABC', ax[:,0]):
    fig.text(-0.4, 1.18, tag, fontsize=35, va='top', transform=axi.transAxes)
for tag, axi in zip('DEF', [ax_auc, ax_prauc, ax_acc]):
    fig.text(-0.20, 1.18, tag, fontsize=35, va='top', transform=axi.transAxes)

fig.savefig(root/'fig_nc/pdf'/'figR4-3.pdf', transparent=True, bbox_inches='tight')

#%%
strengths = np.array([0.02, 0.03, 0.04, 0.05])
uu = np.array([0.15, 0.08, 0.05, 0.03])
results_vary_strength = []
results_vary_strength_ra = []
ra=1.0
for s, u in zip(strengths, uu):
    _fname = f'HHp=0.25s={s:.3f}f=0.080u={u:.3f}_spike_train.dat'
    spk_fname_ra, spk_fname = maybe_downsample(path, _fname, ra, True)

    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p=0.250.npy', T, dt=dt, delay=delay,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=None, algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results_vary_strength.append(df_fig)

    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname_ra, N, path/f'connect_matrix-p=0.250.npy', T, dt=dt, delay=delay,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=None, algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'])
    results_vary_strength_ra.append(df_fig)

#%%
fig = plt.figure(figsize=(26, 20))
gs = fig.add_gridspec(3, len(strengths),
    left=0.05, right=0.95, top=0.95, bottom=0.35,
    wspace=0.4, hspace=0.4)
ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(len(strengths))] for r in range(3)])
gs = fig.add_gridspec(1, 3,
    left=0.05, right=0.95, top=0.25, bottom=0.05,
    wspace=0.4, hspace=0.5)
ax_auc = fig.add_subplot(gs[0,0])
ax_prauc = fig.add_subplot(gs[0,1])
ax_acc = fig.add_subplot(gs[0,2])

for i, (s, u, res, res_ra, axi) in enumerate(zip(strengths, uu, results_vary_strength, results_vary_strength_ra, ax.T)):
    _fname = f'HHp=0.25s={s:.3f}f=0.080u={u:.3f}_spike_train.dat'
    _ofname = f'HHp=0.25s={s:.3f}f=0.080u={u:.3f}_ra={ra:.2f}_spike_train.dat'
    spk_all = np.fromfile(path / _fname, dtype=float, count=20000).reshape(-1, 2)
    spk = np.fromfile(path / _ofname, dtype=float, count=20000).reshape(-1, 2)
    plot_spike_train(axi[0], N, spk, spk_all)
    axi[0].set_title(f'mean firing rate: {spk.shape[0] / (spk[:, 0].max() - spk[:, 0].min())*1000/N:.2f} Hz \n' + fr'($\nu$={u:.2f})', fontsize=22, pad=10)
    axi[0].set_xlim(500,900)
    tmp = res.loc['TE']
    plot_pdif_hist(tmp, axi[1])
    tmp = res_ra.loc['TE']
    plot_pdif_hist(tmp, axi[2])

auc_raw = [res['auc_svm']['TE'] for res in results_vary_strength]
auc_ra = [res['auc_svm']['TE'] for res in results_vary_strength_ra]
prauc_raw = [res['pr_auc_gt']['TE'] for res in results_vary_strength]
prauc_ra = [res['pr_auc_gt']['TE'] for res in results_vary_strength_ra]
acc_raw = [res['acc_gauss']['TE'] for res in results_vary_strength]
acc_ra = [res['acc_gauss']['TE'] for res in results_vary_strength_ra]

for axi, raw_, downsampled_ in zip([ax_auc, ax_prauc, ax_acc], [auc_raw, prauc_raw, acc_raw], [auc_ra, prauc_ra, acc_ra]):
    axi.plot(strengths, raw_, 'o-', label='raw', clip_on=False, ms=15)
    axi.plot(strengths, downsampled_, 'o-', label='downsampled', clip_on=False, ms=12)
    axi.set_ylim(0.4, 1.0)
    axi.set_xlabel(r'S $(mS\cdot cm^{-2})$', fontsize=26)
    axi.ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
    axi.legend(fontsize=26)
    axi.grid(color='gray', alpha=0.3, linestyle='--')
ax_auc.set_ylabel('AUC', fontsize=26)
ax_prauc.set_ylabel('PR-AUC', fontsize=26)
ax_acc.set_ylabel('accuracy', fontsize=26)

for tag, axi in zip('ABC', ax[:,0]):
    fig.text(-0.3, 1.18, tag, fontsize=35, va='top', transform=axi.transAxes)
for tag, axi in zip('DEF', [ax_auc, ax_prauc, ax_acc]):
    fig.text(-0.22, 1.14, tag, fontsize=35, va='top', transform=axi.transAxes)

fig.savefig(root/'fig_nc/pdf'/'figR4-4.pdf', transparent=True, bbox_inches='tight')

#%%
