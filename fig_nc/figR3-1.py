# %%
# packages for plotting
from figrc import *
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
import pickle

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('orange_green', [GREEN, ORANGE], N=256)
from matplotlib.patches import Rectangle

def heatmap(data, ax, vmin=0.5, vmax=1.0, pad=0.005, width=0.01,
            cbar_label='PDIF value', xlabel='to', y_label='from', cmap=None):
    fig = ax.get_figure()
    im = ax.pcolormesh(data, vmin=vmin, vmax=vmax, cmap=cmap)
    pos = ax.get_position()  # save main axes position BEFORE adding colorbar
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
    cbar = fig.colorbar(im, cax=cax, label=cbar_label)
    cbar.ax.yaxis.label.set_fontsize(20)
    # set colorbar tick label size
    cbar.ax.tick_params(labelsize=18)
    ax.set_position(pos) 
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel(y_label, fontsize=26)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    cbar.ax.yaxis.get_major_formatter().set_useMathText(True)
    cbar.ax.yaxis.set_offset_position('left')
    cbar.ax.yaxis.offsetText.set_fontsize(18)
    
    return ax, cbar

# %%
path = root / 'data/Rcon4'
spk_fname = Path(path/'Rconp=0.38s=0.002_spike_train.dat')
vol_fname = 'Rconp=0.38s=0.002_x.dat' # continuous-valued time series file name
spk = np.fromfile(spk_fname, dtype=float).reshape(-1, 2)
x = np.fromfile(path/vol_fname, dtype=float, count = 5*10000).reshape(-1, 5)
conn = np.fromfile(path/'connect_matrix-p=0.375.dat', dtype=float).reshape(4,4)
print(conn)

# %%
# path for your time series data
T = 1e7         # length of time series, unit ms
N = 4           # total number of nodes in the network
ref = 5.0       # refractory period in thresholding to generate surrogated spikes, unit ms
th = 10         # threshold for generating surrogated spikes
dt = 3.0        # time step for descritization unit ms

# %%
# # load the voltage time series
# spk_fname = binarize(path/vol_fname,
#     N=N, threshold=th, T=T, verbose=False,
#     ref=ref, force_regen=True)
# print(spk_fname)


# %%
ssRcon = np.array([0, 0.002, 0.004, 0.010, 0.020])
delay = 0
order = (1,5)
dt = 3.0
path = root / 'data/Rcon4'
recon_Rcon_list = []
auc_Rcon_list = []
acc_Rcon_list = []
for s in ssRcon:
    spk_fname = Path(path/f'Rconp=0.38s={s:.3f}_spike_train.dat')
    estimator = CausalityEstimator(
        path, spk_fname, N, delay=delay, T=T, dt=dt,
        n_thread=12, order=order,
    )
    # Search of optimal delay parameter
    optimal_m = estimator.get_optimal_delay(np.arange(20)*dt, mode=1)
    print(optimal_m)
    data = estimator.fetch_data(new_run=True, delay=9)

    # Compare with ground truth connectivity
    data_matched = match_features(data, N, path/'connect_matrix-p=0.375.dat')
    # ! important note: the 'connect_matrix-p=0.250.npy' file should be a binary adjacency matrix with shape (N, N), with W_{ij} representing the connection from node i to node j.
    # Reconstruction based on PTE-TE values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=20, hist_range=None, algorithm='EM')
    print(df_fig['auc_svm']['TE'], df_fig['auc_svm']['TE'])
    recon_Rcon_list.append(df_recon)
    auc_Rcon_list.append(df_fig['auc_svm']['TE'])
    acc_Rcon_list.append(df_fig.get('acc_svm', {}).get('TE', np.nan))

# %%
ssLcon = np.array([0, 0.01, 0.05, 0.100, 0.20])
delay = 0
order = (4,1)
dt = 0.2
path = root / 'data/Lcon4'
recon_Lcon_list = []
auc_Lcon_list = []
acc_Lcon_list = []
for s in ssLcon:
    spk_fname = Path(path/f'Lconp=0.38s={s:.3f}_spike_train.dat')
    estimator = CausalityEstimator(
        path, spk_fname, N, delay=delay, T=T, dt=dt,
        n_thread=12, order=order,
    )
    # Search of optimal delay parameter
    optimal_m = estimator.get_optimal_delay(np.arange(20)*dt, mode=1)
    print(optimal_m)
    data = estimator.fetch_data(new_run=True)

    # Compare with ground truth connectivity
    data_matched = match_features(data, N, path/'connect_matrix-p=0.375.dat')
    # ! important note: the 'connect_matrix-p=0.250.npy' file should be a binary adjacency matrix with shape (N, N), with W_{ij} representing the connection from node i to node j.
    # Reconstruction based on PTE-TE values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = reconstruction_analysis_TE(data_matched, nbins=20, hist_range=None, algorithm='EM')
    print(df_fig['auc_svm']['TE'], df_fig['auc_svm']['TE'])
    recon_Lcon_list.append(df_recon)
    auc_Lcon_list.append(df_fig['auc_svm']['TE'])
    acc_Lcon_list.append(df_fig.get('acc_svm', {}).get('TE', np.nan))

#%% save results as pickle file
with open(root/'data/four_node_motif.pkl', 'wb') as f:
    pickle.dump({'ssRcon': ssRcon, 'recon_Rcon_list': recon_Rcon_list,
                 'auc_Rcon_list': auc_Rcon_list,
                 'acc_Rcon_list': acc_Rcon_list,
                 'ssLcon': ssLcon, 'recon_Lcon_list': recon_Lcon_list,
                 'auc_Lcon_list': auc_Lcon_list,
                 'acc_Lcon_list': acc_Lcon_list }, f)

#%%
with open(root/'data/four_node_motif.pkl', 'rb') as f:
    data = pickle.load(f)
    ssRcon = data['ssRcon']
    recon_Rcon_list = data['recon_Rcon_list']
    auc_Rcon_list = data['auc_Rcon_list']
    acc_Rcon_list = data['acc_Rcon_list']
    ssLcon = data['ssLcon']
    recon_Lcon_list = data['recon_Lcon_list']
    auc_Lcon_list = data['auc_Lcon_list']
    acc_Lcon_list = data['acc_Lcon_list']

fig = plt.figure(figsize=(12,10),)
gs = fig.add_gridspec(2, 3,
    wspace=0.5, hspace=0.4,
    left=0.02, right=0.98,
    top=0.94, bottom=0.32,)
ax = np.array([
    fig.add_subplot(g) for g in gs
]).reshape(2,-1)

gs = fig.add_gridspec(1, 4,
    wspace=0.6, hspace=0.4,
    left=0.05, right=0.98,
    top=0.20, bottom=0.02,)
ax_auc_R = fig.add_subplot(gs[0, 0])
ax_acc_R = fig.add_subplot(gs[0, 1])
ax_auc_L = fig.add_subplot(gs[0, 2])
ax_acc_L = fig.add_subplot(gs[0, 3])

# load true connectivity matrices for Rcon and Lcon
true_Rcon = np.fromfile(root/'data/Rcon4'/ 'connect_matrix-p=0.375.dat', dtype=float).reshape(N, N)
true_Lcon = np.fromfile(root/'data/Lcon4'/ 'connect_matrix-p=0.375.dat', dtype=float).reshape(N, N)
for i, (recon, axi) in enumerate(zip(recon_Rcon_list, ax[0])):
    recon_conn = np.ones((N,N)) * recon['TE'].min()
    recon_conn[recon['pre_id'], recon['post_id']] = recon['TE']
    heatmap(recon_conn, cmap=cmap, ax=axi, vmin=recon['TE'].min(), vmax=recon['TE'].max(), pad=0.003, width=0.02)
    # overlay cyan squares for ground truth connections
    for (pi, pj), val in np.ndenumerate(true_Rcon):
        if val:
            # rectangle: x=j, y=i, width=1, height=1
            rect = Rectangle((pj, pi), 1, 1, edgecolor='cyan', facecolor='none', linewidth=2, alpha=0.8, clip_on=False)
            axi.add_patch(rect)
    axi.set_title(f's={ssRcon[i]:.3f}', fontsize=26, pad=-10)

for i, (recon, axi) in enumerate(zip(recon_Lcon_list[::2], ax[1])):
    recon_conn = np.ones((N,N)) * recon['TE'].min()
    recon_conn[recon['pre_id'], recon['post_id']] = recon['TE']
    heatmap(recon_conn, cmap=cmap, ax=axi, vmin=recon['TE'].min(), vmax=recon['TE'].max(), pad=0.003, width=0.02)
    # overlay cyan squares for ground truth connections
    for (pi, pj), val in np.ndenumerate(true_Lcon):
        if val:
            rect = Rectangle((pj, pi), 1, 1, edgecolor='cyan', facecolor='none', linewidth=2, alpha=0.8, clip_on=False)
            axi.add_patch(rect)
    axi.set_title(f's={ssLcon[i*2]:.3f}', fontsize=26, pad=-10)

ax_auc_R.plot(ssRcon, auc_Rcon_list, marker='o', color=ORANGE, clip_on=False)
ax_auc_R.set_xlabel('s')
ax_auc_R.set_ylabel('AUC')
ax_auc_R.set_ylim(0, 1)
ax_auc_R.grid(alpha=0.3)
ax_auc_R.tick_params(axis='both', labelsize=20)

ax_acc_R.plot(ssRcon, acc_Rcon_list, marker='o', color=ORANGE, clip_on=False)
ax_acc_R.set_xlabel('s')
ax_acc_R.set_ylabel('accuracy')
ax_acc_R.set_ylim(0, 1)
ax_acc_R.grid(alpha=0.3)
ax_acc_R.tick_params(axis='both', labelsize=20)

ax_auc_L.plot(ssLcon, auc_Lcon_list, marker='o', color=GREEN, clip_on=False)
ax_auc_L.set_xlabel('s')
ax_auc_L.set_ylabel('AUC')
ax_auc_L.set_ylim(0, 1)
ax_auc_L.grid(alpha=0.3)
ax_auc_L.tick_params(axis='both', labelsize=20)

ax_acc_L.plot(ssLcon, acc_Lcon_list, marker='o', color=GREEN, clip_on=False)
ax_acc_L.set_xlabel('s')
ax_acc_L.set_ylabel('accuracy')
ax_acc_L.set_ylim(0, 1)
ax_acc_L.grid(alpha=0.3)
ax_acc_L.tick_params(axis='both', labelsize=20)


for y, tag in enumerate('AB'):
    fig.text(0.01, 0.980-y*0.36, tag, fontsize=30, va='top')

for x, tag in enumerate('CDEF'):
    fig.text(0.01+x*0.25, 0.26, tag, fontsize=30, va='top')

fig.savefig(root/'fig_nc/pdf'/'figR3-1.pdf', transparent=True, bbox_inches='tight')
# %%
