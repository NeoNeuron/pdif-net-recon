# %%
# packages for plotting
from figrc import *
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
import pickle

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('orange_green', [GREEN, ORANGE], N=256)

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

# %%
ssLcon = np.array([0, 0.01, 0.05, 0.100, 0.20])
delay = 0
order = (4,1)
dt = 0.2
path = root / 'data/Lcon4'
recon_Lcon_list = []
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

#%% save results as pickle file
with open(root/'data/four_node_motif.pkl', 'wb') as f:
    pickle.dump({'ssRcon': ssRcon, 'recon_Rcon_list': recon_Rcon_list,
                 'ssLcon': ssLcon, 'recon_Lcon_list': recon_Lcon_list }, f)

#%%
with open(root/'data/four_node_motif.pkl', 'rb') as f:
    data = pickle.load(f)
    ssRcon = data['ssRcon']
    recon_Rcon_list = data['recon_Rcon_list']
    ssLcon = data['ssLcon']
    recon_Lcon_list = data['recon_Lcon_list']

fig = plt.figure(figsize=(12,5),)
ax = fig.subplots(2, len(recon_Rcon_list), 
    gridspec_kw=dict(wspace=0.4, hspace=0.8,
                     left=0.05,  right=0.98,
                     top=0.94,   bottom=0.08))
    
# Search of optimal delay parameter
true_conn = np.zeros((N,N))
true_conn[df_recon['pre_id'], df_recon['post_id']] = df_recon['connection']
for i, (recon, axi) in enumerate(zip(recon_Rcon_list, ax[0])):
    recon_conn = np.ones((N,N)) * recon['TE'].min()
    recon_conn[recon['pre_id'], recon['post_id']] = recon['TE']
    axi.imshow(recon_conn, cmap=cmap)
    axi.set_title(f's={ssRcon[i]:.3f}', fontsize=20, pad=-10)
    axi.set_xlabel('to', fontsize=20)
    axi.set_ylabel('from', fontsize=20)
    axi.set_xticks([])
    axi.set_yticks([])

for i, (recon, axi) in enumerate(zip(recon_Lcon_list, ax[1])):
    recon_conn = np.ones((N,N)) * recon['TE'].min()
    recon_conn[recon['pre_id'], recon['post_id']] = recon['TE']
    axi.imshow(recon_conn, cmap=cmap)
    axi.set_title(f's={ssLcon[i]:.3f}', fontsize=20, pad=-10)
    axi.set_xlabel('to', fontsize=20)
    axi.set_ylabel('from', fontsize=20)
    axi.set_xticks([])
    axi.set_yticks([])

for y, tag in enumerate('AB'):
    fig.text(0.02, 1.000-y*0.5, tag, fontsize=30, va='top')

fig.savefig(root/'fig_nc/pdf'/'figR3-1.pdf', transparent=True, bbox_inches='tight')