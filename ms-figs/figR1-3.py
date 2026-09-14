# %%
# packages for plotting
from figrc import *
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
from matplotlib.ticker import MaxNLocator
import pickle
# %% path for your time series data
path = Path(root/'raw_data/HH100')
T = 1e7         # length of time series, unit ms
dt = 0.5        # time step for descritization unit ms
delay = 3
order = (1,1)
N = 100
conn_type = ['LN', 'U', 'G']
results = []
results_pdif_val = []
ra = 1.0
for i, t in enumerate(conn_type):
    _fname = f'HH{t:s}-p=0.25s=0.020f=0.080u=0.150_spike_train.dat'
    spk_fname, _ = maybe_downsample(path, _fname, ra, False)

    # Compare with ground truth connectivity
    # ! important note: the 'connect_matrix-p=0.250.npy' file should be a binary adjacency matrix with shape (N, N), with W_{ij} representing the connection from node i to node j.
    # Reconstruction based on PTE-TE values, using GMM (ie., EM algorithm) to determine the reconstruction threshold
    df_recon, df_fig = run_reconstruction_TE(
        path, spk_fname, N, path/f'connect_matrix-p=0.250-{t:s}.dat', T, dt=dt, delay=delay,
        order=order, n_thread=128,
        recon_kwargs=dict(nbins=60, hist_range=(-10, -2), algorithm='EM'),
    )
    print(df_fig['auc_svm']['TE'], df_fig['acc_gauss']['TE'])
    results.append(df_fig)
    results_pdif_val.append(df_recon)
# %%
fig, ax = plt.subplots(3,3,figsize=(18,17), gridspec_kw={'wspace':0.4, 'hspace':0.5})
for i, (s, res, res_pdif, axi) in enumerate(zip(conn_type, results, results_pdif_val, ax.T)):
    buff = np.fromfile(path/f'connect_matrix-p=0.250-{s:s}.dat')
    conn = buff[:N*N].reshape(N, N)
    weights = buff[N*N:].reshape(N, N)
    axi[0].hist(weights[conn==1].flatten(), bins=30, color='k', alpha=0.5)
    axi[0].set_xlabel(r'S $(mS\cdot cm^{-2})$', fontsize=26)
    axi[0].set_ylabel('Count', fontsize=26)
    axi[0].ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
    axi[0].xaxis.get_offset_text().set_x(1.15)
    axi[0].set_xlim(0)
    tmp = res.loc['TE']
    plot_pdif_hist(tmp, axi[1], False)
    x = res_pdif['weight'].to_numpy()
    y = res_pdif['TE'].to_numpy()
    axi[2].plot(x, y, '.', color='k', alpha=0.5)
    
    ffit = squarefit(x, y)
    x_line = np.linspace(np.min(x), np.max(x), 300)
    axi[2].plot(x_line, ffit(x_line), color='r', lw=2)
    axi[2].set_xlim(0)
    axi[2].set_ylim(0)
    axi[2].set_xlabel(r'S $(mS\cdot cm^{-2})$', fontsize=26)
    axi[2].set_ylabel('PDIF value', fontsize=26)
    axi[2].ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
    

for tag, axi in zip('ABCDEFGHI', ax.T.flatten()):
    fig.text(-0.25, 1.14, tag, fontsize=35, va='top', transform=axi.transAxes)

fig.savefig(root/'fig_nc/pdf'/'figR1-3.pdf', transparent=True, bbox_inches='tight')

#%%