# %%
from figrc import *
import numpy as np
import matplotlib.pyplot as plt
import pdif.utils as c4u
import pdif.myplot as mplt
from pdif.pdif import CausalityEstimator
data_path = root / 'raw_data'
#%%
# for p in np.arange(0.1, 0.91, 0.1):
#     data = c4u.load_spike_data(
#         f'tmp/HHp={p:.2f}s=0.005f=0.080u=0.150_spike_train.dat', xrange=(0, 1000))
#     plt.figure(figsize=(10, 3))
#     plt.plot(data[:,0], data[:,1], '|')
#     # %  Initialize the CausalityEstimator
#     estimator = CausalityEstimator(
#         path='./tmp/',
#         spk_fname=f'HHp={p:.2f}s=0.005f=0.080u=0.150',
#         N=100, T=1e7, n_thread=120, delay=3, order=(1,1))
#     data = estimator.fetch_data(new_run=True)

#     data_matched = c4u.match_features(data, N=100,
#                                        conn_file=f'./tmp/connect_matrix-p={p:.3f}.npy')
#     data_recon, fig_data = c4u.reconstruction_analysis(data_matched, nbins=100)
#     fig = mplt.reconstruction_illustration(fig_data)
#%%
estimator = CausalityEstimator(
    path=data_path/'HH100',
    spk_fname=f'HHp=0.25s=0.020f=0.080u=0.150',
    N=100, T=1e7, n_thread=120, delay=3, order=(1,1))
data = estimator.fetch_data(new_run=True)

data_matched = c4u.match_features(data, N=100,
                                    conn_file=data_path/f'HH100/connect_matrix-p=0.250.dat')
data_recon, fig_data = c4u.reconstruction_analysis(data_matched, nbins=100)
fig = mplt.reconstruction_illustration(fig_data)
#%%
TE = data['TE'].to_numpy().reshape(100,100)
conn = np.fromfile(data_path/'HH100/connect_matrix-p=0.250.dat', dtype=float).reshape(100,100)
# %%
conn2_chain = (conn.T @ conn.T).T
# conn3_chain = (conn.T @ conn.T @ conn.T).T
conn2_confounder = np.zeros((100, 100))
for i in range(100):
    for j in np.arange(i, 100):
        n_confounder = np.sum(conn[:,i] * conn[:,j])
        conn2_confounder[i,j] = n_confounder
        conn2_confounder[j,i] = n_confounder
# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
mask = (conn==0) * (~np.eye(100, dtype=bool))
# plt.plot(conn2_chain[mask], np.sqrt(TE[mask]), 'o', alpha=0.1)
# plt.plot(conn2_confounder[mask], np.sqrt(TE[mask]), 'o', alpha=0.1)
ax.plot((conn2_chain[mask]+conn2_confounder[mask]), np.sqrt(TE[mask]), 'o', c=GREEN, alpha=0.1)
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
ax.set_xlabel('No. of indirect paths')
ax.set_ylabel(r'$\sqrt{\text{PDIF}}$')
corr = np.corrcoef(conn2_chain[mask]+conn2_confounder[mask], np.sqrt(TE[mask]))[0,1]
ax.set_title(f"R={corr:.3f}", fontsize=20, pad=-20)
ax.tick_params(axis='both', labelsize=18)
ax.yaxis.get_offset_text().set_size(16)
fig.savefig(root/'fig_nc/pdf/figS8.pdf', bbox_inches='tight')
# %%
auc = []
acc = []
Ts = [1e4, 5e4, 1e5, 2e5, 3e5, 4e5, 6e5, 8e5, 1e6]
for T in Ts:
    # %  Initialize the CausalityEstimator
    estimator = CausalityEstimator(
        path=data_path/'HH100',
        spk_fname=f'HHp=0.25s=0.020f=0.080u=0.150',
        N=100, T=T, DT=1e3, n_thread=60, delay=3, order=(1,1))
    data = estimator.fetch_data(new_run=True)

    data_matched = c4u.match_features(
        data, N=100, conn_file=data_path/f'HH100/connect_matrix-p=0.250.dat')
    data_recon, fig_data = c4u._reconstruction_analysis(
        data_matched, 'TE', nbins=100, hist_type='log', algorithm='EM')
    # fig = mplt.reconstruction_illustration(fig_data)
    auc.append(fig_data['auc_svm'])
    # acc.append(fig_data['acc_gauss'])
#%%
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.plot(Ts, auc, 'o-', clip_on=False, c=GREEN, mec='w', ms=10, mew=2)
# ax.plot(Ts, acc, 'o-', clip_on=False, mec='w', ms=10)
ax.set_xlabel('time length (ms)')
ax.set_ylabel('AUC')
ax.set_xlim(0, 1e6)
ax.set_ylim(0.48, 1)
ax.tick_params(axis='both', labelsize=18)
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
ax.xaxis.get_offset_text().set_size(16)
fig.savefig(root/'fig_nc/pdf/figS7.pdf', bbox_inches='tight')