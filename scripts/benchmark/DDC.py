# %%
from pathlib import Path
root_path = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from causal4.ddc import DDC, c_sensitivity, DDC_long
import causal4.myplot as myplot
from causal4.Causality import CausalityEstimator
from causal4.utils import match_features, reconstruction_analysis, fetch_voltage
import time

import yaml
import matplotlib.pyplot as plt
plt.rcParams['font.size']=16
import seaborn as sns

def plot_roc(conn, ddc, ax):
    fpr, tpr,_ = roc_curve(conn.flatten(),ddc.flatten())
    ax.plot(fpr, tpr, lw=4)[0].set_clip_on(False)
    ax.plot([0,1],[0,1],ls='--',c='grey')
    ax.axis('scaled')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

def plot_hist(conn, ddc, ax):
    counts, bins = np.histogram(np.abs(ddc[conn!=0]), bins=100, range=(0,np.abs(ddc).max()))
    nonzero_mask = counts>0
    ax.semilogy(bins[:-1][nonzero_mask], counts[nonzero_mask], lw=2, ls='-',c='b', label='conn')
    counts, bins = np.histogram(np.abs(ddc[conn==0]), bins=100, range=(0,np.abs(ddc).max()))
    nonzero_mask = counts>0
    ax.semilogy(bins[:-1][nonzero_mask], counts[nonzero_mask], lw=2, ls=':',c='b', label='disconn')
    ax.set_xlabel('DDC')
    ax.set_ylabel('density')
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
    ax.legend()

#%%
#! Calculate causality
with open('benchmark10_causal.yml', 'r') as yamlfile:
    pm_causal_set = yaml.load(yamlfile, Loader=yaml.FullLoader)
for key in pm_causal_set.keys():
    pm_causal_set[key]['path'] = root_path / pm_causal_set[key]['path']

# %%
#! Calculate DDC
# load data
N = 10
p = 0.25
conn = data_recon_total[0][['pre_id', 'post_id', 'connection']]
voltages = fetch_voltage(
    './HH/data/EE/N=100/HHp={p:.2f}s={s:.3f}f=0.100u=0.100_voltage.dat',
    N=100, voltage_range = (0,nll[-1]),
)
for ratio in range(1,6):
    dt=0.2*ratio
    ddc = DDC(voltages[:nlines*ratio:ratio,1:].T, dt)
    # column (row) index representing pre_id (post_id)
    xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    ddc_df = pd.DataFrame({'pre_id': xx.flatten(), 'post_id': yy.flatten(), 'recon-ddc': ddc.flatten()})
    data_recon_ddc = conn.merge(ddc_df, how='left', on=['pre_id', 'post_id'])
    data_recon_total.append(data_recon_ddc)
    auc = roc_auc_score(data_recon_ddc['connection'], data_recon_ddc['recon-ddc'])
    c_sens = c_sensitivity(data_recon_ddc['connection'].to_numpy(), data_recon_ddc['recon-ddc'].to_numpy())
    df['type'] = 'DDC'
    df['s'].append(s)
    df['L'].append(nlines)
    df['dt'].append(dt)
    df['auc'].append(auc)
    df['c_sens'].append(c_sens)
df = pd.DataFrame(df)
df.to_pickle(f'TE_vs_DDC_HH{N:d}-EE-p={p:.2f}.pkl')
# %%
#! Load c-sens and AUC data.
p = 0.10
EI_type = 'EE'
N=100
df_DDC = pd.read_pickle(f'TE_vs_DDC_HH{N:d}-{EI_type:s}-p={p:.2f}.pkl')

# ! fix entry which is not precise enough
# df_DDC.loc[(df_DDC['dt']>0.6)*(df_DDC['dt']<0.7), 'dt']=0.6
# df_DDC.to_pickle(f'df_ddc_HH-{EI_type:s}-p={p:.2f}.pkl')
# %%
#! Performance curve of DDC and CC
EI_type='EE'
dt = 1.0
conn_fname = f'HH/data/{EI_type:s}/N=100/connect_matrix-p=0.100.dat'
conn = np.fromfile(conn_fname, dtype=float).reshape(N,N)
for nlines in nll:
    s = 0.02
    #%
    mask = (df_DDC['s']==s)*(df_DDC['dt']==dt)*(df_DDC['L']==nlines)
    ddc = df_DDC.loc[mask, 'ddc'].values[0]
    c_sens = df_DDC.loc[mask, 'c_sens'].values[0]
    auc = df_DDC.loc[mask, 'auc'].values[0]

    # %
    # plotting figure
    # DDC
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    plot_hist(conn, ddc, ax[0])
    ax[0].set_title(f'c-sens = {c_sens:6.3f}    AUC = {auc:6.3f}')

    # CC
    mask = (df_CC['s']==s)*(df_CC['dt']==dt)*(df_CC['L']==nlines)
    CC = df_CC.loc[mask, 'cc'].values[0]
    c_sens = df_CC.loc[mask, 'c_sens'].values[0]
    auc = df_CC.loc[mask, 'auc'].values[0]
    plot_hist(conn, CC, ax[1])
    ax[1].set_title(f'c-sens = {c_sens:6.3f}    AUC = {auc:6.3f}')
    ax[1].set_xlabel(r'$CC^2$')
    [axi.set_xlim(0) for axi in ax]
    plt.tight_layout()
    vfname = f'HHp=0.10s={s:.3f}f=0.100u=0.100'
    plt.savefig('image/DDC-CC_'+EI_type+'_'+f"L={nlines:0.0e}bin={dt:.3f}"+'_'+vfname+'.pdf')

# %%
myplot.hist_causal_with_conn_mask(pm_causal)
myplot.hist_causal_with_conn_mask_linear(pm_causal)
# %%
pm_causal['T']=1_600_000
pm_causal['fname']='HHp=0.25s=0.020f=0.100u=0.100'
pm_causal['con_mat'] = 'connect_matrix-p=0.250.dat'
# Causality.run(True, **pm_causal)
myplot.hist_causal_with_conn_mask(pm_causal)
# myplot.hist_dp(pm_causal)
# %%
#! plot raster
dat=np.fromfile('HH/data/EE/N=100/HHp=0.25s=0.040f=0.100u=0.100_spike_train.dat', dtype=float).reshape(-1,2)
plt.figure(figsize=(10,3))
plt.plot(dat[:1000,0], dat[:1000,1], '|')

# %%


#%%
#! Scan sampling rate: dt
L = 1_600_000
s = 0.04
mask_DDC = (df_DDC['s']==s)*(df_DDC['L']==L)
mask_CC = (df_CC['s']==s)*(df_CC['L']==L)
fig, ax = plt.subplots(1,2, figsize=(10,4), gridspec_kw={'top':0.84, 'bottom':0.2, 'wspace':0.5, 'left':0.1, 'right':0.95})
for axi, col in zip(ax, ('c_sens', 'AUC')):
    axi.plot(df_DDC.loc[mask_DDC, 'dt'], df_DDC.loc[mask_DDC, col.lower()], '-o', label='DDC')
    axi.plot(df_CC.loc[mask_CC, 'dt'], df_CC.loc[mask_CC, col.lower()], '-o', label='our method')
    axi.set_title(col)
    axi.set_xlabel(r'$\Delta t$ (ms)')
    axi.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
ax[1].legend()
fig.suptitle(r's=%.2f,  $L$=%.1e ms'%(s, L), va='top')
plt.savefig(f"image/C-sens_AUC_{EI_type:s}_p={p:.2f}L={L:.2e}s={s:.2f}.pdf")

# %%
# ! concat all data
p = 0.25
EI_type = 'EE'
df_DDC = pd.read_pickle(f'df_ddc_HH-{EI_type:s}-p={p:.2f}.pkl')
df_CC  = pd.read_pickle(f'df_CC_HH-{EI_type:s}-p={p:.2f}.pkl')
df_DDC['p']=p
df_CC['p']=p
for p in (0.10, 0.05):
    df_DDC_buff = pd.read_pickle(f'df_ddc_HH-{EI_type:s}-p={p:.2f}.pkl')
    df_CC_buff  = pd.read_pickle(f'df_CC_HH-{EI_type:s}-p={p:.2f}.pkl')
    df_DDC_buff['p']=p
    df_CC_buff['p']=p
    df_DDC = pd.concat([df_DDC, df_DDC_buff])
    df_CC  = pd.concat([df_CC, df_CC_buff])
df_DDC.to_pickle(f'df_ddc_HH-{EI_type:s}.pkl')
df_CC.to_pickle(f'df_CC_HH-{EI_type:s}.pkl')

# %%
EI_type = 'EE'
df_DDC = pd.read_pickle(f'df_ddc_HH-{EI_type:s}.pkl')
df_CC  = pd.read_pickle(f'df_CC_HH-{EI_type:s}.pkl')
ps = [0.25, 0.10, 0.05]
# %%
#! scan sparsity
L = 1_600_000
s = 0.02
dt = 1.0
mask_DDC = (df_DDC['s']==s)*(df_DDC['L']==L)*(df_DDC['dt']==dt)
mask_CC = (df_CC['s']==s)*(df_CC['L']==L)*(df_CC['dt']==dt)
fig, ax = plt.subplots(1,2, figsize=(10,4), gridspec_kw={'top':0.84, 'bottom':0.2, 'wspace':0.5, 'left':0.1, 'right':0.95})
for axi, col in zip(ax, ('c_sens', 'AUC')):
    axi.plot(df_DDC.loc[mask_DDC, 'p'], df_DDC.loc[mask_DDC, col.lower()], '-o', label='DDC')
    axi.plot(df_CC.loc[mask_CC, 'p'], df_CC.loc[mask_CC, col.lower()], '-o', label='our method')
    axi.set_title(col)
    axi.set_xlabel('Network Sparsity')
ax[1].legend()
fig.suptitle(r's=%.2f,  $L$=%.1e ms'%(s, L), va='top')
plt.savefig(f"image/C-sens_AUC_{EI_type:s}_dt={dt:.1f}L={L:.2e}s={s:.2f}.pdf")

# %%
N=100
conn_fname = f'Lorenz/data/EE/N=100/connect_matrix-p=0.250.dat'
conn = np.fromfile(conn_fname, dtype=float).reshape(N,N)
dt=0.01
ddc = DDC_long('Lorenz/data/EE/N=100/Lp=0.25s=0.250f=0.000u=0.000_voltage.dat', N=100, batch=10_000_000,)
c_sens = c_sensitivity(conn.T, ddc)
auc = roc_auc_score(conn.T.flatten(), ddc.flatten())
print(f"c_sens = {c_sens:.3f}, auc = {auc:.3f}")
np.save('ddc_Lorenz_1e8.npy', ddc)
# %%
fig, ax = plt.subplots(1,1)
plot_hist(conn, ddc, ax)
ax.set_title(f'c-sens = {c_sens:6.3f}    AUC = {auc:6.3f}')
fig.savefig('image/DDC_Lorenz100-L=1e8_p=0.25.pdf')
# %%