# %%
# packages for plotting
from figrc import *
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
import pickle

    
buff = np.load('../DI_roc_auc_results.npz', allow_pickle=True)
print(list(buff.keys()))
print(buff['labels'])
meanDI = buff['mean_DI']
fig, ax = plt.subplots(figsize=(8, 6))
real_xlim = []
offdiag_mask = ~np.eye(100, dtype=bool)
conn_mask = buff['connect_submatrix']==1
vrange = (np.min(meanDI[offdiag_mask]), np.max(meanDI[offdiag_mask]))
counts1, edges = np.histogram(meanDI[offdiag_mask&conn_mask], bins=50, range=vrange)
counts0, edges = np.histogram(meanDI[offdiag_mask&(~conn_mask)], bins=50, range=vrange)
edges = edges[:-1] + (edges[1] - edges[0])/2
for counts, color in zip((counts1, counts0), (ORANGE, GREEN)):
    mask = counts > 0
    real_xlim.append([edges[mask][0], edges[mask][-1]])
    ax.plot(edges[mask], counts[mask], color=color, lw=5, clip_on=True)
    ax.fill_between(edges[mask], 0, counts[mask], color=color, alpha=0.5)
# if threshold:
#     ax.axvline(series['th_gauss'], ls='-', color=PINK, lw=4)
# ax.xaxis.set_major_formatter(sci_formatter)
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
ax.set_ylim(0)
ax.set_xlim(min(x[0] for x in real_xlim), max(x[-1] for x in real_xlim))
ax.set_xlabel('Directed Information', fontsize=26)
ax.set_ylabel('density', fontsize=26)
fig.savefig(root/'fig_nc/pdf'/'figR4-0.pdf', transparent=True, bbox_inches='tight')