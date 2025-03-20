# -*- coding: utf-8 -*-
# Author: Kai Chen

#%%
from pathlib import Path
root = Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from causal4.Causality import CausalityEstimator
from causal4.utils import match_features

fig, ax = plt.subplots(
    2,4,figsize=(14,7),)# gridspec_kw=dict(wspace=0.3, hspace=0.5, width_ratios=(0.8,1,1,1))
# )

def make_graph_diagram(ax, nodesize=2000, fontsize=26):
    G = nx.DiGraph()
    G.add_edges_from([(1,2), (2,3)])
    G = nx.relabel_nodes(G, {1:'X', 2:'Y', 3:'Z'})
    pos = {
        n: coordinate 
        for n, coordinate in zip(G,((1,np.sqrt(3)),(0,0),(2,0),))
    }
    nx.draw_networkx_nodes(
        G, pos=pos, ax=ax,
        node_color='#F49227',
        edgecolors='k',
        node_size=nodesize,
    )
    nx.draw_networkx_labels(
        G, pos=pos, ax=ax,
        labels={n:n for n in G},
        font_size=fontsize,
        font_weight='bold',
    )
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax,
        width=2,
        node_size=nodesize,
        arrowsize=40,
        arrows=True,
    )
    ax.set_clip_on(False)
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-1,3)
    ax.set_ylim(-1,np.sqrt(3)+1)

make_graph_diagram(ax[0,0])

estimator = CausalityEstimator(
    # root/'HH3-chain', 'HHp=0.25s=0.020f=0.080u=0.150', 3, delay=3, T=1e6,
    root/'HH3-chain', 'HHp=0.25s=0.020f=0.200u=0.050', 3, delay=3, T=1e7,
)
data = estimator.fetch_data()
data_matched = match_features(data, 3, root/'HH3-chain/connect_matrix-p=0.250.npy')
gt = np.load(root/'HH3-chain/connect_matrix-p=0.250.npy')
TE = np.zeros((3,3))
TE[~np.eye(3, dtype=bool)] = data_matched['TE']
sns.heatmap(gt, linecolor='#DDDDDD', lw=2, ax=ax[0,1], square=True, cmap='Oranges', cbar=False)
sns.heatmap(TE, linecolor='#DDDDDD', lw=2, ax=ax[0,2], square=True, cmap='Oranges', cbar_kws={'label':'PTD-TE'})
for axi in ax[0,1:]:
    axi.set_xticks(np.arange(3)+0.5, ['X', 'Y', 'Z'])
    axi.set_yticks(np.arange(3)+0.5, ['X', 'Y', 'Z'])
    axi.set_xlabel('To', fontsize=14)
    axi.set_ylabel('From', fontsize=14)
ax[0,1].set_title('Ground truth', fontsize=25, pad=16)
ax[0,2].set_title('PTD-TE', fontsize=25, pad=16)
#%
f = np.arange(0.05,0.21,0.01)
fu = np.arange(1, 5.1, 0.2)*1e-2
ff, fufu = np.meshgrid(f, fu)
ratio = []
for f, fu in zip(ff.flatten(), fufu.flatten()):
    estimator.spk_fname = f"HHp=0.25s=0.020f={f:.3f}u={fu/f:.3f}"
    data = estimator.fetch_data(new_run=True)
    data01 = data[data['pre_id'].eq(0) * data['post_id'].eq(1)]['TE'].values[0]
    data02 = data[data['pre_id'].eq(0) * data['post_id'].eq(2)]['TE'].values[0]
    ratio.append(data01 / data02)
ratio = np.array(ratio).reshape(ff.shape)
#%
Z=np.log10(ratio)
print(f">> Min(ratio) = {ratio.min():.2f}")
pax = ax[0,3].pcolormesh(ff*100, fufu*100, Z, vmin=2, vmax=4, edgecolor='w', lw=0.002)
cb = fig.colorbar(pax, ax=ax[0,3], ticks=[2,3,4], orientation='vertical')
cb.ax.set_yticklabels([r'$10^{2}$',r'$10^{3}$',r'$10^{4}$'])
ax[0,3].set_ylabel(r'$\nu f$ $(\times 10^{-2})$')
ax[0,3].set_xlabel(r'$f$ $(\times 10^{-2})$')
ax[0,3].set_yticks([1,3,5], ['1', '3', '5'])
ax[0,3].set_xticks([5,10,15,20], ['5', '10', '15', '20'])

ax[0,3].set_title(r'$T^\mathrm{PTD}_{X\to Y} / T^\mathrm{PTD}_{X\to Z}$', fontsize=25, pad=16)

#%
G = nx.DiGraph()
G.add_edges_from([(1,2), (1,3)])
G = nx.relabel_nodes(G, {1:'X', 2:'Y', 3:'Z'})
pos = {
    n: coordinate 
    for n, coordinate in zip(G,((1,np.sqrt(3)),(0,0),(2,0),))
}
nx.draw_networkx_nodes(
    G, pos=pos, ax=ax[1,0],
    node_color='#F49227',
    edgecolors='k',
    node_size=2000,
)
nx.draw_networkx_labels(
    G, pos=pos, ax=ax[1,0],
    labels={n:n for n in G},
    font_size=26,
    font_weight='bold',
)
nx.draw_networkx_edges(
    G, pos=pos, ax=ax[1,0],
    width=2,
    node_size=2000,
    arrowsize=40,
    arrows=True,
)
ax[1,0].set_clip_on(False)
ax[1,0].axis('equal')
ax[1,0].axis('off')
ax[1,0].set_xlim(-1,3)
ax[1,0].set_ylim(-1,np.sqrt(3)+1)


estimator = CausalityEstimator(
    root/'HH3-confounder', 'HHp=0.25s=0.020f=0.080u=0.150', 3, delay=3, T=1e7,
)
data = estimator.fetch_data(new_run=True)
data_matched = match_features(data, 3, root/'HH3-confounder/connect_matrix-p=0.250.npy')
gt = np.load(root/'HH3-confounder/connect_matrix-p=0.250.npy')
TE = np.zeros((3,3))
TE[~np.eye(3, dtype=bool)] = data_matched['TE']
sns.heatmap(gt, linecolor='#DDDDDD', lw=2, ax=ax[1,1], square=True, cmap='Oranges', cbar=False)
sns.heatmap(TE, linecolor='#DDDDDD', lw=2, ax=ax[1,2], square=True, cmap='Oranges', cbar_kws={'label':'PTD-TE'})
for axi in ax[1,1:]:
    axi.set_xticks(np.arange(3)+0.5, ['X', 'Y', 'Z'])
    axi.set_yticks(np.arange(3)+0.5, ['X', 'Y', 'Z'])
    axi.set_xlabel('To', fontsize=14)
    axi.set_ylabel('From', fontsize=14)
ax[1,1].set_title('Ground truth', fontsize=25, pad=16)
ax[1,2].set_title('PTD-TE', fontsize=25, pad=16)
#%

f = np.arange(0.05,0.21,0.01)
fu = np.arange(1, 5.1, 0.2)*1e-2
ff, fufu = np.meshgrid(f, fu)
ratio = []
for f, fu in zip(ff.flatten(), fufu.flatten()):
    estimator.spk_fname = f"HHp=0.25s=0.020f={f:.3f}u={fu/f:.3f}"
    data = estimator.fetch_data(new_run=True)
    data01 = data[data['pre_id'].eq(0) * data['post_id'].eq(1)]['TE'].values[0]
    data02 = data[data['pre_id'].eq(1) * data['post_id'].eq(2)]['TE'].values[0]
    ratio.append(data01 / data02)
ratio = np.array(ratio).reshape(ff.shape)
#%
Z=np.log10(ratio)
print(f">> Min(ratio) = {ratio.min():.2f}")
pax = ax[1,3].pcolormesh(ff*100, fufu*100, Z, vmin=1, vmax=4, edgecolor='w', lw=0.002)
cb = fig.colorbar(pax, ax=ax[1,3], ticks=[1,2,3,4], orientation='vertical')
cb.ax.set_yticklabels([r'$10^{1}$', r'$10^{2}$',r'$10^{3}$',r'$10^{4}$'])
ax[1,3].set_ylabel(r'$\nu f$ $(\times 10^{-2})$')
ax[1,3].set_xlabel(r'$f$ $(\times 10^{-2})$')
ax[1,3].set_yticks([1,3,5], ['1', '3', '5'])
ax[1,3].set_xticks([5,10,15,20], ['5', '10', '15', '20'])
ax[1,3].set_title(r'$T^\mathrm{PTD}_{X\to Y} / T^\mathrm{PTD}_{Y\to Z}$', fontsize=25, pad=16)
plt.tight_layout()
fig.savefig(root/'fig_nc/pdf'/'fig2.pdf', transparent=True)
#%%