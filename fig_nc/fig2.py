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
from figrc import *
from matplotlib.ticker import ScalarFormatter

fig, ax = plt.subplots(
    2,4,figsize=(14,7),)# gridspec_kw=dict(wspace=0.3, hspace=0.5, width_ratios=(0.8,1,1,1))
# )

G = nx.DiGraph()
G.add_edges_from([(1,2), (2,3)])
G = nx.relabel_nodes(G, {1:'X', 2:'Y', 3:'Z'})
pos = {
    n: coordinate 
    for n, coordinate in zip(G,((1,np.sqrt(3)),(0,0),(2,0),))
}
make_graph_diagram(G, ax[0,0], pos,)
ax[0,0].set_xlim(-1,3)
ax[0,0].set_ylim(-1,np.sqrt(3)+1)

estimator = CausalityEstimator(
    # root/'HH3-chain', 'HHp=0.25s=0.020f=0.080u=0.150', 3, delay=3, T=1e7,
    root/'HH3-chain', 'HHp=0.25s=0.020f=0.200u=0.050', 3, delay=3, T=1e7,
)
data = estimator.fetch_data()
data_matched = match_features(data, 3, root/'HH3-chain/connect_matrix-p=0.250.npy')
gt = np.load(root/'HH3-chain/connect_matrix-p=0.250.npy')
TE = np.zeros((3,3))
TE[~np.eye(3, dtype=bool)] = data_matched['TE']
sns.heatmap(gt, linecolor='#DDDDDD', lw=2, ax=ax[0,1], square=True, cmap='Oranges', cbar=False)
sns.heatmap(TE, linecolor='#DDDDDD', lw=2, ax=ax[0,2], square=True, cmap='Oranges', cbar_kws={'label':'PTD-TE value'})
cb_ax = ax[1,2].figure.axes[-1]
cb_ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
cb_ax.yaxis.offsetText.set_horizontalalignment('left')
cb_ax.yaxis.offsetText.set_position((0,0))
for axi in ax[0,1:]:
    axi.set_xticks(np.arange(3)+0.5, ['X', 'Y', 'Z'], fontsize=16)
    axi.set_yticks(np.arange(3)+0.5, ['X', 'Y', 'Z'], fontsize=16)
    axi.set_xlabel('To', fontsize=16)
    axi.set_ylabel('From', fontsize=16)
ax[0,1].set_title('Ground truth', fontsize=20, pad=16)
ax[0,2].set_title('PTD-TE', fontsize=20, pad=16)
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
pax = ax[0,3].pcolormesh(ff, fufu, Z, vmin=2, vmax=4, edgecolor='w', lw=0.002)
cb = fig.colorbar(pax, ax=ax[0,3], ticks=[2,3,4], orientation='vertical')
cb.ax.set_yticklabels([r'$10^{2}$',r'$10^{3}$',r'$10^{4}$'])
ax[0,3].set_ylabel(r'$\nu f\,\,(\mathrm{mS}\cdot\mathrm{cm}^{-2}\cdot\mathrm{ms}^{-1})$', fontsize=14)
ax[0,3].set_xlabel(r'$f\,\,(\mathrm{mS}\cdot\mathrm{cm}^{-2})$', fontsize=14)
ax[0,3].set_yticks([0.01,0.03,0.05], ['1', '3', '5'], fontsize=16)
ax[0,3].set_xticks([0.05,0.1,0.15,0.2], ['5', '10', '15', '20'], fontsize=16)

# Customize/force offset text and give it new content
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))  # always use scientific notation

for axis, pos in zip([ax[0,3].xaxis, ax[0,3].yaxis], [(0.9,0), (0,0)]):
    axis.set_major_formatter(formatter)
    # Need a draw so formatter computes the order of magnitude
    fig.canvas.draw_idle()
    off = axis.get_offset_text()
    custom_exponent = -2  # change as needed
    off.set_text(fr'$\times 10^{{{custom_exponent}}}$')
    off.set_horizontalalignment('left')
    off.set_position(pos)
    off.set_visible(True)

ax[0,3].set_title(r'$T^\mathrm{PTD}_{X\to Y} / T^\mathrm{PTD}_{X\to Z}$', fontsize=22, pad=0)

#%
G = nx.DiGraph()
G.add_edges_from([(1,2), (1,3)])
G = nx.relabel_nodes(G, {1:'X', 2:'Y', 3:'Z'})
pos = {
    n: coordinate 
    for n, coordinate in zip(G,((1,np.sqrt(3)),(0,0),(2,0),))
}
make_graph_diagram(G, ax[1,0], pos)
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
sns.heatmap(TE, linecolor='#DDDDDD', lw=2, ax=ax[1,2], square=True, cmap='Oranges', cbar_kws={'label':'PTD-TE value'})
cb_ax = ax[1,2].figure.axes[-1]
cb_ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
cb_ax.yaxis.offsetText.set_horizontalalignment('left')
cb_ax.yaxis.offsetText.set_position((0,0))
for axi in ax[1,1:]:
    axi.set_xticks(np.arange(3)+0.5, ['X', 'Y', 'Z'], fontsize=16)
    axi.set_yticks(np.arange(3)+0.5, ['X', 'Y', 'Z'], fontsize=16)
    axi.set_xlabel('To', fontsize=16)
    axi.set_ylabel('From', fontsize=16)
ax[1,1].set_title('Ground truth', fontsize=20, pad=16)
ax[1,2].set_title('PTD-TE', fontsize=20, pad=16)
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
pax = ax[1,3].pcolormesh(ff, fufu, Z, vmin=1, vmax=4, edgecolor='w', lw=0.002)
cb = fig.colorbar(pax, ax=ax[1,3], ticks=[1,2,3,4], orientation='vertical')
cb.ax.set_yticklabels([r'$10^{1}$', r'$10^{2}$',r'$10^{3}$',r'$10^{4}$'])
ax[1,3].set_ylabel(r'$\nu f\,\,(\mathrm{mS}\cdot\mathrm{cm}^{-2}\cdot\mathrm{ms}^{-1})$', fontsize=14)
ax[1,3].set_xlabel(r'$f\,\,(\mathrm{mS}\cdot\mathrm{cm}^{-2})$', fontsize=14)
ax[1,3].set_yticks([0.01,0.03,0.05],['1', '3', '5'], fontsize=16)
ax[1,3].set_xticks([0.05,0.1,0.15,0.20],['5', '10', '15', '20'], fontsize=16)

# Customize/force offset text and give it new content
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))  # always use scientific notation

for axis, pos in zip([ax[1,3].xaxis, ax[1,3].yaxis], [(0.9,0), (0,0)]):
    axis.set_major_formatter(formatter)
    # Need a draw so formatter computes the order of magnitude
    fig.canvas.draw_idle()
    off = axis.get_offset_text()
    custom_exponent = -2  # change as needed
    off.set_text(fr'$\times 10^{{{custom_exponent}}}$')
    off.set_horizontalalignment('left')
    off.set_position(pos)
    off.set_visible(True)

ax[1,3].set_title(r'$T^\mathrm{PTD}_{X\to Y} / T^\mathrm{PTD}_{Y\to Z}$', fontsize=22, pad=0)
plt.tight_layout()

for tag, axi in zip('abcdefgh', ax.flatten()):
    if tag in 'ae':
        axi.text(-0., 1.25, tag, transform=axi.transAxes,
                 fontsize=32, fontweight='bold', va='top')
    else:
        axi.text(-0.25, 1.25, tag, transform=axi.transAxes,
                fontsize=32, fontweight='bold', va='top')

fig.savefig(root/'fig_nc/pdf'/'fig2.pdf', transparent=True)
#%%