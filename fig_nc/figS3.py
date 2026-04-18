#%%
from figrc import *
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize']  = 26
plt.rcParams['axes.titlesize']  = 26
import pickle
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
data_path = Path(__file__).resolve().parents[1] / 'data'
def Linear_R2(x:np.ndarray, y:np.ndarray, pval:np.ndarray)->float:
    """Compute R-square value for poly fitting.

    Args:
        x (np.ndarray): variable of function
        y (np.ndarray): image of function
        pval (np.ndarray): parameter of linear fitting

    Returns:
        float: R square value
    """
    mask = ~np.isnan(x)*~np.isnan(y)*~np.isinf(x)*~np.isinf(y)# filter out nan
    deg = len(pval)
    if deg < 2:
        raise ValueError(f'len(pval) must be greater than 2, (len(pval) = {deg:d})')
    y_predict = np.zeros_like(y[mask])
    for i in np.arange(deg, dtype=int):
        y_predict += pval[i]*x[mask]**(deg-i-1)
    R = np.corrcoef(y[mask], y_predict)[0,1]
    return R**2

def move_axis_offset(ax:plt.Axes, units=None):
    if units is None:
        units = ('', '')
    elif isinstance(units, str):
        units = (units, units)
    for axis, unit in zip([ax.xaxis, ax.yaxis], units):
        axis.get_offset_text().set_visible(False)
        offset_text = axis.get_offset_text().get_text()
        if offset_text:
            if unit:
                axis.set_label_text(axis.get_label_text()+' (%s '%offset_text+unit+')')
            else:
                axis.set_label_text(axis.get_label_text()+' (%s)'%offset_text)
        else:
            if unit:
                axis.set_label_text(axis.get_label_text()+' (%s)'%unit)
    return ax

with open(data_path/'HH100_conn_types_prl.pkl', 'rb') as file:
    data_raw = pickle.load(file)
conn_ = 'HH100-LN'
conn_name_ = 'Log-normal'
fig,ax=plt.subplots(1,2,figsize=(15,5), gridspec_kw={'top':0.96, 'bottom':0.1, 'left':0.1, 'right':0.90, 'hspace':0.4, 'wspace':0.3})
data = pd.DataFrame(data_raw[conn_]).loc['TE']
# TE histogram
# mask = np.ones_like(data['hist_conn'], dtype=bool)
RED, GREEN = '#F49227', '#194955'
mask = data['hist_conn']>0
ax[0].plot(data['edges'][mask], data['hist_conn'][mask], color=RED, lw=5, label='PDIF with A_{ij}=1')
ax[0].fill_between(data['edges'][mask], 0, data['hist_conn'][mask], color=RED, alpha=0.5)
# mask = np.ones_like(data['hist_disconn'], dtype=bool)
mask = data['hist_disconn']>0
ax[0].plot(data['edges'][mask], data['hist_disconn'][mask], color=GREEN, lw=5, label='PDIF with A_{ij}=0')
ax[0].fill_between(data['edges'][mask], 0, data['hist_disconn'][mask], color=GREEN, alpha=0.5)
ymax = np.hstack((data['hist_conn'], data['hist_disconn'])).max()
ax[0].set_ylim(0)
ax[0].set_xlabel('PDIF value')
ax[0].set_ylabel('density')
print(f"{conn_name_:15s} recon acc : {data['acc_gauss']*100:6.3f} %")
ax[0].axvline(data['kmean_th'], ymax=ymax/ax[0].get_ylim()[1], color='#F26A9D',lw=4, label='Threshold')
ax[0].xaxis.set_major_formatter(sci_formatter)
ax[0].set_xlim(-9,-4)
add_log_minor_ticks(ax[0], (-9, -4), where='x')
ax[0].tick_params(axis='x', which='major', length=6)
ax[0].tick_params(axis='x', which='minor', length=3)

# historgram of connection strength
conn = pd.DataFrame(data_raw[conn_]).loc['conn', 'raw_data']
axins = inset_axes(ax[0], width="100%", height="100%",
                bbox_to_anchor=(.14, .60, .35, .35),
                bbox_transform=ax[0].transAxes, loc='center')
# axins.spines['left'].set_visible(False)

counts, edges = np.histogram(conn[conn>0], bins=80)
axins.bar(edges[:-1], counts, width=edges[1]-edges[0], align='edge', color='gray')
# axins.set_xticklabels(axins.get_xticks(), fontsize=12)
axins.set_xlabel(r'S ($\mathrm{mS}\cdot \mathrm{cm}^{-2})$', fontsize=14)
axins.set_ylabel('Counts', fontsize=13, rotation=0, y=1.0, labelpad=-15)
# axins.set_title(conn_name_, fontsize=14)
axins.set_xticks([0.025, 0.05], labels=['0.025', '0.05'])

# TE v.s. S
TE = data['raw_data']
mask = conn > 0
ax[1].plot(conn[mask], 10**TE[mask], 'o', mec=GREEN, mfc='none', ms=3, alpha=0.35, clip_on=False)
ax[1].set_xlabel(r'S ($\mathrm{mS}\cdot \mathrm{cm}^{-2})$')
ax[1].set_ylabel('PDIF value')
ax[1].ticklabel_format(style='sci', scilimits=(0,1), axis='y', useMathText=True)
# quadratic fit
fit_func = lambda x, a: a*x**2
pval, _ = curve_fit(fit_func, conn[mask], 10**TE[mask])
R2 = Linear_R2(conn[mask], 10**TE[mask], np.hstack([pval,[0,0]]))
print('R^2 for PDIF vs S is %0.4f\n'%R2)
xrange = np.linspace(0, conn[mask].max(), 100)
ax[1].plot(xrange, fit_func(xrange, pval[0]), color='r', lw=1.5, label=f'$R^2$={R2:.2f}')
ax[1].set_xlim(0)
ax[1].set_ylim(0)

for axi, letter in zip(ax.flatten(), 'ab'):
    axi.text(-0.15,1.05,'%s'%letter,fontsize=28,weight='bold', transform=axi.transAxes)
fig.savefig(root / 'fig_nc/pdf/figS3.pdf', dpi=300, bbox_inches='tight', transparent=True)
# %%