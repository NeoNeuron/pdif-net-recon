#%%
import pickle
import h5py
import yaml

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
plt.rcParams['font.size']=16
plt.rcParams['axes.labelsize']=16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

from pdif.pdif import CausalityEstimator
from pdif.utils import Gaussian, match_features, reconstruction_analysis, optimal_delay_estimator
from pdif.figrc import line_rc, c_inv
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
key_map = {'TE': 'TE', 'MI': 'sum(MI)', 'CC': 'sum(CC2)', 'GC': 'GC'}
#%%
for out_dir in Path('./visualbehavior/').iterdir():
    if not out_dir.is_dir():
        continue
    print(out_dir.stem)
    #%%
    # session_id = 1049273528 #1047969464
    # out_dir = Path(f"./visualbehavior/{session_id}/")
    # out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"preprocessed_spike_time_data.pkl", 'rb') as f:
        data_pickle = pickle.load(f)
        stimulus_names = list(data_pickle.keys())
        del data_pickle
    units = pd.read_pickle(out_dir / f"units.pkl")
    n_unit = len(units)
    # high_rate_mask = data_pickle['rate_tight'] >= 0.05
    # unit_rate_mask_union = data_pickle['rate_raw'] >= 0.05
    # run causality measures
    stimulus_group = {
        'all': ['active', 'passive'],
    }
    stimulus_names = np.append(stimulus_names, list(stimulus_group.keys()))
    #%%
    heter_delay_toggle = False
    # further data selection according to refractory periods
    # ! data selection configurationaccording to refractory periods
    t_ref = 5.0    # msecond
    gap_width = 250

    def fnaming(name, gap=gap_width):
        return f"{name:s}_ref={t_ref:.0f}_gap={gap:.0f}"

    # setup configurations
    order = (1,5)
    dt = 1
    delay = 0
    suffix = 250
    # ablation test: shuffle spike trains to destroy causal structure (null control)
    shuffle_toggle = True

    fig_suffix = f"ref={t_ref:.0f}-gap={gap_width:.0f}-sfx={suffix:.0f}-K={order[0]:d}_{order[1]:d}-bin={dt:.2f}"
    if not heter_delay_toggle:
        fig_suffix += f"-delay={delay:.2f}"
    if shuffle_toggle:
        fig_suffix += "-shuffle"

    # %
    #! ====================
    #! Draw histogram of causal values for each stimuli
    #! ====================
    stimulus_names_plot = ['active', 'passive', ]
    # stimulus_names_plot = ['flashes', 'gabors', 'spontaneous']
    # fig_sfx = '_new3' 
    gap_vals = np.ones(len(stimulus_names_plot),dtype=int)*gap_width
    # gap_vals = [1, 250, 250, 250,]
    # sfx = np.ones(len(stimulus_names_plot), dtype=int)*suffix
    # sfx[-2] = 500
    # fig_sfx = '_raw4'

    hf = h5py.File(out_dir / 'metadata_firing_rate.h5','r')
    new_rate = np.array([hf[fnaming(stimulus)][:] for stimulus in ['active', 'passive']])
    # new_rate = np.array([hf[fnaming(stimulus)][:] for stimulus in stimulus_names_plot])
    high_rate_mask = new_rate >= 0.08#0.335
    unit_rate_mask_union = high_rate_mask.sum(0) == high_rate_mask.shape[0]
    print(f">> {np.sum(unit_rate_mask_union):d} units are under mask")
    #%
    N = n_unit
    pm = dict(
        spk_fname = fnaming(stimulus_names_plot[0], gap_vals[0]),
        N = n_unit,
        order = order,
        T = hf[fnaming(stimulus_names_plot[0])].attrs['T'] + suffix*1e3,
        DT = 2e4,
        dt = dt,
        delay = delay,
        path = str(out_dir)+'/',
        shuffle = shuffle_toggle,
    )
    estimator = CausalityEstimator(**pm, n_thread=60)
    #%%
    # delays = np.arange(21)
    # optimal_delay = estimator.get_optimal_delay(delays)
    # print(optimal_delay)
    # tmp = []
    # for delay_ in delays:
    #     estimator.delay = delay_
    #     data = estimator.fetch_data()
    #     data['delay'] = delay_
    #     tmp.append(data)
    # data = pd.concat(tmp)
    # #%%
    # # sns.pointplot(data=data[(data['pre_id']==120)], x='delay', y='TE', stats='se')
    # sns.pointplot(data=data, x='delay', y='TE', errorbar='se')
        

    #%%
    # Load parameters from yaml file
    fit_p0_default = [0.5, -5.5, -4.2, .1, .1]
    fit_p0 = fit_p0_default
    if (out_dir/'fig_p0.yml').exists():
        with open(out_dir/'fig_p0.yml', 'r') as file:
            parameters = yaml.safe_load(file)
        if fig_suffix in parameters:
            fit_p0 = parameters[fig_suffix]
        else:
            with open(out_dir/'fig_p0.yml', 'a') as file:
                yaml.dump({fig_suffix: fit_p0_default}, file)
    else:
        with open(out_dir/'fig_p0.yml', 'w') as file:
            yaml.dump({fig_suffix: fit_p0_default}, file)

    hf = h5py.File(out_dir / 'metadata_firing_rate.h5','r')
    new_N = int(np.sum(unit_rate_mask_union))
    chosen_unit_set = np.nonzero(unit_rate_mask_union)[0]
    fig, ax = plt.subplots(1,3, figsize=(18,6))
    data_fig_all = {}
    data_recon_list = []

    delays = np.arange(21)
    for stimulus_, gap_, axi in zip(stimulus_names_plot, gap_vals, ax.flatten()):
        # fetch causality data
        estimator.spk_fname = fnaming(stimulus_, gap_)
        estimator.T = hf[fnaming(stimulus_)].attrs['T'] + suffix*1e3
        # print(estimator.get_optimal_delay(delays))
        if heter_delay_toggle:
            data = optimal_delay_estimator(estimator, delays)
        else:
            data = estimator.fetch_data(new_run=True)
        data = data[(data['pre_id'].isin(chosen_unit_set)) & (data['post_id'].isin(chosen_unit_set))].copy()
        data_matched = match_features(data, N=new_N)
        vrange=(-8,-2)
        data_recon, data_fig = reconstruction_analysis(data_matched, nbins=60, hist_range=vrange, fit_p0=fit_p0, algorithm='EM')
        data_fig = data_fig.dropna(axis=1, how='all')
        data_fig_all[stimulus_] = data_fig.copy()
        data_recon['stimulus'] = stimulus_
        data_recon_list.append(data_recon.copy())

        ax_hist = inset_axes(axi, width="100%", height="100%",
                        bbox_to_anchor=(.2, .2, .65, .55),
                        bbox_transform=axi.transAxes, loc='center', axes_kwargs={'facecolor':[1,1,1,0]})

        for key in ('CC', 'MI', 'GC', 'TE'):
            edges = data_fig['edges'][key]
            ax_hist.plot(edges, data_fig['hist'][key], **line_rc[key])
            if 'log_norm_fit_pval' not in data_fig:
                continue
            pval = data_fig['log_norm_fit_pval'][key].copy()
            if not hasattr(pval, '__len__'):
                continue
            gauss1 = Gaussian(edges, pval[1], pval[3]) * (1-pval[0])
            gauss2 = Gaussian(edges, pval[2], pval[4]) * pval[0]
            # print(pval[0])
            ax_hist.plot(edges,gauss1, color=line_rc[key]['color'], ls='--')
            ax_hist.plot(edges,gauss2, color=line_rc[key]['color'], ls='--')
            ax_hist.axvline(data_fig['th_gauss'][key], color=line_rc[key]['color'], ls='--')
            fpr, tpr = data_fig['roc_blind'][key]
            axi.plot(fpr, tpr, color=line_rc[key]['color'], lw=line_rc[key]['lw']*2, label=line_rc[key]['label'])[0].set_clip_on(False)

            print(f"{key:s}: {data_fig['auc_gauss'][key]:.3f}", end='\t')
        ax_hist.spines['left'].set_visible(False)
        ax_hist.set_yticks([])
        # ax_hist.set_xlim(*vrange)
        # xticks = np.arange(-8, -1, 2)
        # ax_hist.set_xticks(xticks)
        ax_hist.set_xticklabels([r"$10^{%.0f}$"%val for val in ax_hist.get_xticks()])
        print('')
        ax_hist.set_ylim(0)
        axi.legend(loc='upper right')
        # if '-' in stimulus_:
        #     arr_image = plt.imread('../'+stimulus_.split('-')[0]+'.png', format='png')
        # else:
        #     arr_image = plt.imread('../'+stimulus_+'.png', format='png')
        # axins = inset_axes(axi, width="100%", height="100%",
        #                 bbox_to_anchor=(.05, .05, .23, .23),
        #                 bbox_transform=axi.transAxes, loc='center')

        # axins.imshow(arr_image)
        # axins.axis('off')
        # ax_hist.set_ylabel('probability density')
        # ax_hist.set_xlabel('causal value')
        axi.set_xlim(0,1)
        axi.set_ylim(0,1)
    ax[0].set_ylabel('True Positive Rate', fontsize=30)
    [axi.set_xlabel('False Positive Rate', fontsize=30) for axi in ax]

    data_recon = pd.concat(data_recon_list)

    plt.tight_layout()
    plt.savefig(out_dir/f"hist-all-allen-{fig_suffix:s}.pdf")
    hf.close()

    #%%
    #! ====================
    #! Draw histogram of causal values for each stimuli filtered with inconsistent masks
    #! ====================
    tmp = pd.DataFrame(
        data_recon.groupby(['pre_id', 'post_id']).sum()[
            ['recon-gauss-CC', 'recon-gauss-MI', 'recon-gauss-GC', 'recon-gauss-TE']])
    for key in ('CC', 'MI', 'GC', 'TE'):
        print(f"{key:s}: {tmp['recon-gauss-'+key].eq(1).mean()*100:>5.2f} %")
    #%%

    inconsist_hist = {key:{} for key in stimulus_names_plot}
    for key in ('CC', 'MI', 'GC', 'TE'):
        mask = tmp['recon-gauss-'+key].eq(1)
        selected_data = pd.merge(tmp[mask], data_recon, how='left', left_index=True, right_on=['pre_id', 'post_id'])
        fig, ax = plt.subplots(1,3, figsize=(16,5))
        for stim, axi in zip(stimulus_names_plot, ax.flatten()):
            vrange=(-8,-2)
            counts, bins = np.histogram(selected_data[selected_data['stimulus'].eq(stim)]['log-'+key], bins=60, range=vrange, density=True)
            pop_ratio = tmp['recon-gauss-'+key].eq(1).mean()
            axi.plot(bins[:-1], gaussian_filter1d(counts*pop_ratio, 1), lw=6, color='#00C2A0', label=line_rc[key]['label'], zorder=1)
            # axi.plot(bins[:-1], counts*pop_ratio, lw=6, color='#00C2A0', label=line_rc[key]['label'], zorder=1)
            inconsist_hist[stim][key] = counts*pop_ratio,
            # print(pop_ratio, counts.sum()*pop_ratio*(np.diff(bins)[0]))
            if 'log_norm_fit_pval' not in data_fig_all[stim]:
                continue
            popt = data_fig_all[stim]['log_norm_fit_pval'][key]
            if not hasattr(popt, '__len__'):
                continue
            axi.plot(bins[:-1], Gaussian(bins[:-1], popt[1], popt[3])*(1-popt[0]), lw=4, alpha=1.0, color=line_rc[key]['color'],zorder=0)
            axi.plot(bins[:-1], Gaussian(bins[:-1], popt[2], popt[4])*(popt[0]), lw=4, alpha=1.0, color=c_inv[line_rc[key]['color']],zorder=0)
            # calculate threshold
            axi.axvline(data_fig_all[stim]['th_gauss'][key], color='k', ls='-')

            axi.set_xlim(*vrange)
            xticks = np.arange(-8, -1, 2)
            axi.set_xticks(xticks)
            axi.set_xticklabels([r"$10^{%.0f}$"%val for val in axi.get_xticks()])
            axi.set_ylim(0)
            # if '-' in stim:
            #     arr_image = plt.imread('../'+stim.split('-')[0]+'.png', format='png')
            # else:
            #     arr_image = plt.imread('../'+stim+'.png', format='png')
            # axins = inset_axes(axi, width="100%", height="100%",
            #                 bbox_to_anchor=(.05, .75, .23, .23),
            #                 bbox_transform=axi.transAxes, loc='center')

            # axins.imshow(arr_image)
            # axins.axis('off')
        ax[0].set_ylabel('probability density', fontsize=30)
        [axi.set_xlabel(key, fontsize=30) for axi in ax];

        plt.tight_layout()
        plt.savefig(out_dir/f"hist-inconsist-{key:s}_allen-{fig_suffix:s}.pdf")

    for key in data_fig_all.keys():
        data_fig_all[key] = data_fig_all[key].merge(
            pd.DataFrame(inconsist_hist[key], index=['hist_inconsist']).T,
            left_index=True, right_index=True, how='left')
    #%%
    #! Draw the heatmap of correlation coefficient matrix
    #! ----
    data_fig_all['consistency']={}
    data_fig_all['consistency_binary']={}
    for idx, key in enumerate(('CC', 'MI', 'GC', 'TE')):
        tmp = data_recon[['stimulus', 'pre_id', 'post_id', 'log-'+key, 'recon-gauss-'+key]]
        # causality data
        causal_data = pd.concat(
            [df[1].set_index(['pre_id', 'post_id'])['log-'+key].rename(df[0])
             for df in list(tmp.groupby('stimulus'))],
            axis=1)
        causal_data.dropna(inplace=True)
        data = np.corrcoef(causal_data.to_numpy().T)
        data_fig_all['consistency'][key]=data
        # reconstructed data
        recon_data = pd.concat(
            [df[1].set_index(['pre_id', 'post_id'])['recon-gauss-'+key].rename(df[0])
             for df in list(tmp.groupby('stimulus'))],
            axis=1)
        recon_data.dropna(inplace=True)
        data_fig_all['consistency_binary'][key]=np.corrcoef(recon_data.to_numpy().T)
        mask = np.triu(np.ones_like(data, dtype=bool),k=1)
        fig, g = plt.subplots(1,1, figsize=(10,10), dpi=200, 
            gridspec_kw=dict(bottom=0.2, left=0.2, top=0.95, right=0.95))
        g = sns.heatmap(data, mask=mask,
            vmin=0, vmax=1, 
            cmap=plt.cm.OrRd, 
            square=True,
            lw=.5,
            ax=g,
            annot=True,
            annot_kws={"fontsize":25}
            )

        g.set_xticklabels([])
        g.set_yticklabels([])

        length = 1./data.shape[0]-0.01
        # draw x-axis
        # for i in range(len(stimulus_names_plot)):
        #     if '-' in stimulus_names_plot[i]:
        #         arr_image = plt.imread('../'+stimulus_names_plot[i].split('-')[0]+'.png', format='png')
        #     else:
        #         arr_image = plt.imread('../'+stimulus_names_plot[i]+'.png', format='png')
        #     axins = inset_axes(g, width="100%", height="100%",
        #                     bbox_to_anchor=(.005+i*(length+0.01), -length-0.01, length, length),
        #                     bbox_transform=g.transAxes, loc='center')

        #     axins.imshow(arr_image)
        #     axins.axis('off')

        # # draw y-axis
        # for i in range(len(stimulus_names_plot)):
        #     if '-' in stimulus_names_plot[i]:
        #         arr_image = plt.imread('../'+stimulus_names_plot[i].split('-')[0]+'.png', format='png')
        #     else:
        #         arr_image = plt.imread('../'+stimulus_names_plot[i]+'.png', format='png')
        #     axins = inset_axes(g, width="100%", height="100%",
        #                     bbox_to_anchor=(-length-0.01, 1-length-0.005-i*(length+0.01), length, length),
        #                     bbox_transform=g.transAxes, loc='center')

        #     axins.imshow(arr_image)
        #     axins.axis('off')

        plt.savefig(out_dir/f"visualbehavior-RSA4-{key:s}-{fig_suffix:s}.pdf")
        print(f"Minimum coincidence rate : {data.min():6.3f}")
        print(f"Maximum coincidence rate : {np.sort(np.unique(data))[-2]:6.3f}")
        print(data)

    #%%
    #! ============================================================
    #! Draw the histogram of dp and Delta p for each stimuli
    #! ============================================================
    N = n_unit
    new_N = int(np.sum(unit_rate_mask_union))
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    for i, axi in enumerate(ax.flatten()[:len(stimulus_names_plot)]):
        bins = data_fig_all[stimulus_names_plot[i]]['edges']['dp']
        counts = data_fig_all[stimulus_names_plot[i]]['hist']['dp']
        axi.plot(bins, counts, label=r'$\Delta_p$', lw=4)
        axi.axvline(0, ls=':', color='r')
        axi.set_ylim(0)
        axi.set_xlim(-4,1)
        xticks = np.array([-4,-2,0,1])
        axi.set_xticks(xticks)
        # axi.set_xticklabels([r'$10^{%d}$'%val for val in xticks])

        # if '-' in stimulus_names_plot[i]:
        #     arr_image = plt.imread('../'+stimulus_names_plot[i].split('-')[0]+'.png', format='png')
        # else:
        #     arr_image = plt.imread('../'+stimulus_names_plot[i]+'.png', format='png')
        # axins = inset_axes(axi, width="100%", height="100%",
        #                 bbox_to_anchor=(.05, .3, .23, .23),
        #                 bbox_transform=axi.transAxes, loc='center')

        # axins.imshow(arr_image)
        # axins.axis('off')
    [axi.set_xlabel(r'$|\Delta p_m|$ values', fontsize=30) for axi in ax]
    ax[0].set_ylabel('probability density', fontsize=24)
    plt.tight_layout()
    plt.savefig(out_dir/f"hist-dp-{fig_suffix:s}.pdf")

    data_recon.to_pickle(out_dir/f"reconstruction-data-{fig_suffix:s}.pkl")
    with open(out_dir/f'allen-data-{fig_suffix:s}.pkl', 'wb') as f:
        pickle.dump(data_fig_all, f)

# %%
# drifting_gratings :   1884568 ms
# static_gratings :     1503250 ms
# natural_scenes :      1490757 ms
# natural_movie_one :   601496  ms
# natural_movie_three : 1202032 ms
# 1884568+1503250+1490757+601496+1202032