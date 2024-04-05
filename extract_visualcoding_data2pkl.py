# %% [markdown]
# # Visual Coding - Neuropixels
# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
data_directory = Path('./data/neuropixel/')
cache = EcephysProjectCache.from_warehouse(manifest=data_directory/"manifest.json")
sessions = cache.get_session_table()
for session_id in sessions[sessions.session_type.eq('brain_observatory_1.1')].index.values:
    #%%
    # session_id = 774875821 #715093703
    # session_id = 715093703
    nwb_path = data_directory/f'session_{session_id:d}.nwb'
    session_id = int(nwb_path.stem.split('_')[-1])
    session = EcephysSession.from_nwb_path(nwb_path, api_kwargs={
            "amplitude_cutoff_maximum": np.inf,
            "presence_ratio_minimum": -np.inf,
            "isi_violations_maximum": np.inf
        })

    out_dir = Path(f'./visualcoding/{session_id:d}/')
    out_dir.mkdir(parents=True, exist_ok=True)
    # print arguments
    # print([attr_or_method for attr_or_method in dir(session) if attr_or_method[0] != '_'])
    # %%
    stimulus_block = session.stimulus_presentations.groupby(['stimulus_block','stimulus_name']).duration.sum().reset_index()
    stimulus_block['stimulus_name'].unique()

    # %%
    units = cache.get_units(amplitude_cutoff_maximum = np.inf,
                            presence_ratio_minimum = -np.inf,
                            isi_violations_maximum = np.inf)

    len(units)
    # %%
    from scipy.ndimage import gaussian_filter1d
    plt.rcParams.update({'font.size': 14})

    def plot_metric(data, bins, x_axis_label, color, max_value=-1):
        
        h, b = np.histogram(data, bins=bins, density=True)

        x = b[:-1]
        y = gaussian_filter1d(h, 1)

        plt.plot(x, y, color=color)
        plt.xlabel(x_axis_label)
        plt.gca().get_yaxis().set_visible(False)
        [plt.gca().spines[loc].set_visible(False) for loc in ['right', 'top', 'left']]
        if max_value < np.max(y) * 1.1:
            max_value = np.max(y) * 1.1
        plt.ylim([0, max_value])
        
        return max_value
    region_dict = {'cortex' : ['VISp', 'VISl', 'VISrl', 'VISam', 'VISpm', 'VIS', 'VISal','VISmma','VISmmp','VISli'],
                'thalamus' : ['LGd','LD', 'LP', 'VPM', 'TH', 'MGm','MGv','MGd','PO','LGv','VL',
                                'VPL','POL','Eth','PoT','PP','PIL','IntG','IGL','SGN','VPL','PF','RT'],
                'hippocampus' : ['CA1', 'CA2','CA3', 'DG', 'SUB', 'POST','PRE','ProS','HPF'],
                'midbrain': ['MB','SCig','SCiw','SCsg','SCzo','PPT','APN','NOT','MRN','OP','LT','RPF','CP']}

    color_dict = {'cortex' : '#08858C',
                'thalamus' : '#FC6B6F',
                'hippocampus' : '#7ED04B',
                'midbrain' : '#FC9DFE'}

    bins = np.linspace(-3,2,100)
    max_value = -np.inf

    plt.clf()
    for idx, region in enumerate(region_dict.keys()):
        
        data = np.log10(units[units.ecephys_structure_acronym.isin(region_dict[region])]['firing_rate'])
        
        max_value = plot_metric(data, bins, 'log$_{10}$ firing rate (Hz)', color_dict[region], max_value)
        
    _ = plt.legend(region_dict.keys())
    plt.tight_layout()
    plt.savefig(out_dir/'firing_rate_distribution.png', dpi=300)
    # %%
    print(f'{session.units.shape[0]} units total')
    units_high_snr = session.units[session.units['snr'] > 4]
    units_high_fr = session.units[session.units['firing_rate'] > 0.05]

    units_chosen = session.units[(session.units['snr'] > 4)]# * (session.units['firing_rate'] > 0.05)]
    # drop abnormal unit
    # units_chosen = units_chosen.drop(950942603)
    print(f'{units_high_snr.shape[0]} units have snr > 4')
    print(f'{units_high_fr.shape[0]} units have firing_rate > 0.05')

    # grab an arbitrary (though high-snr!) unit (we made units_with_high_snr above)
    # high_snr_unit_ids = units_with_very_high_snr.index.values
    high_snr_unit_ids = units_chosen.index.values
    unit_id = high_snr_unit_ids[0]

    # %%
    stimulus_names = [
        'drifting_gratings',
        'static_gratings',
        'natural_scenes',
        'natural_movie_one',
        'natural_movie_three',
        'flashes',
        'gabors',
        # 'spontaneous',
    ]

    units_chosen.to_pickle(out_dir/'units.pkl')
    data_pickle = {}
    firing_rate_selection = {key:np.zeros(units_chosen.shape[0]) for key in stimulus_names}
    #%%
    fig, ax = plt.subplots(len(stimulus_names),1, figsize=(12, len(stimulus_names)*3), sharex=True)
    for stimulus, axi in zip(stimulus_names, ax.flatten()):
        total_time_period = 0
        spike_times = []
        for block_id in stimulus_block[stimulus_block.stimulus_name == stimulus].stimulus_block.values:
            stimulus_presentation = session.stimulus_presentations[
                (session.stimulus_presentations.stimulus_block.eq(block_id))
            ]
            start_time = stimulus_presentation.start_time.min()
            end_time = stimulus_presentation.stop_time.max()

            # select all spikes responding for stimulus
            times = {key: val[(val>=start_time)*(val<end_time)]
                        for key, val in session.spike_times.items()
                            if key in units_chosen.index.values}

            # output to *.dat files
            counter = 0
            for key, val in times.items():
                if len(val) > 0:
                    spike_times.append(np.vstack((val, np.ones_like(val)*key)).T)
                    firing_rate_selection[stimulus][counter] += spike_times[-1].shape[0]
                counter += 1
            total_time_period += end_time-start_time
        firing_rate_selection[stimulus] /= total_time_period
        spike_times = np.vstack(spike_times)
        # sort spike time
        spike_times = spike_times[spike_times[:,0].argsort()]
        data_pickle[stimulus] = spike_times.copy()
        num_cells = np.unique(spike_times[:,1]).shape[0]
        print(f"{num_cells:d} units in total for {stimulus:s} section.")
        # print(len(spike_times), num_cells, total_time_period)
        print(f"mean firing rate: {len(spike_times)/num_cells/total_time_period:.3f} Hz.")
        axi.plot(spike_times[:,0], spike_times[:,1], '|')
        axi.set_title(stimulus)
        axi.set_ylabel('Neuronal Indices')

    ax[-1].set_xlabel('Time (seconds)', )
    plt.tight_layout()
    plt.savefig(out_dir/'allen_raster.png', dpi=300)
    # %%
    firing_rate_selection = pd.DataFrame(firing_rate_selection, index=units_chosen.index)
    firing_rate_selection.to_pickle(out_dir / 'firing_rate_selection.pkl')
    with open(out_dir/'preprocessed_spike_time_data.pkl', 'wb') as f:
        pickle.dump(data_pickle, f)
#%%