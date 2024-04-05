# %% [markdown]
# # Visual Behavior - Neuropixels
# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import struct
import pickle
from causal4.figrc import fig_path, data_path
from pathlib import Path

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
from scipy.ndimage import gaussian_filter1d
plt.rcParams.update({'font.size': 14})

# load metadata
cache_dir = "/tmp/vbn_s3_cache"
cache_dir = Path(cache_dir)
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
cache.load_manifest('visual-behavior-neuropixels_project_manifest_v0.5.0.json')

# %% view total number of sessions
ecephys_sessions = cache.get_ecephys_session_table()

print(f"Total number of ecephys sessions: {len(ecephys_sessions)}")

ecephys_sessions.head()
#%%
# session_id = 1052342277
for session_id in ecephys_sessions.index.values:
    #%%
    session_id = 1049273528
    session = cache.get_ecephys_session(ecephys_session_id=session_id)

    out_dir = Path(f"./visualbehavior/{session_id}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    # %%
    # print arguments
    print(*[attr_or_method for attr_or_method in dir(session) if attr_or_method[0] != '_'], sep='\n')

    # %%
    units = cache.get_unit_table()
    #%%
    units = units[units.ecephys_session_id == session_id]
    print(len(units))
    # %%
    def plot_metric(data, bins, x_axis_label, color, max_value=-1):
        
        h, b = np.histogram(data, bins=bins, density=True)

        x = b[:-1]
        y = gaussian_filter1d(h, 2)

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
        
        data = np.log10(units[units.structure_acronym.isin(region_dict[region])]['firing_rate'])
        
        max_value = plot_metric(data, bins, 'log$_{10}$ firing rate (Hz)', color_dict[region], max_value)
        
    _ = plt.legend(region_dict.keys())
    plt.tight_layout()
    plt.savefig(out_dir/'population_firing_rate.png', dpi=300)
    # %%
    print(f'{units.shape[0]} units total')
    units_high_snr = units[units['snr'] > 4]
    units_high_fr = units[units['firing_rate'] > 0.05]

    units_chosen = units[(units['snr'] > 4)]# * (session.units['firing_rate'] > 0.05)]
    # drop abnormal unit
    # units_chosen = units_chosen.drop(950942603)
    print(f'{units_high_snr.shape[0]} units have snr > 4')
    print(f'{units_high_fr.shape[0]} units have firing_rate > 0.05')

    # grab an arbitrary (though high-snr!) unit (we made units_with_high_snr above)
    # high_snr_unit_ids = units_with_very_high_snr.index.values
    high_snr_unit_ids = units_chosen.index.values
    unit_id = high_snr_unit_ids[0]
    # columns to be used: unit_id (neuron index), firing_rate, structure_acronym
    units_chosen.to_pickle(out_dir/'units.pkl')

    # %%
    # stimulus_names = session.stimulus_presentations.stimulus_name.unique()  # all stimulus names
    stimulus_names = ['active', 'passive']
    stimulus_blocks = [0, 5]
    
    #%% [markdown]
    #  # Information about stimulus blocks
    # 6 differnet blocks in total:
    # Block index and corresponding stimulus name:
    # 
    # 0. active behavior: Natural_Images_Lum_Matched_set_ophys_H_2019
    # 1. spontaneous
    # 2. gabors : gabor_20_deg_250ms
    # 3. spontaneous
    # 4. flashes : flash_250ms
    # 5. passive viewing images
    #%%
    data_pickle = {}
    firing_rate_selection = {key:[] for key in stimulus_names}
    fig, ax = plt.subplots(len(stimulus_blocks),1, figsize=(12, len(stimulus_blocks)*3), sharex=True)
    for axi, stimulus, stimulus_block_id in zip(ax, stimulus_names, stimulus_blocks):
        # get spike times from the first block of drifting gratings presentations 
        stimulus_presentation = session.stimulus_presentations[
            (session.stimulus_presentations.stimulus_block.eq(stimulus_block_id))
        ]
        start_time = stimulus_presentation.start_time.min()
        end_time = stimulus_presentation.end_time.max()

        # select all spikes responding for drift gratings
        times = {key: val[(val>=start_time)*(val<end_time)]
                    for key, val in session.spike_times.items()
                        if key in units_chosen.index.values}
        # output to *.dat files
        spike_times = []
        for key, val in times.items():
            if len(val) > 0:
                spike_times.append(np.vstack((val, np.ones_like(val)*key)).T)
                firing_rate_selection[stimulus].append(spike_times[-1].shape[0] / (end_time-start_time))
            else:
                firing_rate_selection[stimulus].append(0.0)

        spike_times = np.vstack(spike_times)
        # sort spike time
        spike_times = spike_times[spike_times[:,0].argsort()]
        data_pickle[stimulus] = spike_times.copy()
        num_cells = np.unique(spike_times[:,1]).shape[0]
        print(f"{num_cells:d} units in total for {stimulus:s} section.")
        print(f"mean firing rate: {len(spike_times)/num_cells/(end_time-start_time):.3f} Hz.")
        axi.plot(spike_times[:,0], spike_times[:,1], '|')
        axi.set_title(stimulus)
        axi.set_ylabel('Neuronal Indices')

    firing_rate_selection = pd.DataFrame(firing_rate_selection, index=units_chosen.index)
    firing_rate_selection.to_pickle(out_dir / 'firing_rate_selection.pkl')

    ax[-1].set_xlabel('Time (seconds)', )
    plt.tight_layout()
    plt.savefig(out_dir/'allen_raster.png', dpi=300)
    with open(out_dir/f'preprocessed_spike_time_data.pkl', 'wb') as f:
        pickle.dump(data_pickle, f)
    #%%