#%%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size']=16
plt.rcParams['axes.labelsize']=16

from pdif.utils import force_refractory, save2bin
from pdif.pdif import CausalityEstimator
from pdif.figrc import fig_path, data_path
from multiprocessing import Pool
from pathlib import Path

import pickle
import h5py
import argparse

import warnings
warnings.filterwarnings('ignore')
# %% view total number of sessions

argparser = argparse.ArgumentParser()
argparser.add_argument('--project', type=str, default='visualcoding', help='Project name')
args = argparser.parse_args()

if args.project == 'visualcoding':
    data_dir = Path("./visualcoding/")
    stimulus_group = {
        'gratings': ['drifting_gratings', 'static_gratings'],
        'natural_movie': ['natural_movie_one', 'natural_movie_three'],
        'natural': ['natural_scenes', 'natural_movie_one', 'natural_movie_three'],
        'all': ['drifting_gratings', 'static_gratings', 'natural_scenes', 'natural_movie_one', 'natural_movie_three'],
    }
elif args.project == 'visualbehavior':
    data_dir = Path("./visualbehavior/")
    stimulus_group = {
        'all': ['active', 'passive'],
    }
else:
    raise ValueError("Invalid project name")

for out_dir in data_dir.iterdir():
    session_id=int(out_dir.stem)
    # %%
    # session_id = 1052533639
    out_dir = data_dir/f"{session_id}/"
    with open(out_dir / f"preprocessed_spike_time_data.pkl", 'rb') as f:
        data_pickle = pickle.load(f)
    # run causality measures
    stimulus_names = list(data_pickle.keys())
    stimulus_names = np.append(stimulus_names, list(stimulus_group.keys()))

    units = pd.read_pickle(out_dir / f"units.pkl")
    firing_rates = pd.read_pickle(out_dir / f"firing_rate_selection.pkl")
    #%%
    # further data selection according to refractory periods
    for t_ref in [0, 5, 10]:
        #%%
        t_ref = 5.0    # msecond
        gap_width = 250

        def fnaming(stimulus):
            return f"{stimulus:s}_ref={int(t_ref):d}_gap={int(gap_width):d}"

        n_unit = units.shape[0]
        #%%
        force_regen = False
        hf = h5py.File(out_dir/'metadata_firing_rate.h5','a')
        data_group = {key:None for key in stimulus_group.keys()}
        n_figs = len(stimulus_names)
        # fig, ax = plt.subplots(n_figs,1, figsize=(10,2*n_figs), sharex=True, 
        #     gridspec_kw=dict(top=0.95,bottom=0.05, left=0.1, right=0.95, hspace=0.5))
        # [axi.spines['top'].set_visible(False) for axi in ax]
        # [axi.spines['right'].set_visible(False) for axi in ax]
        for idx, stimulus in enumerate(data_pickle.keys()):
            if force_regen or not os.path.isfile(out_dir/f"{fnaming(stimulus):s}_spike_train.dat"):
                data_out = data_pickle[stimulus].copy()
                # change index
                for new_id, old_id in enumerate(units.index.values):
                    data_out[data_out[:,1]==old_id, 1] = new_id

                for key, val in stimulus_group.items():
                    if stimulus in val:
                        if data_group[key] is None:
                            data_group[key] = data_out.copy() 
                        else:
                            data_group[key] = np.vstack((data_group[key],data_out))

                # align temporally with the first spike
                data_out[:,0] -= data_out[0,0]

                # change the width of big gap in raster
                if gap_width > 0:
                    spk_diff = np.diff(data_out[:,0])
                    raster_gaps, = np.nonzero(spk_diff > 100)
                    for gap in raster_gaps:
                        data_out[gap+1:,0] -= data_out[gap+1,0] - data_out[gap,0] - gap_width

                data_out_filtered = force_refractory(data_out, t_ref)

                T = np.ceil(data_out_filtered[-1,0])
                # caluculate mean firing rates after downsampling with forced refractory
                rate = np.array([
                    np.sum(data_out_filtered[:,1]==i)*1000.0/T
                    # np.sum(data_out_filtered[:,1]==i)*1.0/data_pickle['stimulus_present_time'][stimulus]
                    for i in range(units.shape[0])
                ])
                if fnaming(stimulus) in hf:
                    hds = hf[fnaming(stimulus)]
                    hds[:] = rate
                else:
                    hds = hf.create_dataset(fnaming(stimulus), data=rate)
                hds.attrs['stimulus']=str(stimulus)
                hds.attrs['t_ref']=t_ref
                hds.attrs['gap_width']=gap_width
                hds.attrs['T']=T

                save2bin(
                    out_dir / f"{fnaming(stimulus):s}_spike_train.dat",
                    data_out_filtered
                    )
            else:
                data_out_filtered = np.fromfile(
                    out_dir / f"{fnaming(stimulus):s}_spike_train.dat",
                    dtype=float).reshape(-1,2)

            print(f"{stimulus:20s} : T = {hf[fnaming(stimulus)].attrs['T']/1e3:.3f} seconds")
            # ax[idx].plot(data_out_filtered[:,0]/1000., data_out_filtered[:,1], '|')
            # ax[idx].set_title(stimulus)
            # if idx%2 == 0:
            #     ax[idx].set_ylabel('Neuronal Indices')

        # process the grouped data
        # ----
        counter = len(data_pickle)
        for key in data_group.keys():
            if force_regen or not os.path.isfile(out_dir / f"{fnaming(key):s}_spike_train.dat"):
                data_group[key] = data_group[key][np.argsort(data_group[key][:,0], axis=0),:]
                data_group[key][:,0] -= data_group[key][0,0]
                # change the width of big gap in raster
                if gap_width > 0:
                    spk_diff = np.diff(data_group[key][:,0])
                    raster_gaps, = np.nonzero(spk_diff > 100)
                    for gap in raster_gaps:
                        data_group[key][gap+1:,0] -= data_group[key][gap+1,0] - data_group[key][gap,0] - gap_width
                data_out_filtered = force_refractory(data_group[key], t_ref)

                T = np.ceil(data_out_filtered[-1,0])
                # caluculate mean firing rates after downsampling with forced refractory
                # stimulus_present_time = np.sum([data_pickle['stimulus_present_time'][stimulus] for stimulus in stimulus_group[key]])
                rate = np.array([
                    np.sum(data_out_filtered[:,1]==i)*1000.0/T
                    # np.sum(data_out_filtered[:,1]==i)*1.0/stimulus_present_time
                    for i in range(units.shape[0])
                ])
                if fnaming(key) in hf:
                    hds = hf[fnaming(key)]
                    hds[:] = rate
                else:
                    hds = hf.create_dataset(fnaming(key), data=rate)
                hds.attrs['stimulus']=key
                hds.attrs['t_ref']=t_ref
                hds.attrs['gap_width']=gap_width
                hds.attrs['T']=T

                save2bin(
                    out_dir / f"{fnaming(key):s}_spike_train.dat",
                    data_out_filtered
                    )
            else:
                data_out_filtered = np.fromfile(
                    out_dir / f"{fnaming(key):s}_spike_train.dat",
                    dtype=float).reshape(-1,2)
            print(f"{key:20s} : T = {hf[fnaming(key)].attrs['T']/1e3:.3f} seconds")
            # ax[counter].plot(data_out_filtered[:,0]/1000., data_out_filtered[:,1], '|')
            # ax[counter].set_title(key)
            # if key == 'all':
            #     ax[counter].set_xlim(0)
            # if counter%2 == 0:
            #     ax[counter].set_ylabel('Neuronal Indices')
            counter += 1

        hf.close()
        # fig.savefig(fig_path/f"{fnaming('raster'):s}.png", dpi=200)
#%%