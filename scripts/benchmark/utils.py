import yaml
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path, PosixPath
root_path = Path(__file__).parents[2] 
from subprocess import call

def arg_wrapper(pm_dict:dict):
    wrapped_list = []
    for key, val in pm_dict.items():
        if isinstance(val, tuple):
            wrapped_list.append(f'--{key}='+' '.join([str(i) for i in val]))
            f'--{key}={val[0]}:{val[1]}'
        elif isinstance(val, PosixPath):
            wrapped_list.append(f'--{key}={str(val)}/')
        else:
            wrapped_list.append(f'--{key}={val}')
    return wrapped_list

def run_simulation(pm_simulation:dict):
    _pm = pm_simulation.copy()
    simulator = _pm.pop('simulator')
    simulator = str(root_path / simulator)
    if '.py' in simulator:
        call(['python', simulator] + arg_wrapper(_pm))
    else:
        call([simulator] + arg_wrapper(_pm))
    # dump yaml config to pm_simulation['record_path']
    with open(_pm['record_path'] + 'config.yml', 'w') as yamlfile:
        yaml.dump(pm_simulation, yamlfile)

def get_vol_fname(fname:str, key:str):
    if key in ['Lorenz', 'Lcon']:
        return fname + '_x'
    elif key == 'Gaussian':
        return fname.replace('th=0.020', '')+'_voltage'
    else:
        return fname + '_voltage'

def joyplot_voltage(data, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    for i in range(data.shape[1]-1):
        time = data[:,0]
        vol = data[:,i+1]
        vol = (vol - vol.min())/(vol.max()-vol.min())*0.9+i
        ax.plot(time, vol, **kwargs)
    ax.set_yticks(np.arange(data.shape[1])+.5, range(1,data.shape[1]+1))
    return ax