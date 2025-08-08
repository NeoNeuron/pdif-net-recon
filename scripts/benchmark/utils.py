import yaml
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path, PosixPath
root_path = Path(__file__).parents[2] 
from subprocess import call, run

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

def run_shell_command(command: str):
    result = run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with error: {result.stderr}")
    return result.stdout

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

def get_vfname(fname: str, sfx:str=''):
    if fname.startswith('Gaussian'):
        fname = fname.replace('th=0.020','') + '_voltage'
    elif fname.startswith('RNN'):
        fname = fname.replace('th=0.200','') + '_voltage'
    elif fname.startswith('Lp') or fname.startswith('Lcon') or fname.startswith('Rcon'):
        fname = fname + '_x'
    else:
        fname = fname + '_voltage'
    return fname + sfx + '.npy'

def get_spk_fname(fname: str, sfx:str='', th: float=None, ref: float=None):
    if len(sfx) == 0:
        return fname
    else:
        if fname.startswith('Gaussian'):
            fname = fname.replace('th=0.020','')
        elif fname.startswith('RNN'):
            fname = fname.replace('th=0.200','')
        fname += sfx + f'_th={th:.2f}ref={ref:.2f}'
        return fname

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