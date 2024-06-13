# %%
import numpy as np
from causal4.utils import *
from numba import njit
from pathlib import Path
data_path = Path(__file__).parents[0]/'Gaussian/data/EE/'

@njit
def evolve_gauss(noise:np.ndarray, W:np.ndarray, 
        x_init:np.ndarray=None, tau:float=20., dt:float=1., 
    ):

    """Evolve Gaussian regression model.

    Args:
        noise (np.ndarray): (num_time_points, num_units), Gausian white process that drives the dynamics.
        W (np.ndarray): Connectivity matrix.
        x_init (np.ndarray, optional): Initial state values. Default to None.
        tau (float, optional): Time constant of dynamics. Defaults to 20..
        dt (float, optional): Discretized time step. Defaults to 1..

    Returns:
        np.ndarray: (number_time_points, num_units), Trajectory of Gaussian process.
    """
    num_len, num_units = noise.shape
    signal = np.zeros((num_len+1, num_units))
    if x_init is not None:
        signal[0,:] = x_init
    eta = 1-dt/tau
    sigma = dt/tau
    s_noise = 0.1
    for i in range(num_len):
        signal[i+1,:] = eta*signal[i,:] + sigma*(W@signal[i,:] + s_noise*noise[i,:])

    return signal[1:]

def sim_Gaussian(N:int=2,
        dt:float=1,p:float=0.1,s:float=0.1,tau:float=20.,
        threshold:float=0.2, ref:float=2, T:float=1e7,
        record_v:bool=True, force_regen:bool=False,
        seed:int=None, **kwargs,
    )->tuple:
    """Simulate Gaussian regression process.

    Args:
        N (int, optional): number of nodes. Defaults to 2.
        dt (float, optional): time step. Defaults to 1.
        p (float, optional): network sparsity (Erdős–Rényi). Defaults to 0.1.
        s (float, optional): coupling strength. Defaults to 0.1.
        tau (float, optional): time constant of the system. Defaults to 20..
        threshold (float, optional): threshold for pulse-output generation. Defaults to 0.2.
        ref (float, optional): time of refractory period. Defaults to 2.
        T (float, optional): whole stimulation period. Defaults to 1e7.
        record_v (bool, optional): True for recording voltage. Defaults to True.
        force_regen (bool, optional): force regenerate pulse-output data 
                even if corresponding data files already exit. Defaults to False.
        seed (int, optional): random seed. Defaults to None.

    Returns:
        tuple: _description_
    """

    PATH = data_path / f"N={N:d}"
    PATH.mkdir(exist_ok=True, parents=True)
    vol_fname = PATH / f"Gaussianp={p:.2f}s={s:.3f}tau={tau:.0f}ref={int(ref):d}l={T:.0e}_voltage.dat"
    spk_fname = PATH / f"Gaussianp={p:.2f}s={s:.3f}tau={tau:.0f}ref={int(ref):d}th={threshold:.3f}l={T:.0e}_spike_train.dat"
    if ref > 0:
        ref_L = int(np.ceil(ref/dt))
        if ref % dt == 0:
            ref_L += 1
    # Generate Gaussian data
    Tn = np.ceil(T/dt).astype(int)
    if seed is not None:
        np.random.seed(seed)
    if force_regen or not spk_fname.exists():
        buff_size = 1e8  # number of measurements in buffer, requires arround 0.745 GiB RAM
        buff_lines = int(buff_size/N)
        if Tn < buff_lines:
            tranges = [[0, Tn],]
        else:
            period_time = buff_lines*dt
            if Tn%buff_lines == 0:
                n_iters = int(Tn // buff_lines)
            else:
                n_iters = int(Tn // buff_lines) + 1
            tranges = np.vstack((np.arange(n_iters)*period_time, np.ones(n_iters)*buff_lines)).T
            tranges[-1,1] = Tn - (n_iters-1)*buff_lines
        print(tranges)

        if not vol_fname.exists():
            W = (np.random.rand(N,N)<p).astype(float)
            W[np.eye(N,dtype=bool)] = 0.0
            if N == 2:
                W = np.array([[0,1],[0,0]]).astype(float)
            x0 = np.zeros(N)
            gen_voltage_flag = True
        else:
            vol_file = open(vol_fname, 'rb')
            gen_voltage_flag = False

        offset = 0
        for start, period in tranges:
            if gen_voltage_flag:
                noise = np.random.randn(int(period), N)
                x = evolve_gauss(noise, W*s, x0, tau, dt)
                x0 = x[-1, :].copy()
                time = np.arange(int(period), dtype=float)*dt + start
                vol_out = np.vstack((time, x.T)).T
                if start == 0:
                    if record_v:
                        save2bin(vol_fname, vol_out, 'wb')
                    save2bin(PATH/f"connect_matrix-p={p:.3f}.dat", W.T)
                else:
                    if record_v:
                        save2bin(vol_fname, vol_out, 'ab')
            else:
                vol_out = np.fromfile(vol_fname, dtype=float, count=int(period*(N+1)), offset=offset).reshape(-1,N+1)
                offset += int(period*(N+1)*8)
                time = vol_out[:,0]
                x = vol_out[:,1:]

            spike_mask = (np.diff(x>threshold, axis=0) == 1)
            if ref > 0:
                spike_mask = apply_refractory(ref_L, spike_mask)
            spike_time = (np.tile(time[:-1], (N,1)).T)[spike_mask]
            spike_idx = np.tile(np.arange(N), (time.shape[0]-1, 1))[spike_mask]
            spike_data = np.vstack((spike_time, spike_idx)).T
            spike_data[:,0] /= 1000.
            if start == 0:
                save2bin(spk_fname, spike_data, 'wb')
            else:
                save2bin(spk_fname, spike_data, 'ab')
        if not gen_voltage_flag:
            vol_file.close()
    return spk_fname, vol_fname
# %%
if __name__ == '__main__':
    # %%
    import matplotlib.pyplot as plt
    from causal4.Causality import CausalityEstimator
    pm_dym = dict(
        dtype = 'Gaussian',
        N     = 100,
        dt    = 1,
        p     = 0.25,
        s     = 0.03,
        tau   = 20., # membrane constant time, ms
        threshold = 0.02,
        ref   = 10, # refractory period, ms
    )
    N = pm_dym['N']
    T = 1e8
    # %%
    spk_fname, vol_fname = sim_Gaussian(T=T, **pm_dym, force_regen=False, seed=100)
    # %%
    pm_causal = dict(
        N           = pm_dym['N'],
        T           = T,  # ms,
        DT          = 1e5,  # ms,
        dt          = 1,   #  ms,
        order       = (1,1), # x, y,
        delay       = 16,   # ms,
        spk_fname   = spk_fname.stem.rstrip('_spike_train'),
        conn_file   = f"connect_matrix-p={pm_dym['p']:.3f}.dat",
        path        = data_path/f"N={pm_dym['N']:d}/",
    )

    estimator = CausalityEstimator(**pm_causal, n_thread=60)
    data = estimator.fetch_data(new_run=True)
    # %%
    from causal4.utils import match_features, reconstruction_analysis
    data = match_features(data, N=pm_causal['N'], conn_file=pm_causal['path']/pm_causal['conn_file'])
    #%%
    from seaborn import histplot
    histplot(data, x='log-TE', hue='connection')