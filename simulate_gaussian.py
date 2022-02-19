# %%
import numpy as np
import os, sys
REPO_PATH = os.path.dirname(__file__)
sys.path.append(REPO_PATH+'/python_code_data/python_code')
from utils import *
from numba import njit
DATA_PATH=REPO_PATH+'/Gaussian/data/'

@njit
def evolve_gauss(noise:np.ndarray, W:np.ndarray, x0:np.ndarray=None, tau:float=20., dt:float=1., rowvar=True):
    """Evolve Gaussian regression model.

    Args:
        noise (np.ndarray): Gausian white process that drives the dynamics.
        W (np.ndarray): Connectivity matrix.
        x0 (np.ndarray, optional): Initial state values. Default to None.
        tau (float, optional): Time constant of dynamics. Defaults to 20..
        dt (float, optional): Discretized time step. Defaults to 1..
        rowvar (bool, optional): True for each row is a variable. Defaults to True.

    Returns:
        np.ndarray: Trajectory of Gaussian process.
    """
    if not rowvar:
        noise = noise.T
    signal = np.zeros(noise.shape)
    if x0 is not None:
        signal[0,:] = x0
    T, _ = signal.shape
    eta = 1-dt/tau
    sigma = dt/tau
    s_noise = 0.1
    for i in range(T-1):
        signal[i+1,:] = eta*signal[i,:] + sigma*(W@signal[i,:] + s_noise*noise[i+1,:])

    if not rowvar:
        noise = noise.T
    return signal

def sim_Gaussian(N:int=2,
    dt:float=1,p:float=0.1,s:float=0.1,tau:float=20.,
    threshold:float=0.2, ref:float=2, T:float=1e7,
    force_regen:bool=False, seed:int=None, **kwargs):

    PATH = DATA_PATH + f"EE/N={N:d}/"
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    fname_prefix = PATH + f"Gaussianp={p:.2f}s={s:.3f}tau={tau:.3f}ref={int(ref):d}th={threshold:.3f}l={T:.0e}"
    spk_fname = fname_prefix + "_spike_train.dat"
    vol_fname = fname_prefix + "_voltage.dat"
    # Generate Gaussian data
    Tn = np.ceil(T/dt).astype(int)
    if seed is not None:
        np.random.seed(seed)
    if force_regen or not os.path.isfile(spk_fname):
        W = (np.random.rand(N,N)<p).astype(float)
        W[np.eye(N,dtype=bool)] = 0.0
        if N == 2:
            W = np.array([[0,1],[0,0]]).astype(float)

        buff_size = 1e8  # requires arround 0.745 GiB RAM
        buff_lines = int(buff_size/N)
        if Tn < buff_lines:
            tranges = [[0, Tn],]
        else:
            period_time = buff_lines*dt
            if Tn%buff_lines == 0:
                n_iters = int(Tn//buff_lines)
            else:
                n_iters = int(Tn//buff_lines) + 1
            tranges = np.vstack((np.arange(n_iters)*period_time, np.ones(n_iters)*buff_lines)).T
            tranges[-1,1] = Tn - (n_iters-1)*buff_lines
        print(tranges)

        x0 = np.zeros(N)
        # noise = np.random.randn(Tn, N)
        # noise_flag = 0
        for start, period in tranges:
            noise = np.random.randn(int(period), N)
            x = evolve_gauss(noise, W*s, x0, tau, dt)
            # x = evolve_gauss(noise[noise_flag:noise_flag+int(period), :], W*s, x0, tau, dt)
            # noise_flag += int(period)
            # print(x.shape)
            x0 = x[-1, :].copy()

            time = np.arange(int(period), dtype=float)*dt + start
            vol_out = np.vstack((time, x.T)).T
            spike_mask = (np.diff(x>threshold, axis=0) == 1)
            spike_time = [time[:-1][spike_mask[:,i]] for i in range(N)]
            spike_time = np.array([
                np.vstack((element, np.ones_like(element)*idx)).T
                for idx, element in enumerate(spike_time)
                ])
            spike_time = np.vstack(spike_time)
            spike_time = spike_time[np.argsort(spike_time[:,0], axis=0),:]
            spike_time[:,0] /= 1000.
            spike_time = force_refractory(spike_time, t_ref=ref)
            if start == 0:
                # save2bin(vol_fname, vol_out, 'wb')
                save2bin(spk_fname, spike_time, 'wb')
                save2bin(PATH+f"connect_matrix-p={p:.3f}.dat", W.T)
            else:
                # save2bin(vol_fname, vol_out, 'ab')
                save2bin(spk_fname, spike_time, 'ab')
    return spk_fname, vol_fname
# %%
if __name__ == '__main__':
    # %%
    import matplotlib.pyplot as plt
    from Causality import run
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
    spk_fname, vol_fname = sim_Gaussian(T=T, **pm_dym, force_regen=True, seed=100)
    # %%
    vol_data = np.fromfile(vol_fname, dtype=float).reshape(-1,pm_dym['N']+1)
    spk_data = np.fromfile(spk_fname, dtype=float).reshape(-1,2)
    fig, ax = plt.subplots(2,1,figsize=(12,5), sharex=True)
    xmax = 10000
    ax[0].plot(vol_data[:xmax, 0], vol_data[:xmax, 90], label='1')
    ax[0].plot(vol_data[:xmax, 0], vol_data[:xmax, 2], label='2')
    ax[0].axhline(pm_dym['threshold'], ls='--', color='r')
    ax[1].plot(spk_data[:, 0], spk_data[:, 1], '|')
    ax[1].set_xlim(xmax-100,xmax)
    print(f"mean firing rate is {spk_data.shape[0]/spk_data[-1,0]/vol_data.shape[1]*1000.:f}")
    plt.savefig('test_gaussian1111.png')
    # %%
    pm_causal = dict(
        Ne          = pm_dym['N'],
        Ni          = 0,
        T           = T,  # ms,
        DT          = 2e4,  # ms,
        auto_T_max  = 0,
        bin         = 1,   #  ms,
        order       = (1,1), # x, y,
        delay       = 16,   # ms,
        fname       = spk_fname.split('/')[-1].replace('_spike_train.dat', ''),
        con_mat     = f"connect_matrix-p={pm_dym['p']:.3f}.dat",
        path_input  = DATA_PATH,
        path_output = DATA_PATH,
    )

    run(False, **pm_causal)