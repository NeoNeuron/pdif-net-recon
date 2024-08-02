# %%
import numpy as np
from causal4.utils import *
from numba import njit
from pathlib import Path
from typing import Union

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

def sim_Gaussian(data_path, N:int=2,
        dt:float=1,p:float=0.1,s:float=0.1,tau:float=20.,
        threshold:float=0.2, ref:float=2, T:float=1e7,
        record_v:bool=True, record_vlim:Union[None, tuple]=None, force_regen:bool=False,
        seed:int=None, **kwargs,
    )->tuple:
    """Simulate Gaussian regression process.

    Args:
        data_path (Path): path to save the data.
        N (int, optional): number of nodes. Defaults to 2.
        dt (float, optional): time step. Defaults to 1.
        p (float, optional): network sparsity (Erdős-Rényi). Defaults to 0.1.
        s (float, optional): coupling strength. Defaults to 0.1.
        tau (float, optional): time constant of the system. Defaults to 20..
        threshold (float, optional): threshold for pulse-output generation. Defaults to 0.2.
        ref (float, optional): time of refractory period. Defaults to 2.
        T (float, optional): whole stimulation period. Defaults to 1e7.
        record_v (bool, optional): True for recording voltage. Defaults to True.
        record_vlim (tuple, optional): Time range ot record voltage.
            Defaults to None for recording the whole simulation period.
        force_regen (bool, optional): force regenerate pulse-output data 
                even if corresponding data files already exit. Defaults to False.
        seed (int, optional): random seed. Defaults to None.

    Returns:
        tuple: _description_
    """

    data_path = Path(data_path)
    data_path.mkdir(exist_ok=True, parents=True)
    vol_fname = data_path / f"Gaussianp={p:.2f}s={s:.3f}tau={tau:.0f}ref={int(ref):d}l={T:.0e}_voltage.dat"
    spk_fname = data_path / f"Gaussianp={p:.2f}s={s:.3f}tau={tau:.0f}ref={int(ref):d}th={threshold:.3f}l={T:.0e}_spike_train.dat"
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
                    save2bin(data_path/f"connect_matrix-p={p:.3f}.dat", W.T)
                write_mode = 'wb' if start == 0 else 'ab'
                if record_v:
                    if record_vlim is None or \
                        (record_vlim[0] <= start and record_vlim[1] >= start+period):
                        save2bin(vol_fname, vol_out, write_mode)
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
            # spike_data[:,0] /= 1000.
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
    import argparse
    def parse_floats(value):
        if value is not None:
            try:
                return tuple(float(x) for x in value.split(' '))
            except ValueError:
                raise argparse.ArgumentTypeError("Value must be a whitespace-separated list of floats")

    # Create the parser
    parser = argparse.ArgumentParser(description='Simulate random network with linear Gaussian dynamics.')

    # Add arguments
    parser.add_argument('--N', type=int, default=100, help='Number of neurons. Default is 100.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--T_Max', type=float, default=1e5, help='Time length of simulation.')
    parser.add_argument('--T_step', type=float, default=1.0, help='Time step in ms. Default is 1 ms.')
    parser.add_argument('--P_c', type=float, default=0.25, help='Probability of connection between neurons. Default is 0.25.')
    parser.add_argument('--S', type=float, default=0.03, help='Standard deviation of Gaussian distribution. Default is 0.03.')
    parser.add_argument('--tau', type=float, default=20.0, help='Membrane constant time in ms. Default is 20 ms.')
    parser.add_argument('--threshold', type=float, default=0.02, help='Spike threshold. Default is 0.02.')
    parser.add_argument('--ref', type=int, default=10, help='Refractory period in ms. Default is 10 ms.')
    parser.add_argument('--record_path', type=str, default='data/', help='Path to save the data. Default is data/.')
    parser.add_argument('--record_v', type=int, default=0, help='Record voltage data. Default 0.')
    parser.add_argument('--record_vlim', type=parse_floats, default=None, help='Time range to record voltage data. Default None.')

    # Parse the arguments
    args = parser.parse_args()

    # Update pm_dym dictionary with parsed arguments
    pm_dym = dict(
        data_path=args.record_path,
        N=args.N,
        seed=args.seed,
        T=args.T_Max,
        dt=args.T_step,
        p=args.P_c,
        s=args.S,
        tau=args.tau,
        threshold=args.threshold,
        ref=args.ref,
        record_v=bool(args.record_v),
        record_vlim=args.record_vlim,
    ) 
    spk_fname, vol_fname = sim_Gaussian(**pm_dym, force_regen=False)
    print(spk_fname, vol_fname)