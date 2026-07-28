"""Run the balanced E-I Hodgkin-Huxley network (simHH).

Production driver for the parameter set found by HH/find_balanced_params.py and
decorrelated by HH/reduce_correlation.py.  Self-contained on purpose: it does
not import either of those search modules, so the search code can keep changing
without affecting runs made for analysis.

Network
-------
320 excitatory + 80 inhibitory conductance-based HH neurons.  Fixed in-degree:
every neuron receives exactly K=40 excitatory and K=40 inhibitory recurrent
inputs, no self-connections, from one pinned random graph (seed 0).  Each
neuron also gets its own independent Poisson drive at rate Nu.

Parameters and what they buy (all validated at T_Max = 1e5 ms)
--------------------------------------------------------------
    Jee = Jie =  0.3553      fE = fI = 0.2333       Nu = 0.9 kHz
    Jei = Jii = -8.3810      K = 40, T_step = 0.05

    rate            11.26 Hz (E) / 10.85 Hz (I), no silent neurons
    ISI CV          0.800                   irregular single-neuron firing
    pairwise corr   0.0055 (20 ms bins)     asynchronous
    g_E / g_I       0.348 / 2.040 mS/cm^2   high-conductance (leak G_L = 0.3)
    v_eff          -66.8 mV                 ~12 mV below spike initiation, so
                                            firing is fluctuation-driven
    gamma_ratio     2.37                    no population rhythm (was 8.5
                                            before decorrelation)

Note on the coupling values: E and I cells are identical in simHH (V_th, G_Na,
G_K, T_ref are compile-time constants), and the balance targets are symmetric,
so Jie=Jee, Jii=Jei and fI=fE are forced -- these are not four and two free
numbers but two and one.  Jei/Jee ~ 23.6 looks lopsided but is expected: at
v ~ -67 mV the inhibitory driving force (v-V_I ~ 13 mV) is far smaller than the
excitatory one (V_E-v ~ 67 mV), and only 10% of the excitation is recurrent.

Usage
-----
    python HH/run_balanced_EINet.py --t-max 1e5                  # spikes only
    python HH/run_balanced_EINet.py --t-max 1e4 --record-v --record-currents
    python HH/run_balanced_EINet.py --check                      # verify state
    python HH/run_balanced_EINet.py --t-max 1e5 --seed 3 --out-dir data/EINet/trial3

Outputs land in --out-dir as simHH's usual
`HHp=0.25s=...f=0.233u=0.900_spike_train.dat` etc.; the connectivity actually
used is written alongside as `conn.npy` so a run is always self-describing.
"""
# %%
import argparse
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BIN = REPO_ROOT / "bin/simHH"
DEFAULT_OUT = REPO_ROOT / "data/EINet/balanced"

# ----------------------------------------------------------------- parameters
NE, NI = 320, 80
N = NE + NI
K = 40                 # in-degree from EACH population
CONN_SEED = 0          # pins the connectivity graph
T_STEP = 0.05          # ms; 0.2 is too coarse at g_total ~ 2.7 mS/cm^2

JEE = JIE = 0.3553     # E->E, E->I    (magnitudes; sign comes from cell type)
JEI = JII = 8.3810     # I->E, I->I
FE = FI = 0.2333333333333333   # feedforward Poisson strength
NU = 0.9               # feedforward Poisson rate, kHz

# expected values at these parameters, used by --check
EXPECTED = dict(rate=11.2, isi_cv=0.80, pair_corr=0.0055, g_E=0.348,
                g_ratio=5.86, v_eff=-66.8)


def build_connectivity(seed=CONN_SEED, Jee=JEE, Jie=JIE, Jei=JEI, Jii=JII):
    """(N, N) magnitude-only weight matrix, row = presynaptic, col = post.

    simHH (--full_mode 2) takes each nonzero entry as the coupling strength
    directly and infers E/I identity from the presynaptic index, so every
    stored value is positive.  Weights are the J's divided by sqrt(K).
    """
    rng = np.random.default_rng(seed)
    mat = np.zeros((N, N))
    for post in range(N):
        e_pool = np.delete(np.arange(NE), post) if post < NE else np.arange(NE)
        i_pool = np.arange(NE, N)
        if post >= NE:
            i_pool = np.delete(i_pool, post - NE)
        e_ids = rng.choice(e_pool, K, replace=False)
        i_ids = rng.choice(i_pool, K, replace=False)
        mat[e_ids, post] = (Jee if post < NE else Jie) / np.sqrt(K)
        mat[i_ids, post] = (Jei if post < NE else Jii) / np.sqrt(K)
    assert np.all(np.diag(mat) == 0), "no autapses"
    assert set((mat[:NE] != 0).sum(0)) == {K}, "E in-degree must be exactly K"
    assert set((mat[NE:] != 0).sum(0)) == {K}, "I in-degree must be exactly K"
    return mat


def run(t_max=1e5, out_dir=DEFAULT_OUT, seed=CONN_SEED, poisson_seed=None,
        record_spk=True, record_v=False, record_currents=False,
        record_window=None, state_path=None, verbose=False):
    """Run simHH once. Returns (out_dir, stdout)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = out_dir / "conn.npy"
    np.save(conn, build_connectivity(seed))

    if record_window is None:                      # default: last 2 s
        record_window = (max(t_max - 2000.0, 0.0), t_max)

    cmd = [
        str(BIN),
        "--T_Max", f"{t_max:.10g}",
        "--T_step", str(T_STEP),
        "--NE", str(NE), "--NI", str(NI),
        "--full_mode", "2",
        "--conn_matrix_file", str(conn),
        "--overwrite_conn", "1",
        "--fE", f"{FE:.6f}", "--fI", f"{FI:.6f}",
        "--Nu", f"{NU:.6f}",
        "--record_path", str(out_dir) + "/",
        "--record_spk", "1" if record_spk else "0",
        "--record_v", "1" if record_v else "0",
        "--record_IE", "1" if record_currents else "0",
        "--record_II", "1" if record_currents else "0",
    ]
    if record_v:
        cmd += ["--record_vlim", f"{record_window[0]:.10g} {record_window[1]:.10g}"]
    if record_currents:
        cmd += ["--record_Ilim", f"{record_window[0]:.10g} {record_window[1]:.10g}"]
    if poisson_seed is not None:
        # simHH's --seed takes two values: connectivity seed and Poisson seed.
        # The connectivity comes from the .npy, so only the second matters here.
        cmd += ["--seed", f"{seed} {poisson_seed}"]
    if state_path is not None:
        # NB: simHH prepends --record_path to --state_path, so this must be
        # relative to out_dir, not absolute (Read_parameters.h).
        cmd += ["--state_path", str(state_path)]

    res = subprocess.run(cmd, capture_output=True, universal_newlines=True)
    if res.returncode != 0:
        raise RuntimeError(f"simHH failed:\n{res.stderr}\n{res.stdout[-2000:]}")
    if verbose:
        print(res.stdout)
    return out_dir, res.stdout


# ------------------------------------------------------------------- checking
def load_spikes(out_dir, t_range=None):
    """(n_spikes, 2) array of [time_ms, neuron_id]."""
    path = next(Path(out_dir).glob("*_spike_train.dat"))
    spk = np.fromfile(path, dtype=float)
    spk = spk[: (spk.size // 2) * 2].reshape(-1, 2)
    if t_range is not None and spk.size:
        spk = spk[(spk[:, 0] >= t_range[0]) & (spk[:, 0] < t_range[1])]
    return spk


def check(t_max=2e4, out_dir=None, burn_in=500.0):
    """Short run + report of the properties that define the balanced state."""
    out_dir = Path(out_dir or DEFAULT_OUT) / "check"
    run(t_max=t_max, out_dir=out_dir, record_spk=True)
    spk = load_spikes(out_dir, (burn_in, t_max))
    dur = (t_max - burn_in) * 1e-3
    rates = np.bincount(spk[:, 1].astype(int), minlength=N) / dur

    cvs = []
    order = np.argsort(spk[:, 1], kind="stable")
    ids, ts = spk[order, 1].astype(int), spk[order, 0]
    edges = np.searchsorted(ids, np.arange(N + 1))
    for i in range(N):
        tt = np.sort(ts[edges[i]:edges[i + 1]])
        if tt.size >= 6:
            isi = np.diff(tt)
            cvs.append(isi.std() / isi.mean())

    bw = 20.0
    e = np.arange(burn_in, t_max + bw, bw)
    sel = np.random.default_rng(0).choice(N, 100, replace=False)
    M = np.stack([np.histogram(spk[spk[:, 1] == i, 0], bins=e)[0] for i in sel])
    M = M.astype(float)
    M = M[M.std(axis=1) > 0]
    C = np.corrcoef(M)
    corr = float(np.nanmean(C[np.triu_indices_from(C, 1)]))

    pr = np.histogram(spk[:, 0], bins=np.arange(burn_in, t_max + 1, 1.0))[0].astype(float)
    x = pr - pr.mean()
    f = np.fft.rfftfreq(len(x), d=1e-3)
    P = np.convolve(np.abs(np.fft.rfft(x)) ** 2, np.ones(201) / 201, "same")
    band = (f > 2) & (f < 300)
    gamma = float(P[band].max() / np.median(P[band]))

    got = dict(rate_E=float(rates[:NE].mean()), rate_I=float(rates[NE:].mean()),
               isi_cv=float(np.median(cvs)), pair_corr=corr,
               frac_silent=float((rates < 0.5).mean()), gamma_ratio=gamma)
    print(f"--- balanced-state check ({t_max:.0f} ms) ---")
    print(f"  rate E / I     {got['rate_E']:.2f} / {got['rate_I']:.2f} Hz"
          f"      (expected ~{EXPECTED['rate']:.1f})")
    print(f"  ISI CV         {got['isi_cv']:.3f}"
          f"                (expected ~{EXPECTED['isi_cv']:.2f}, irregular)")
    print(f"  pair corr      {got['pair_corr']:.4f}"
          f"               (expected ~{EXPECTED['pair_corr']:.4f}, < 0.1)")
    print(f"  gamma ratio    {got['gamma_ratio']:.2f}"
          f"                 (expected ~2.4; >5 means the raster will stripe)")
    print(f"  silent frac    {got['frac_silent']:.3f}                (expected 0)")
    ok = (5 < got["rate_E"] < 25 and 0.7 <= got["isi_cv"] <= 1.4
          and got["pair_corr"] < 0.1 and got["frac_silent"] < 0.05
          and got["gamma_ratio"] < 5)
    print("  => " + ("OK, network is in the balanced state" if ok else
                     "MISMATCH -- check bin/simHH and the parameters above"))
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t-max", type=float, default=1e5, help="ms (default 1e5)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=CONN_SEED,
                    help="connectivity seed (default 0, the pinned graph)")
    ap.add_argument("--poisson-seed", type=int, default=None,
                    help="vary the drive while keeping the same network")
    ap.add_argument("--record-v", action="store_true")
    ap.add_argument("--record-currents", action="store_true",
                    help="per-neuron I_E and I_I")
    ap.add_argument("--record-window", type=float, nargs=2, default=None,
                    metavar=("START", "END"),
                    help="ms window for v/current recording (default: last 2000)")
    ap.add_argument("--check", action="store_true",
                    help="short run verifying the balanced state, then exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.check:
        check(out_dir=args.out_dir)
    else:
        out, _ = run(t_max=args.t_max, out_dir=args.out_dir, seed=args.seed,
                     poisson_seed=args.poisson_seed, record_v=args.record_v,
                     record_currents=args.record_currents,
                     record_window=(tuple(args.record_window)
                                    if args.record_window else None),
                     verbose=args.verbose)
        print(f"done -> {out}")
        for p in sorted(out.glob("*")):
            print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
