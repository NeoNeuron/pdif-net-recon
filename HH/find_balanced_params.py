"""Find simHH parameters that put an E-I network in a balanced (asynchronous
irregular, high-conductance) state.

Network: NE=320, NI=80, fixed in-degree K=40 from *each* population (every
neuron receives exactly K excitatory and K inhibitory recurrent inputs), no
autapses, one pinned connectivity seed.

Why this replaces the earlier nu/Jei sweeps
-------------------------------------------
simHH is conductance-based (I_E = -(G_f+G_se)(v-0), I_I = -G_si(v+80), with
double-exponential synapses: rise 0.5 ms, decay 3 ms for E and 7 ms for I --
see HH/Def.h and Update_neu_G in HH/Runge_Kutta4.h).  The BrainPy HHNet this
was ported from used FullProjDelta, i.e. *current-based* delta kicks, so its
weights do not carry over.

With the old parameters (Jee=0.1, Jei=-0.4, Fe=0.1, nu=0.05) the time-averaged
conductances are g_f~0.047, g_se~0.003, g_si~0.007 mS/cm^2 against a leak
G_L=0.3: the total synaptic conductance is ~5x *below* leak and recurrent
excitation is ~7% of all excitation.  That is a feedforward-driven network, and
no I->E weight can balance it -- which is exactly what the Jei sweep showed
(the "zero crossing" it found was an oscillatory regime, not balance).

Targets used here
-----------------
Operating point v* ~ -58 mV (below V_th=-50, so firing is fluctuation-driven):

    g_E_total >= G_L = 0.3          -> high-conductance state
    g_I / g_E ~ (0-v*)/(v*+80) ~ 2.6 -> E and I currents comparable
    g_se / g_E ~ 0.5                 -> recurrent excitation ~ feedforward

Reaching g_E >= G_L at K=40 forces large unitary PSPs (3-7 mV); that trade-off
is deliberate (small PSPs, high conductance and strong recurrence are mutually
incompatible at this K).

Parameterisation
----------------
E and I cells are identical in simHH (V_th, G_Na, G_K, T_ref are #defines), and
the balance targets are symmetric, so the parameters collapse to

    Jie = Jee = J_E,    Jii = Jei = J_I,    fI = fE = f,    r_E = r_I

leaving a 1-D fixed point in the emergent rate.  Time-averaged conductance for
a synapse of strength w driven at rate R (kHz) is

    g = w * R * sigma_r * sigma_d          (exact: int G dt = w*sigma_r*sigma_d)

so 1.5*w*R for E synapses (0.5*3) and 3.5*w*R for I synapses (0.5*7).  Hence

    g_f  = f * Nu * 1.5                       (rate-independent: Nu is imposed)
    g_se = (J_E/sqrt(K)) * K * r * 1.5
    g_si = (J_I/sqrt(K)) * K * r * 3.5
    =>  J_I/J_E = (g_si/g_se) * (1.5/3.5), independent of r.

With the targets below (g_si=2.095, g_se=0.035) that predicts J_I/J_E = 25.7;
the validated parameters give 8.381/0.3553 = 23.6, closing to 8%.

Usage
-----
    python HH/find_balanced_params.py sweep      # parallel grid (batch)
    python HH/find_balanced_params.py fixedpoint # sequential iteration
    python HH/find_balanced_params.py validate   # long run + full metrics
"""
# %%
import argparse
import json
import re
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BIN = REPO_ROOT / "bin/simHH"
OUT_ROOT = REPO_ROOT / "data/EINet/balanced_search"
CONN_PATH = OUT_ROOT / "mat_K40_seed0.npy"

# ---------------------------------------------------------------- network
NE, NI = 320, 80
N = NE + NI
K = 40
CONN_SEED = 0

# ------------------------------------------------- model constants (Def.h)
G_L = 0.3        # mS/cm^2
E_L = -54.387    # mV
V_TH = -50.0     # mV, spike-detection threshold used by simHH
V_INIT = -55.0   # mV, approximate voltage at which HH actually initiates a
                 # spike under slow depolarisation (below simHH's V_TH, which
                 # is only a crossing detector placed on the upstroke)
V_E, V_I = 0.0, -80.0
SIGMA_R_E, SIGMA_D_E = 0.5, 3.0
SIGMA_R_I, SIGMA_D_I = 0.5, 7.0
INT_E = SIGMA_R_E * SIGMA_D_E   # 1.5, = int g dt per unit weight (E synapse)
INT_I = SIGMA_R_I * SIGMA_D_I   # 3.5, same for I synapses

# ------------------------------------------------------------ balance targets
#
# The inhibition target is set by where the membrane's *effective reversal
# potential* sits, not by cancelling the two synaptic currents against each
# other.  Including the leak (which is depolarising at these voltages, since
# E_L = -54.4):
#
#     v_eff = (g_E*V_E + g_I*V_I + G_L*E_L) / (g_E + g_I + G_L)
#
# Fluctuation-driven (irregular) firing requires v_eff to sit a couple of
# sigma_V below the spike-initiation voltage, so that threshold crossings are
# driven by fluctuations rather than by the mean.  With sigma_V ~ 5 mV and
# V_INIT ~ -55, that means v_eff ~ -65.
#
# An earlier version of this file targeted g_I/g_E = (V_E-v*)/(v*-V_I) ~ 2.6,
# which omits the leak and amounts to demanding zero *synaptic* net current.
# That places v_eff at ~ -57, i.e. right at spike initiation -- a mean-driven
# neuron.  Empirically it produced ISI CV ~ 0.47-0.56 over 86 grid points,
# and CV only cleared 0.7 once g_ratio exceeded ~4.99, which is precisely the
# ratio at which v_eff reaches -65.  Hence the reformulation below.
G_E_TARGET = 0.35                           # >= G_L
V_EFF_TARGET = -67.0                        # mV, ~3*sigma_V below V_INIT.
                                            # Located empirically: a 15-point
                                            # grid over v_eff x Nu put the ISI
                                            # CV = 0.7 crossing at v_eff ~ -67
                                            # (CV 0.671 at -65.0, 0.696 at
                                            # -65.9, 0.717 at -67.1), with CV
                                            # flat in Nu -- i.e. the driving
                                            # fluctuations are recurrent, not
                                            # external shot noise.
G_I_TARGET = ((G_E_TARGET * V_E + G_L * E_L - V_EFF_TARGET * (G_E_TARGET + G_L))
              / (V_EFF_TARGET - V_I))       # ~1.73
G_RATIO = G_I_TARGET / G_E_TARGET           # ~4.94
G_RATIO_TOL = 0.30                          # accept g_ratio within +-30%
REC_FRAC = 0.1                              # recurrent share of g_E.
# Lowered from 0.5 to decorrelate the spike trains.  At REC_FRAC=0.5 the
# network showed a 76 Hz ING population rhythm (spectral peak 8.5x the median
# background) visible as vertical striping in the raster, with pairwise
# correlation 0.049 at 5 ms bins.  Correlation comes from *shared* input, so it
# is set by how much of the excitation is recurrent; sweeping REC_FRAC at a
# fixed balance point gave a monotonic 6x reduction (0.049 -> 0.008) with rate,
# ISI CV, g_E, g_ratio and v_eff all unchanged.  Weight heterogeneity made the
# rhythm *worse*, drive heterogeneity did nothing, and cutting J_ii only helped
# by knocking the network off its operating point -- see HH/reduce_correlation.py.
# Note this costs little in overall recurrence: the dominant recurrent
# conductance is inhibitory (g_si ~ 2.0, unchanged), so the recurrent share of
# *total* synaptic conductance only moves 0.93 -> 0.87.  What it does cost is
# the E->E unitary EPSP, 2.2 mV -> 0.5 mV.
G_SE_TARGET = G_E_TARGET * REC_FRAC
G_F_TARGET = G_E_TARGET * (1 - REC_FRAC)


def v_eff(g_E, g_I):
    """Effective reversal potential of the full membrane (leak included)."""
    return (g_E * V_E + g_I * V_I + G_L * E_L) / (g_E + g_I + G_L)

NU = 0.5        # kHz. The one free scale (see module docstring / Q3-Q4):
                # 4x below the old 2.0, but high enough that a single external
                # event stays ~3 mV rather than ~7 mV.
T_STEP = 0.05   # ms. Below the 0.2 default: total conductance reaches
                # ~1.6 mS/cm^2, so the membrane time constant drops to ~0.6 ms.

# ------------------------------------------------------------ search settings
SWEEP_T_MAX = 2e3
FIXPOINT_T_MAX = 2e3
BURN_IN = 500.0        # ms discarded (from rest g_I=0, so there is a transient)
MEASURE_WINDOW = 2000.0  # ms of v/I_E/I_I recorded per run. Spike-based
                         # metrics (rate, ISI CV, pair_corr) always use the
                         # full spike record, which is what makes them
                         # trustworthy on the long validation run.
SPIKE_MASK_V = -40.0   # samples with v above this are inside a spike
POST_SPIKE_MS = 10.0   # ms after a spike excluded from the v_clean average
RATE_MIN, RATE_MAX = 1.0, 60.0   # sanity filter on emergent rates

RATE_RE = re.compile(r"mean rate \(Hz\) = ([\d.]+) \(E : ([\d.]+), I : ([\d.]+)\)")


# ============================================================ connectivity
def generate_connectivity(J_E, J_I, seed=CONN_SEED, path=None):
    """(N, N) magnitude-only matrix, row=pre, col=post, for --full_mode=2.

    Every neuron gets exactly K excitatory and K inhibitory presynaptic
    partners, self-connections excluded.  simHH takes E/I identity from the
    presynaptic index, so all stored weights are positive.
    """
    rng = np.random.default_rng(seed)
    mat = np.zeros((N, N))
    cs_e, cs_i = J_E / np.sqrt(K), J_I / np.sqrt(K)
    for post in range(N):
        e_pool = np.delete(np.arange(NE), post) if post < NE else np.arange(NE)
        i_pool = np.arange(NE, N)
        if post >= NE:
            i_pool = np.delete(i_pool, post - NE)
        mat[rng.choice(e_pool, K, replace=False), post] = cs_e
        mat[rng.choice(i_pool, K, replace=False), post] = cs_i
    assert np.all(np.diag(mat) == 0)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mat)
    return mat


def f_from_Nu(Nu=NU):
    """Feedforward strength giving g_f = G_F_TARGET. Exact, rate-independent."""
    return G_F_TARGET / (Nu * INT_E)


def J_from_rate(rate_hz, Nu=NU):
    """Seed couplings from an assumed emergent rate (kHz = Hz/1000)."""
    r = rate_hz * 1e-3
    J_E = G_SE_TARGET * np.sqrt(K) / (K * r * INT_E)
    J_I = G_I_TARGET * np.sqrt(K) / (K * r * INT_I)
    return J_E, J_I


# ============================================================ running simHH
def run_sim(J_E, J_I, f, Nu, t_max, out_dir, record_state=True, seed=CONN_SEED,
            window=None):
    """Run simHH once into its own directory and return that directory.

    Each run needs a private record_path: with record_spk=1 simHH always
    rewrites connect_matrix*.npy into it, so parallel workers sharing one
    directory would race.

    `window` (ms) overrides MEASURE_WINDOW for the v/I_E/I_I recording only;
    spike-based metrics always use the whole run.  Batches that only need a
    short example trace should shrink it -- the state files are ~190 KB per
    simulated ms at T_step=0.05, so they dominate disk use in a parallel sweep.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn_path = out_dir / "conn.npy"
    generate_connectivity(J_E, J_I, seed=seed, path=conn_path)

    win_start = max(t_max - (MEASURE_WINDOW if window is None else window), BURN_IN)
    cmd = [
        str(BIN),
        "--T_Max", f"{t_max:.6g}",
        "--T_step", str(T_STEP),
        "--NE", str(NE), "--NI", str(NI),
        "--record_path", str(out_dir) + "/",
        "--full_mode", "2",
        "--conn_matrix_file", str(conn_path),
        "--overwrite_conn", "1",
        "--fE", f"{f:.6f}", "--fI", f"{f:.6f}",
        "--Nu", f"{Nu:.6f}",
        "--record_spk", "1",
    ]
    if record_state:
        cmd += [
            "--record_v", "1", "--record_vlim", f"{win_start:.6g} {t_max:.6g}",
            "--record_IE", "1", "--record_II", "1",
            "--record_Ilim", f"{win_start:.6g} {t_max:.6g}",
        ]
    res = subprocess.run(cmd, capture_output=True, universal_newlines=True)
    if res.returncode != 0:
        raise RuntimeError(f"simHH failed (J_E={J_E}, J_I={J_I}):\n{res.stderr}")
    return out_dir, (win_start, t_max)


# ============================================================ measurement
def _read_trace(path, t_range):
    """(n_t, N+1) array [t, per-neuron values] restricted to t_range."""
    data = np.fromfile(path, dtype=float)
    data = data[: (data.size // (N + 1)) * (N + 1)].reshape(-1, N + 1)
    return data[(data[:, 0] >= t_range[0]) & (data[:, 0] < t_range[1])]


def load_spikes(out_dir, t_range):
    spk = np.fromfile(next(Path(out_dir).glob("*_spike_train.dat")), dtype=float)
    spk = spk[: (spk.size // 2) * 2].reshape(-1, 2)
    if spk.size == 0:
        return spk
    return spk[(spk[:, 0] >= t_range[0]) & (spk[:, 0] < t_range[1])]


def spike_metrics(spk, t_range, bin_ms=20.0, max_pairs_neurons=120):
    """Rates, ISI irregularity and population synchrony."""
    dur_s = (t_range[1] - t_range[0]) * 1e-3
    counts = np.bincount(spk[:, 1].astype(int), minlength=N) if spk.size else np.zeros(N)
    rates = counts / dur_s
    out = {
        "rate_E": float(rates[:NE].mean()),
        "rate_I": float(rates[NE:].mean()),
        "rate_all": float(rates.mean()),
        "frac_silent": float((rates < 0.5).mean()),
    }

    # ISI CV, per neuron, median over neurons with enough spikes
    cvs = []
    if spk.size:
        order = np.argsort(spk[:, 1], kind="stable")
        ids, ts = spk[order, 1].astype(int), spk[order, 0]
        edges = np.searchsorted(ids, np.arange(N + 1))
        for i in range(N):
            tt = np.sort(ts[edges[i]:edges[i + 1]])
            if tt.size >= 6:
                isi = np.diff(tt)
                if isi.mean() > 0:
                    cvs.append(isi.std() / isi.mean())
    out["isi_cv"] = float(np.median(cvs)) if cvs else np.nan
    out["n_cv_neurons"] = len(cvs)

    # pairwise spike-count correlation (asynchrony)
    active = np.where(rates > 0.5)[0]
    if active.size >= 10 and spk.size:
        if active.size > max_pairs_neurons:
            active = np.random.default_rng(0).choice(active, max_pairs_neurons, replace=False)
        edges_t = np.arange(t_range[0], t_range[1] + bin_ms, bin_ms)
        mat = np.stack([
            np.histogram(spk[spk[:, 1] == i, 0], bins=edges_t)[0] for i in active
        ]).astype(float)
        keep = mat.std(axis=1) > 0
        mat = mat[keep]
        if mat.shape[0] >= 10:
            c = np.corrcoef(mat)
            iu = np.triu_indices_from(c, k=1)
            out["pair_corr"] = float(np.nanmean(c[iu]))
        else:
            out["pair_corr"] = np.nan
    else:
        out["pair_corr"] = np.nan

    # population spike-count CV (kept for comparability with the old sweeps)
    if spk.size:
        pc, _ = np.histogram(spk[:, 0], bins=np.arange(t_range[0], t_range[1] + 8, 8))
        out["pop_cv"] = float(pc.std() / pc.mean()) if pc.mean() > 0 else np.nan
    else:
        out["pop_cv"] = np.nan
    return out


def conductance_metrics(out_dir, t_range, f, Nu):
    """Recover time-averaged conductances from the recorded currents.

    I_E = -(G_f+G_se)(v-V_E) and I_I = -G_si(v-V_I), so
        g_E = I_E / (V_E - v),   g_I = I_I / (V_I - v).
    Samples inside a spike are dropped: at v ~ +40 mV I_E flips sign and I_I
    grows ~8x, which corrupts any population average (this is the likely cause
    of the unsigned-imbalance blow-up in the earlier Jei sweep).

    The recorded I_E lumps feedforward and recurrent excitation together, so
    g_se is obtained by subtracting the analytic g_f = f*Nu*INT_E.
    """
    out_dir = Path(out_dir)
    v = _read_trace(next(out_dir.glob("*_voltage.dat")), t_range)
    ie = _read_trace(next(out_dir.glob("*_IE.dat")), t_range)
    ii = _read_trace(next(out_dir.glob("*_II.dat")), t_range)
    n = min(len(v), len(ie), len(ii))
    if n == 0:
        return {k: np.nan for k in
                ("g_E", "g_se", "g_si", "g_ratio", "v_star", "frac_spiking", "rec_frac")}
    vv, iee, iii = v[:n, 1:], ie[:n, 1:], ii[:n, 1:]

    sub = vv < SPIKE_MASK_V
    g_E = np.where(sub, iee / (V_E - vv), np.nan)
    g_I = np.where(sub, iii / (V_I - vv), np.nan)

    g_E_mean = float(np.nanmean(g_E))
    g_I_mean = float(np.nanmean(g_I))
    g_f = f * Nu * INT_E
    g_se = g_E_mean - g_f

    # v_star over `sub` alone still averages over the post-spike
    # after-hyperpolarisation, which drags it down by several mV at high rate.
    # v_clean additionally drops POST_SPIKE_MS after every upward crossing, so
    # it reports the drive between spikes rather than the AHP.
    # chunked over neurons: the index arrays below are the same size as the
    # voltage trace, and a long validation window would otherwise need GBs
    n_post = int(POST_SPIKE_MS / T_STEP)
    row = np.arange(vv.shape[0], dtype=np.int32)[:, None]
    means, stds, frac_clean = [], [], []
    for lo in range(0, vv.shape[1], 100):
        blk = vv[:, lo:lo + 100]
        last = np.maximum.accumulate(
            np.where(blk > SPIKE_MASK_V, row, np.int32(-2 ** 30)), axis=0)
        ok = (row - last) > n_post
        blk_clean = np.where(ok, blk, np.nan)
        means.append(np.nanmean(blk_clean, axis=0))
        stds.append(np.nanstd(blk_clean, axis=0))
        frac_clean.append(ok.mean(axis=0))
    v_clean_mean = float(np.nanmean(np.concatenate(means)))
    v_clean_sd = float(np.nanmean(np.concatenate(stds)))
    frac_clean = float(np.mean(np.concatenate(frac_clean)))

    return {
        "g_E": g_E_mean,
        "g_f": float(g_f),
        "g_se": float(g_se),
        "g_si": g_I_mean,
        "g_ratio": float(g_I_mean / g_E_mean) if g_E_mean > 0 else np.nan,
        "rec_frac": float(g_se / g_E_mean) if g_E_mean > 0 else np.nan,
        "v_eff": float(v_eff(g_E_mean, g_I_mean)),
        "v_star": float(np.nanmean(np.where(sub, vv, np.nan))),
        "v_std": float(np.nanmean(np.nanstd(np.where(sub, vv, np.nan), axis=0))),
        "v_clean": v_clean_mean,
        "v_clean_std": v_clean_sd,
        "frac_clean": frac_clean,
        "frac_spiking": float(1.0 - sub.mean()),
    }


def evaluate(J_E, J_I, f, Nu, t_max, out_dir, cleanup=True, seed=CONN_SEED,
             window=None):
    """One simulation -> full metric dict."""
    out_dir, win = run_sim(J_E, J_I, f, Nu, t_max, out_dir, seed=seed,
                           window=window)
    res = {"J_E": float(J_E), "J_I": float(J_I), "f": float(f), "Nu": float(Nu),
           "J_ratio": float(J_I / J_E), "t_max": float(t_max)}
    res.update(spike_metrics(load_spikes(out_dir, (BURN_IN, t_max)), (BURN_IN, t_max)))
    res.update(conductance_metrics(out_dir, win, f, Nu))
    res["sane"] = bool(
        RATE_MIN <= res["rate_all"] <= RATE_MAX
        and res["frac_silent"] < 0.5
        and np.isfinite(res.get("g_E", np.nan))
    )
    res["balanced"] = bool(
        res["sane"]
        and 0.7 <= res.get("isi_cv", 0) <= 1.4
        and (not np.isfinite(res.get("pair_corr", np.nan)) or res["pair_corr"] < 0.1)
        and res.get("g_E", 0) >= 0.8 * G_L
        and (1 - G_RATIO_TOL) * G_RATIO <= res.get("g_ratio", 0) <= (1 + G_RATIO_TOL) * G_RATIO
        and res.get("v_eff", 0) < V_INIT
        and res.get("v_clean", 0) < V_TH
    )
    if cleanup:
        for pat in ("*_voltage.dat", "*_IE.dat", "*_II.dat", "*_state.dat"):
            for p in out_dir.glob(pat):
                p.unlink()
    return res


# ============================================================ modes
def _sweep_worker(args):
    idx, J_E, J_I = args
    f = f_from_Nu(NU)
    tag = f"JE{J_E:.3f}_JI{J_I:.3f}"
    try:
        return evaluate(J_E, J_I, f, NU, SWEEP_T_MAX, OUT_ROOT / "sweep" / tag)
    except Exception as exc:  # a blown-up run should not kill the batch
        return {"J_E": float(J_E), "J_I": float(J_I), "J_ratio": float(J_I / J_E),
                "error": str(exc)[:300], "sane": False, "balanced": False}


def mode_sweep(n_proc=8, scales=None, ratios=None):
    """Batch grid over the overall coupling scale and the J_I/J_E ratio.

    The analytic prediction is J_I/J_E = (g_si/g_se)*(INT_E/INT_I) ~ 5.1 for
    the targets in this file; the grid brackets it because that derivation
    assumes r_E = r_I and that the nominal conductance targets are reached.
    """
    J_E0, J_I0 = J_from_rate(10.0)
    scales = np.array(scales if scales is not None else [0.25, 0.5, 1.0, 1.5, 2.0])
    ratios = np.array(ratios if ratios is not None else [1.5, 2.25, 3.0, 4.0])
    grid = [(i, J_E0 * s, J_E0 * s * q)
            for i, (s, q) in enumerate((s, q) for s in scales for q in ratios)]
    print(f"seed J_E={J_E0:.3f} J_I={J_I0:.3f} (r=10 Hz), f={f_from_Nu(NU):.4f}, "
          f"Nu={NU}, {len(grid)} points")
    with Pool(min(n_proc, len(grid))) as pool:
        results = pool.map(_sweep_worker, grid)
    _save(results, OUT_ROOT / "sweep_results.json")
    _report(results)
    return results


def g_I_for_veff(v_eff_target, g_E=G_E_TARGET):
    """Inhibitory conductance placing the membrane's effective reversal at
    `v_eff_target`.  Lower (more hyperpolarised) v_eff means firing is driven
    further out into the fluctuation tail, which raises ISI CV."""
    return ((g_E * V_E + G_L * E_L - v_eff_target * (g_E + G_L))
            / (v_eff_target - V_I))


def _grid_worker(point):
    J_E, J_I, Nu = point["J_E"], point["J_I"], point.get("Nu", NU)
    f = point.get("f", f_from_Nu(Nu))
    tag = f"JE{J_E:.3f}_JI{J_I:.3f}_Nu{Nu:.3f}"
    try:
        res = evaluate(J_E, J_I, f, Nu, SWEEP_T_MAX, OUT_ROOT / "grid" / tag)
        res.update({k: v for k, v in point.items() if k not in res})
        return res
    except Exception as exc:
        return {**point, "error": str(exc)[:300], "sane": False, "balanced": False}


def mode_grid(points, n_proc=8, out_name="grid_results.json"):
    """Batch-evaluate an explicit list of {J_E, J_I, Nu} dicts."""
    print(f"{len(points)} points")
    with Pool(min(n_proc, len(points))) as pool:
        results = pool.map(_grid_worker, points)
    _save(results, OUT_ROOT / out_name)
    _report(results)
    return results


def mode_fixedpoint(n_iter=8, damping=0.5, J_E=None, J_I=None):
    """Iterate J_E, J_I so the measured conductances hit their targets.

    Multiplicative correction with exponent `damping`: raising J raises the
    rate which raises g again, so an undamped g_target/g_measured step
    overshoots.
    """
    f = f_from_Nu(NU)
    if J_E is None or J_I is None:
        J_E, J_I = J_from_rate(10.0)
    history = []
    for it in range(n_iter):
        res = evaluate(J_E, J_I, f, NU, FIXPOINT_T_MAX, OUT_ROOT / "fixedpoint" / f"it{it:02d}")
        res["iter"] = it
        history.append(res)
        print(f"[{it}] J_E={J_E:.4f} J_I={J_I:.4f} | rate={res['rate_all']:.1f} Hz "
              f"CV={res['isi_cv']:.2f} corr={res['pair_corr']:.3f} | "
              f"g_E={res['g_E']:.3f} g_se={res['g_se']:.3f} g_si={res['g_si']:.3f} "
              f"ratio={res['g_ratio']:.2f} v_eff={res['v_eff']:.1f} "
              f"v_clean={res['v_clean']:.1f} | balanced={res['balanced']}")
        if res["balanced"]:
            print("converged")
            break
        if not np.isfinite(res.get("g_se", np.nan)) or res["g_se"] <= 0:
            print("g_se non-positive: recurrent excitation vanished, raising J_E blindly")
            J_E *= 1.6
            continue
        J_E *= (G_SE_TARGET / res["g_se"]) ** damping
        J_I *= (G_I_TARGET / max(res["g_si"], 1e-6)) ** damping
        J_E, J_I = float(np.clip(J_E, 1e-3, 50)), float(np.clip(J_I, 1e-3, 100))
    _save(history, OUT_ROOT / "fixedpoint_results.json")
    return history


def mode_validate(J_E, J_I, t_max=1e5, seed=CONN_SEED):
    """Long production-length run with the full metric set."""
    f = f_from_Nu(NU)
    res = evaluate(J_E, J_I, f, NU, t_max, OUT_ROOT / "validate", cleanup=False, seed=seed)
    _save([res], OUT_ROOT / "validate_results.json")
    generate_connectivity(J_E, J_I, seed=seed, path=CONN_PATH)
    print(json.dumps(res, indent=2))
    print(f"\nconnectivity saved to {CONN_PATH}")
    return res


# ============================================================ helpers
def _save(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"saved {path}")


def _report(results):
    cols = ("J_E", "J_I", "J_ratio", "rate_all", "rate_E", "rate_I", "isi_cv",
            "pair_corr", "g_E", "g_se", "g_si", "g_ratio", "v_eff", "v_clean",
            "balanced")
    print("  ".join(f"{c:>9}" for c in cols))
    for r in sorted(results, key=lambda x: (x.get("J_E", 0), x.get("J_I", 0))):
        print("  ".join(
            f"{r.get(c, float('nan')):>9.3f}" if isinstance(r.get(c), (int, float))
            else f"{str(r.get(c)):>9}" for c in cols))
    good = [r for r in results if r.get("balanced")]
    print(f"\n{len(good)}/{len(results)} points met the balance criteria")
    for r in good:
        print(f"  J_E={r['J_E']:.3f} J_I={r['J_I']:.3f} rate={r['rate_all']:.1f} Hz "
              f"CV={r['isi_cv']:.2f} g_E={r['g_E']:.3f} ratio={r['g_ratio']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["sweep", "fixedpoint", "validate"])
    ap.add_argument("--n-proc", type=int, default=8)
    ap.add_argument("--J-E", type=float, default=None)
    ap.add_argument("--J-I", type=float, default=None)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--t-max", type=float, default=1e5)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "sweep":
        mode_sweep(n_proc=args.n_proc)
    elif args.mode == "fixedpoint":
        mode_fixedpoint(n_iter=args.n_iter, J_E=args.J_E, J_I=args.J_I)
    else:
        if args.J_E is None or args.J_I is None:
            ap.error("validate needs --J-E and --J-I")
        mode_validate(args.J_E, args.J_I, t_max=args.t_max)
