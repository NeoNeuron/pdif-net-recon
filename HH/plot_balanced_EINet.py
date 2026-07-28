"""Figures for the balanced simHH network found by HH/find_balanced_params.py.

Fig 1 (balance):  E and I input currents of one example neuron, their sum, the
                  membrane potential, and a population raster.
Fig 2 (transfer): mean firing rate vs. the feedforward Poisson rate Nu, at
                  FIXED synaptic strength f -- so the external drive really is
                  being varied.  (find_balanced_params.f_from_Nu does the
                  opposite: it holds g_f = f*Nu*INT_E constant, which is what
                  you want when searching for parameters and exactly what you
                  must not do when measuring a transfer function.)
Fig 3-4 (nu scan): the same experiment on a uniform Nu = 0.1..2.0 grid (0.1
                  spacing), giving rate/current/conductance summaries plus one
                  I_E/I_I trace panel per Nu.  Separate output files from fig 2,
                  which spans 0.01-4.0 and would lose both its low-Nu plateau
                  and its high-Nu compressive branch if overwritten.

Run from the repo root (as a module -- it imports HH.find_balanced_params):
    python -m HH.plot_balanced_EINet            # everything
    python -m HH.plot_balanced_EINet figs       # figs 1-2 only
    python -m HH.plot_balanced_EINet nu-scan    # figs 3-4 only (20 runs, ~1 min)
"""
# %%
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from HH.find_balanced_params import (
    BURN_IN, G_L, E_L, G_RATIO, G_RATIO_TOL, INT_E, K, N, NE, NI, OUT_ROOT,
    SPIKE_MASK_V, T_STEP, V_E, V_I, _read_trace, conductance_metrics, evaluate,
    f_from_Nu, generate_connectivity, load_spikes, run_sim, spike_metrics,
)
from HH.run_balanced_EINet import DEFAULT_OUT as BALANCED_RUN_DIR

# validated parameters (data/EINet/balanced_search/best_params.json).
# Decorrelated set: recurrent fraction 0.1, chosen to remove the 76 Hz ING
# rhythm that striped the raster at the earlier 0.5 (see HH/reduce_correlation.py).
J_E, J_I = 0.3553, 8.381
NU_REF = 0.9
F_FIXED = f_from_Nu(NU_REF)   # 0.2333; holds g_f at its target for this Nu

FIG_DIR = OUT_ROOT / "figures"
WIN_MS = 500.0        # length of the example trace shown in fig 1
T_MAX_TRACE = 3e3
T_MAX_SWEEP = 1e4     # long enough for a stable rate at every Nu


def subthreshold_mask(vc, dilate_ms=2.0):
    """True where a single-neuron voltage trace is *not* inside a spike.

    During an action potential v reaches ~+40 mV, so I_I = -g_si(v+80)
    transiently hits ~-300 uA/cm^2 and I_E changes sign (the cell is then above
    the excitatory reversal potential); plotting or averaging those swamps the
    subthreshold structure that balance is about.  Same masking the metrics in
    find_balanced_params.py use, dilated by +-`dilate_ms`: the bare v < -40 test
    still admits samples on the spike's rising and falling edges, which show up
    as large spurious transients.
    """
    w = int(dilate_ms / T_STEP)
    spiking = np.convolve((vc >= SPIKE_MASK_V).astype(float),
                          np.ones(2 * w + 1), mode="same") > 0
    return ~spiking


# ------------------------------------------------------------------ figure 1
def figure_balance(nu=NU_REF, blank_spikes=True, out_name=None):
    """Example-neuron E/I input currents.

    `blank_spikes=False` keeps the trace through each action potential.  Be
    aware of what that shows: at the peak of a spike v ~ +40 mV, so
    I_I = -g_si(v+80) reaches ~-300 uA/cm^2 and I_E changes sign (the cell is
    then *above* the excitatory reversal potential, so excitatory channels
    carry outward current).  Those excursions are real membrane currents, but
    they are spike-generated, not the synaptic drive that balance is about,
    and they set the y-scale.
    """
    out_dir, win = run_sim(J_E, J_I, F_FIXED, nu, T_MAX_TRACE,
                           OUT_ROOT / f"fig_balance_nu{nu:g}")
    t0 = win[1] - WIN_MS
    v = _read_trace(next(Path(out_dir).glob("*_voltage.dat")), (t0, win[1]))
    ie = _read_trace(next(Path(out_dir).glob("*_IE.dat")), (t0, win[1]))
    ii = _read_trace(next(Path(out_dir).glob("*_II.dat")), (t0, win[1]))
    spk = load_spikes(out_dir, (t0, win[1]))
    n = min(len(v), len(ie), len(ii))
    v, ie, ii = v[:n], ie[:n], ii[:n]
    t = v[:, 0]

    # example neuron: the excitatory cell whose rate is closest to the E mean
    counts = np.bincount(spk[:, 1].astype(int), minlength=N)
    cell = int(np.argmin(np.abs(counts[:NE] - counts[:NE].mean())))
    vc = v[:, 1 + cell]
    sub = subthreshold_mask(vc)
    blank = (lambda a: np.where(sub, a, np.nan)) if blank_spikes else (lambda a: a)
    I_E, I_I = blank(ie[:, 1 + cell]), blank(ii[:, 1 + cell])
    I_L = blank(G_L * (E_L - vc))
    net = I_E + I_I + I_L
    cell_spk = spk[spk[:, 1] == cell, 0]

    fig, axes = plt.subplots(
        4, 1, figsize=(9, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4, 1.2, 1.8]})

    ax = axes[0]
    ax.plot(t, I_E, lw=0.8, color="#c0392b", label=r"$I_E=-(g_f{+}g_{se})(v-0)$")
    ax.plot(t, I_I, lw=0.8, color="#2471a3", label=r"$I_I=-g_{si}(v+80)$")
    ax.axhline(np.nanmean(np.where(sub, ie[:, 1 + cell], np.nan)),
               color="#c0392b", ls="--", lw=0.7)
    ax.axhline(np.nanmean(np.where(sub, ii[:, 1 + cell], np.nan)),
               color="#2471a3", ls="--", lw=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel(r"current ($\mu$A/cm$^2$)")
    ax.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.9)
    ax.set_title(
        f"simHH E neuron #{cell} ($J_E$={J_E}, $J_I$={J_I}, $N_u$={nu:g} kHz, "
        f"K={K}, spikes {'blanked' if blank_spikes else 'shown'})", fontsize=11)

    # net on the SAME y-scale as panel 0 -- that visual comparison is the
    # whole point: two large opposing currents leaving a small residual.
    ax = axes[1]
    ax.plot(t, net, lw=0.8, color="k", label=r"net $I_E+I_I+I_{leak}$")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylim(axes[0].get_ylim())
    ax.set_ylabel(r"net ($\mu$A/cm$^2$)")
    ax.legend(loc="upper right", fontsize=8)
    # Statistics are ALWAYS subthreshold-only, whether or not the trace is
    # blanked: averaging the spike excursions in would report the action
    # potential, not the synaptic balance.
    sE, sI = np.where(sub, ie[:, 1 + cell], np.nan), np.where(sub, ii[:, 1 + cell], np.nan)
    sL = np.where(sub, G_L * (E_L - vc), np.nan)
    snet = sE + sI + sL
    ratio = np.nanmean(np.abs(snet)) / np.nanmean(np.abs(sE))
    ax.text(0.01, 0.08,
            f"subthreshold: mean $I_E$={np.nanmean(sE):+.1f}, "
            f"mean $I_I$={np.nanmean(sI):+.1f}, mean net={np.nanmean(snet):+.1f}   "
            f"(mean|net| / mean|$I_E$| = {ratio:.2f})",
            transform=ax.transAxes, fontsize=8)

    ax = axes[2]
    ax.plot(t, v[:, 1 + cell], lw=0.6, color="#444444")
    ax.axhline(SPIKE_MASK_V, color="gray", ls=":", lw=0.6)
    ax.set_ylabel("V (mV)")

    ax = axes[3]
    ax.plot(spk[:, 0], spk[:, 1], "|", ms=1.6, color="k", alpha=0.6)
    ax.plot(cell_spk, np.full_like(cell_spk, cell), "|", ms=8, color="#c0392b")
    ax.axhline(NE, color="#2471a3", lw=0.8)
    ax.set_ylabel("neuron id")
    ax.set_xlabel("time (ms)")
    ax.set_xlim(t[0], t[-1])

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if out_name is None:
        out_name = (f"balance_example_neuron_nu{nu:g}"
                    f"{'' if blank_spikes else '_withspikes'}.png")
    fig.savefig(FIG_DIR / out_name, dpi=160)
    print(f"saved {FIG_DIR / out_name}")

    stats = {
        "Nu": float(nu), "cell": cell, "blank_spikes": bool(blank_spikes),
        "subthreshold": {
            "mean_I_E": float(np.nanmean(sE)), "mean_I_I": float(np.nanmean(sI)),
            "mean_I_leak": float(np.nanmean(sL)),
            "mean_net": float(np.nanmean(snet)), "std_net": float(np.nanstd(snet)),
            "std_I_E": float(np.nanstd(sE)), "std_I_I": float(np.nanstd(sI)),
            "abs_net_over_abs_IE": float(ratio),
        },
        "including_spikes": {
            "mean_I_E": float(np.nanmean(ie[:, 1 + cell])),
            "mean_I_I": float(np.nanmean(ii[:, 1 + cell])),
            "min_I_I": float(np.nanmin(ii[:, 1 + cell])),
            "max_I_E": float(np.nanmax(ie[:, 1 + cell])),
            "min_I_E": float(np.nanmin(ie[:, 1 + cell])),
        },
        "rate_of_cell_Hz": float(len(cell_spk) / (WIN_MS * 1e-3)),
    }
    print(json.dumps(stats, indent=2))
    return stats


# ------------------------------------------------------------- figure 1b
def figure_balance_saved(data_dir=BALANCED_RUN_DIR, out_name="balance_from_run.png",
                          spike_blank_ms=4.0):
    """Example-neuron E/I input currents from an EXISTING run_balanced_EINet.py
    output. Pure plotting: only reads .dat files already on disk, never calls
    simHH (contrast figure_balance, which simulates its own trace).

    run_balanced_EINet's default `--record-currents` run does not save
    voltage.dat (only `--record-v` does), so there is no membrane trace to
    threshold a spike out of. Spikes are instead blanked directly from the
    recorded spike train: +-spike_blank_ms around each spike time of the
    example cell. This cell's own recorded I_E/I_I already carry its action
    potential's driving-force swing (I_E=-(g_f+g_se)(v-V_E) etc. use this
    cell's own v), which decays over ~3-4 ms here -- checked empirically
    against the raw trace -- hence the wider default than the 2 ms voltage-
    based dilation figure_balance uses. Net current also omits the leak term
    for the same reason (I_L = G_L*(E_L-v) needs v).
    """
    data_dir = Path(data_dir)
    ie = np.fromfile(next(data_dir.glob("*_IE.dat")), dtype=float)
    ie = ie[: (ie.size // (N + 1)) * (N + 1)].reshape(-1, N + 1)
    ii = np.fromfile(next(data_dir.glob("*_II.dat")), dtype=float)
    ii = ii[: (ii.size // (N + 1)) * (N + 1)].reshape(-1, N + 1)
    n = min(len(ie), len(ii))
    ie, ii = ie[:n], ii[:n]
    t = ie[:, 0]
    t_range = (float(t[0]), float(t[-1]))
    spk = load_spikes(data_dir, t_range)

    # example neuron: the excitatory cell whose spike count is closest to the
    # E-population mean, same convention as figure_balance
    counts = np.bincount(spk[:, 1].astype(int), minlength=N)
    cell = int(np.argmin(np.abs(counts[:NE] - counts[:NE].mean())))
    I_E, I_I = ie[:, 1 + cell], ii[:, 1 + cell]
    cell_spk = spk[spk[:, 1] == cell, 0]

    dt = float(t[1] - t[0])
    w = int(round(spike_blank_ms / dt))
    mask = np.ones(n, dtype=bool)
    for i in np.searchsorted(t, cell_spk):
        mask[max(0, i - w):min(n, i + w + 1)] = False
    I_E, I_I = np.where(mask, I_E, np.nan), np.where(mask, I_I, np.nan)
    net = I_E + I_I

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.6]})

    ax = axes[0]
    ax.plot(t, I_E, lw=0.8, color="#c0392b", label=r"$I_E$")
    ax.plot(t, I_I, lw=0.8, color="#2471a3", label=r"$I_I$")
    ax.plot(t, net, lw=0.8, color="k", alpha=0.75, label=r"net $I_E{+}I_I$")
    ax.axhline(np.nanmean(I_E), color="#c0392b", ls="--", lw=0.7)
    ax.axhline(np.nanmean(I_I), color="#2471a3", ls="--", lw=0.7)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel(r"current ($\mu$A/cm$^2$)")
    ax.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.9)
    ax.set_title(f"E neuron #{cell}, from {data_dir} (spikes blanked "
                 f"+-{spike_blank_ms:g} ms, no leak term -- no voltage.dat "
                 f"in this run)", fontsize=10)

    ax = axes[1]
    ax.plot(spk[:, 0], spk[:, 1], "|", ms=1.6, color="k", alpha=0.6)
    ax.plot(cell_spk, np.full_like(cell_spk, cell), "|", ms=8, color="#c0392b")
    ax.axhline(NE, color="#2471a3", lw=0.8)
    ax.set_ylabel("neuron id")
    ax.set_xlabel("time (ms)")
    ax.set_xlim(t[0], t[-1])

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / out_name, dpi=160)
    print(f"saved {FIG_DIR / out_name}")

    stats = {
        "data_dir": str(data_dir), "cell": cell, "t_range_ms": t_range,
        "spike_blank_ms": spike_blank_ms,
        "mean_I_E": float(np.nanmean(I_E)), "mean_I_I": float(np.nanmean(I_I)),
        "mean_net": float(np.nanmean(net)),
        "rate_of_cell_Hz": float(len(cell_spk) / ((t_range[1] - t_range[0]) * 1e-3)),
    }
    print(json.dumps(stats, indent=2))
    return stats


# ------------------------------------------------------------------ figure 2
def _nu_worker(nu):
    """Rate at feedforward rate `nu`, with f held FIXED (drive really varies)."""
    try:
        r = evaluate(J_E, J_I, F_FIXED, nu, T_MAX_SWEEP,
                     OUT_ROOT / "nu_transfer" / f"nu_{nu:.4f}")
        return {"Nu": float(nu), **{k: r[k] for k in
                ("rate_all", "rate_E", "rate_I", "isi_cv", "pair_corr",
                 "g_E", "g_si", "g_ratio", "v_eff", "frac_silent")}}
    except Exception as exc:
        return {"Nu": float(nu), "error": str(exc)[:200]}


def figure_nu_transfer(nu_list=None, n_proc=8):
    # spans 400x in drive: the low end exposes the self-sustained plateau, the
    # high end the compressive (shunting-dominated) branch
    if nu_list is None:
        nu_list = np.concatenate([[0.01, 0.02, 0.05, 0.1, 0.2],
                                  np.linspace(0.35, 4.0, 12)])
    nu_list = np.asarray(nu_list)
    with Pool(min(n_proc, len(nu_list))) as pool:
        res = pool.map(_nu_worker, nu_list)
    res = [r for r in res if "error" not in r]
    res.sort(key=lambda r: r["Nu"])
    nu = np.array([r["Nu"] for r in res])
    ra = np.array([r["rate_all"] for r in res])
    re_, ri = (np.array([r[k] for r in res]) for k in ("rate_E", "rate_I"))
    gE = np.array([r["g_E"] for r in res])

    gI = np.array([r["g_si"] for r in res])
    veff = np.array([r["v_eff"] for r in res])

    # global straight line, and one fitted only to the upper (non-plateau) part
    coef = np.polyfit(nu, ra, 1)
    pred = np.polyval(coef, nu)
    r2 = 1 - float(((ra - pred) ** 2).sum()) / float(((ra - ra.mean()) ** 2).sum())
    hi = nu >= 0.35
    coef_hi = np.polyfit(nu[hi], ra[hi], 1)
    r2_hi = 1 - float(((ra[hi] - np.polyval(coef_hi, nu[hi])) ** 2).sum()) / \
        float(((ra[hi] - ra[hi].mean()) ** 2).sum())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.plot(nu, ra, "o-", color="k", label="all", zorder=3)
    ax.plot(nu, re_, "o-", color="#c0392b", ms=4, label="E")
    ax.plot(nu, ri, "o-", color="#2471a3", ms=4, label="I")
    ax.plot(nu, np.polyval(coef_hi, nu), "--", color="gray",
            label=(f"fit on $N_u\\geq$0.35: {coef_hi[0]:.2f}$N_u${coef_hi[1]:+.1f}"
                   f"\n$R^2$={r2_hi:.4f} (there), {r2:.4f} (all)"))
    ax.axvline(NU_REF, color="green", ls=":", lw=1)
    ax.set_xlabel(r"$N_u$ (feedforward Poisson rate, kHz)")
    ax.set_ylabel("mean firing rate (Hz)")
    ax.set_title("rate vs. drive ($f$ fixed)")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    ax.semilogx(nu, ra, "o-", color="k")
    ax.axvline(NU_REF, color="green", ls=":", lw=1)
    ax.axhline(ra[0], color="gray", ls="--", lw=0.8)
    ax.set_xlabel(r"$N_u$ (kHz, log)")
    ax.set_ylabel("mean firing rate (Hz)")
    ax.set_title(f"{nu[-1]/nu[0]:.0f}$\\times$ drive $\\to$ only "
                 f"{ra[-1]/ra[0]:.1f}$\\times$ rate")
    ax.text(0.03, 0.9, f"self-sustained at\n$N_u\\to$0: {ra[0]:.1f} Hz",
            transform=ax.transAxes, fontsize=8, va="top")

    ax = axes[2]
    ax.plot(nu, gE, "o-", color="#c0392b", ms=4, label=r"$g_E$")
    ax.plot(nu, gI, "o-", color="#2471a3", ms=4, label=r"$g_I$")
    ax.plot(nu, gE + gI + G_L, "o-", color="k", ms=4, label=r"$g_E{+}g_I{+}G_L$")
    ax.axhline(G_L, color="gray", ls="--", lw=0.8, label=r"$G_L$")
    ax.axvline(NU_REF, color="green", ls=":", lw=1)
    ax.set_xlabel(r"$N_u$ (kHz)")
    ax.set_ylabel(r"conductance (mS/cm$^2$)")
    ax.legend(fontsize=7, loc="upper left")
    axt = ax.twinx()
    axt.plot(nu, veff, "s--", color="#7d3c98", ms=3, lw=0.9)
    axt.set_ylabel(r"$v_{eff}$ (mV)", color="#7d3c98")
    axt.tick_params(axis="y", colors="#7d3c98")
    ax.set_title("shunting (black) vs. depolarisation (purple)")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "rate_vs_Nu.png", dpi=160)
    print(f"saved {FIG_DIR / 'rate_vs_Nu.png'}")

    with open(FIG_DIR / "rate_vs_Nu.json", "w") as fh:
        json.dump({"fit_slope_Hz_per_kHz": float(coef[0]),
                   "fit_intercept_Hz": float(coef[1]),
                   "r2_full": float(r2), "r2_upper_half": float(r2_hi),
                   "points": res}, fh, indent=2)
    for r in res:
        print(f"Nu={r['Nu']:.3f}  rate={r['rate_all']:6.2f} Hz "
              f"(E {r['rate_E']:6.2f} / I {r['rate_I']:6.2f})  CV={r['isi_cv']:.3f}  "
              f"g_E={r['g_E']:.3f} g_I={r['g_si']:.3f} v_eff={r['v_eff']:.1f}")
    print(f"\nlinear fit: rate = {coef[0]:.2f}*Nu {coef[1]:+.2f},  "
          f"R^2 = {r2:.4f} (full), {r2_hi:.4f} (upper half)")
    return res


# ------------------------------------------- figures 3-4: the 0.1..2.0 Nu scan
#
# Same experiment as figure_nu_transfer (f pinned at F_FIXED so the drive really
# varies) on a uniform 0.1..2.0 grid, but each run also keeps a short example
# trace so the E/I currents can be shown case by case.  Written to its own file
# names: rate_vs_Nu.png covers 0.01-4.0 and would lose its low-Nu plateau and
# high-Nu compressive branch if overwritten with this subset.
NU_SCAN = np.round(np.arange(0.1, 2.0 + 1e-9, 0.1), 3)
SCAN_T_MAX = 2e4        # a run costs ~10 s, so length is not the constraint.
                        # Rate converges over 400 neurons long before this; ISI
                        # CV is the slow one and is still biased low here (~0.78
                        # vs 0.80 at 1e5), so read it as a trend across Nu, not
                        # as a per-Nu value.
SCAN_TRACE_MS = 500.0   # recorded per run; ~190 KB per simulated ms for v+IE+II
SCAN_PLOT_MS = 300.0    # shown per panel
SCAN_CELLS = 6          # candidate example neurons kept per run
SCAN_DIR = OUT_ROOT / "nu_scan"


def _scan_cells():
    """The same candidate E neurons in every run, so panels are comparable."""
    return np.sort(np.random.default_rng(11).choice(NE, SCAN_CELLS, replace=False))


def _scan_worker(nu):
    """One Nu: metrics + a short example trace, then drop the big .dat files."""
    out_dir = SCAN_DIR / f"nu_{nu:.2f}"
    try:
        out_dir, win = run_sim(J_E, J_I, F_FIXED, nu, SCAN_T_MAX, out_dir,
                               window=SCAN_TRACE_MS)
        res = {"Nu": float(nu)}
        res.update(spike_metrics(load_spikes(out_dir, (BURN_IN, SCAN_T_MAX)),
                                 (BURN_IN, SCAN_T_MAX)))
        res.update(conductance_metrics(out_dir, win, F_FIXED, nu))

        v = _read_trace(next(Path(out_dir).glob("*_voltage.dat")), win)
        ie = _read_trace(next(Path(out_dir).glob("*_IE.dat")), win)
        ii = _read_trace(next(Path(out_dir).glob("*_II.dat")), win)
        n = min(len(v), len(ie), len(ii))
        v, ie, ii = v[:n], ie[:n], ii[:n]

        # Population-level subthreshold currents: the summary version of the
        # per-case traces, and the only place the *net* current is quantified
        # across the scan (conductance_metrics returns conductances, not
        # currents).  Averaged over all N neurons, spikes excluded.
        vv = v[:, 1:]
        sub = vv < SPIKE_MASK_V
        sE = np.where(sub, ie[:, 1:], np.nan)
        sI = np.where(sub, ii[:, 1:], np.nan)
        sL = np.where(sub, G_L * (E_L - vv), np.nan)
        res.update({
            "I_E_mean": float(np.nanmean(sE)), "I_E_std": float(np.nanstd(sE)),
            "I_I_mean": float(np.nanmean(sI)), "I_I_std": float(np.nanstd(sI)),
            "I_L_mean": float(np.nanmean(sL)),
            "net_mean": float(np.nanmean(sE + sI + sL)),
            "net_std": float(np.nanstd(sE + sI + sL)),
        })

        cells = _scan_cells()
        trace = SCAN_DIR / f"trace_nu_{nu:.2f}.npz"
        np.savez_compressed(
            trace, t=v[:, 0], cells=cells, v=v[:, 1 + cells],
            I_E=ie[:, 1 + cells], I_I=ii[:, 1 + cells],
            rates=np.bincount(load_spikes(out_dir, (BURN_IN, SCAN_T_MAX))[:, 1]
                              .astype(int), minlength=N)
            / ((SCAN_T_MAX - BURN_IN) * 1e-3))
        res["trace"] = str(trace)
        return res
    except Exception as exc:
        return {"Nu": float(nu), "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        # in a finally block on purpose: an exception escaping before this would
        # strand ~95 MB per run of v/I_E/I_I on disk.  The connectivity goes too
        # -- every run uses the same pinned graph, and simHH writes two more
        # copies of it per run, so keeping them costs 3.8 MB x 20 for nothing;
        # scan_nu saves one shared copy alongside.
        for pat in ("*_voltage.dat", "*_IE.dat", "*_II.dat", "*_state.dat",
                    "conn.npy", "connect_matrix*.npy"):
            for p in out_dir.glob(pat):
                p.unlink()


def scan_nu(nu_list=NU_SCAN, n_proc=8):
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    generate_connectivity(J_E, J_I, path=SCAN_DIR / "conn.npy")   # shared copy
    with Pool(min(n_proc, len(nu_list))) as pool:
        res = pool.map(_scan_worker, nu_list)
    bad = [r for r in res if "error" in r]
    for r in bad:
        print(f"  Nu={r['Nu']:.2f} FAILED: {r['error']}")
    res = sorted((r for r in res if "error" not in r), key=lambda r: r["Nu"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIG_DIR / "nu_scan.json", "w") as fh:
        json.dump({"J_E": J_E, "J_I": J_I, "f": F_FIXED, "t_max": SCAN_T_MAX,
                   "n_failed": len(bad), "points": res}, fh, indent=2)
    print(f"saved {FIG_DIR / 'nu_scan.json'}  ({len(res)}/{len(nu_list)} ok)")
    return res


def _example_cell(res):
    """One cell for all panels, chosen once at the reference Nu.

    figure_balance picks the cell closest to the E-population mean rate *at that
    Nu*; reused per panel that would show a different neuron in each and destroy
    the comparison.
    """
    ref = min(res, key=lambda r: abs(r["Nu"] - NU_REF))
    z = np.load(ref["trace"])
    cells, rates = z["cells"], z["rates"]
    return int(cells[np.argmin(np.abs(rates[cells] - rates[:NE].mean()))])


def balance_window(res):
    """(lo, hi) Nu over which the scan still meets the balance criteria.

    Varying Nu at fixed f moves the operating point: g_E is proportional to the
    drive, so the low end is leak-dominated (g_E << G_L) and the high end is
    excitation-heavy (g_ratio below its band).  Only the middle of the requested
    0.1-2.0 range is actually a balanced network, and the figure should say so
    rather than let "E and I grow together" be read as balance throughout.
    Thresholds are evaluate()'s own, not eyeballed.
    """
    nu = np.array([r["Nu"] for r in res])
    ok = (np.array([r["g_E"] for r in res]) >= 0.8 * G_L) & (
        np.abs(np.array([r["g_ratio"] for r in res]) - G_RATIO)
        <= G_RATIO_TOL * G_RATIO)
    if not ok.any():
        return None
    # the run of consecutive passing points containing the operating point
    i = int(np.argmin(np.abs(nu - NU_REF)))
    lo = hi = i
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return (float(nu[lo]), float(nu[hi])) if ok[i] else None


def figure_nu_scan_rate(res):
    nu = np.array([r["Nu"] for r in res])
    ra, re_, ri = (np.array([r[k] for r in res])
                   for k in ("rate_all", "rate_E", "rate_I"))
    gE = np.array([r["g_E"] for r in res])
    gI = np.array([r["g_si"] for r in res])
    win = balance_window(res)
    coef = np.polyfit(nu, ra, 1)
    r2 = 1 - float(((ra - np.polyval(coef, nu)) ** 2).sum()) / \
        float(((ra - ra.mean()) ** 2).sum())

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    ax = axes[0]
    ax.plot(nu, ra, "o-", color="k", label="all", zorder=3)
    ax.plot(nu, re_, "o-", color="#c0392b", ms=4, label="E")
    ax.plot(nu, ri, "o-", color="#2471a3", ms=4, label="I")
    ax.plot(nu, np.polyval(coef, nu), "--", color="gray",
            label=f"{coef[0]:.2f}$N_u${coef[1]:+.1f},  $R^2$={r2:.4f}")
    ax.axvline(NU_REF, color="green", ls=":", lw=1)
    ax.set_xlabel(r"$N_u$ (kHz)")
    ax.set_ylabel("mean firing rate (Hz)")
    ax.set_title(f"{nu[-1]/nu[0]:.0f}$\\times$ drive $\\to$ "
                 f"{ra[-1]/ra[0]:.2f}$\\times$ rate")
    ax.legend(fontsize=8, loc="lower right")
    # R^2 alone would read as "linear" here; it is high because a straight line
    # through a gently saturating curve still explains most of the variance.
    # The end-to-end slope ratio is the honest statement of the curvature.
    lo_s = (ra[2] - ra[0]) / (nu[2] - nu[0])
    hi_s = (ra[-1] - ra[-3]) / (nu[-1] - nu[-3])
    ax.text(0.03, 0.95,
            f"slope {lo_s:.1f} Hz/kHz at $N_u\\!\\approx${nu[1]:.1f}\n"
            f"       {hi_s:.1f} Hz/kHz at $N_u\\!\\approx${nu[-2]:.1f}\n"
            f"$\\Rightarrow$ sublinear ({lo_s/hi_s:.1f}$\\times$ gain drop)",
            transform=ax.transAxes, fontsize=8, va="top")

    # subthreshold currents: the population summary of the per-case traces
    ax = axes[1]
    for key, col, lab in (("I_E", "#c0392b", r"$I_E$"),
                          ("I_I", "#2471a3", r"$I_I$")):
        m = np.array([r[f"{key}_mean"] for r in res])
        s = np.array([r[f"{key}_std"] for r in res])
        ax.plot(nu, m, "o-", color=col, ms=4, label=lab)
        ax.fill_between(nu, m - s, m + s, color=col, alpha=0.18, lw=0)
    m = np.array([r["net_mean"] for r in res])
    s = np.array([r["net_std"] for r in res])
    ax.plot(nu, m, "o-", color="k", ms=4, label=r"net (incl. $I_{leak}$)")
    ax.fill_between(nu, m - s, m + s, color="k", alpha=0.18, lw=0)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(NU_REF, color="green", ls=":", lw=1)
    ax.set_xlabel(r"$N_u$ (kHz)")
    ax.set_ylabel(r"subthreshold current ($\mu$A/cm$^2$)")
    ax.set_title("E and I grow together, net stays small")
    # the band pools across neurons and time, so it is not the temporal
    # fluctuation that drives spiking -- say so rather than let it read that way
    ax.legend(fontsize=8, loc="upper left", title=r"mean $\pm$ SD (neurons & time)",
              title_fontsize=7)   # centre is where the net trace is

    ax = axes[2]
    ax.plot(nu, gE, "o-", color="#c0392b", ms=4, label=r"$g_E$")
    ax.plot(nu, gI, "o-", color="#2471a3", ms=4, label=r"$g_I$")
    ax.plot(nu, gE + gI + G_L, "o-", color="k", ms=4, label=r"$g_E{+}g_I{+}G_L$")
    ax.axhline(G_L, color="gray", ls="--", lw=0.8, label=r"$G_L$")
    ax.axvline(NU_REF, color="green", ls=":", lw=1)
    ax.set_xlabel(r"$N_u$ (kHz)")
    ax.set_ylabel(r"conductance (mS/cm$^2$)")
    ax.legend(fontsize=8, loc="upper left")
    axt = ax.twinx()
    axt.plot(nu, [r["v_eff"] for r in res], "s--", color="#7d3c98", ms=3, lw=0.9)
    axt.set_ylabel(r"$v_{eff}$ (mV)", color="#7d3c98")
    axt.tick_params(axis="y", colors="#7d3c98")
    ax.set_title("shunting (black) vs. depolarisation (purple)")

    if win is not None:
        for ax in axes:
            ax.axvspan(win[0], win[1], color="green", alpha=0.07, lw=0, zorder=0)
        axes[0].text(0.5 * (win[0] + win[1]), 0.97, "balanced", color="green",
                     fontsize=8, ha="center", va="top",
                     transform=axes[0].get_xaxis_transform())

    sub = (f"balance criteria ($g_E\\geq$0.8$G_L$, $g_I/g_E$ within "
           f"{G_RATIO_TOL:.0%} of {G_RATIO:.1f}) hold only for "
           f"$N_u\\in$[{win[0]:.1f}, {win[1]:.1f}] kHz (shaded); "
           f"outside it the network is leak-dominated (low $N_u$) or "
           f"excitation-heavy (high $N_u$)"
           if win is not None else "")
    fig.suptitle(f"simHH balanced network, $N_u$ scan at fixed $f$={F_FIXED:.4f} "
                 f"($J_E$={J_E}, $J_I$={J_I}, K={K}, T={SCAN_T_MAX:.0f} ms)\n"
                 + sub, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG_DIR / "nu_scan_rate.png", dpi=160)
    print(f"saved {FIG_DIR / 'nu_scan_rate.png'}")
    return coef, r2


def figure_nu_scan_currents(res, cell=None, ncol=5):
    """One I_E/I_I panel per Nu, same neuron and same y-scale throughout."""
    cell = _example_cell(res) if cell is None else cell
    win = balance_window(res)
    nrow = int(np.ceil(len(res) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.1 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    lo, hi = np.inf, -np.inf
    drawn = []
    for ax, r in zip(axes, res):
        z = np.load(r["trace"])
        j = int(np.where(z["cells"] == cell)[0][0])
        t, vc = z["t"], z["v"][:, j]
        keep = t >= t[-1] - SCAN_PLOT_MS
        t, vc = t[keep], vc[keep]
        # blank spikes: the rate rises ~4x across the scan, so unblanked spike
        # artifacts would grow with Nu and break comparability at exactly the
        # end of the range this scan is about
        sub = subthreshold_mask(vc)
        I_E = np.where(sub, z["I_E"][keep, j], np.nan)
        I_I = np.where(sub, z["I_I"][keep, j], np.nan)
        net = I_E + I_I + np.where(sub, G_L * (E_L - vc), np.nan)
        t0 = t - t[0]
        ax.plot(t0, I_E, lw=0.5, color="#c0392b")
        ax.plot(t0, I_I, lw=0.5, color="#2471a3")
        ax.plot(t0, net, lw=0.5, color="k", alpha=0.75)
        ax.axhline(0, color="gray", lw=0.4)
        # spike times of this cell, from its own voltage trace
        spk_t = t0[np.flatnonzero((vc[:-1] < SPIKE_MASK_V)
                                  & (vc[1:] >= SPIKE_MASK_V))]
        ax.plot(spk_t, np.full_like(spk_t, 1.0), "|", ms=5, color="#7d3c98",
                transform=ax.get_xaxis_transform(), clip_on=False)
        ax.set_title(f"$N_u$={r['Nu']:.1f} kHz   {r['rate_all']:.1f} Hz",
                     fontsize=9, pad=8)
        # tint the panels that actually meet the balance criteria: the traces
        # look qualitatively similar throughout, but only these are a balanced
        # network (see balance_window)
        if win is not None and win[0] - 1e-9 <= r["Nu"] <= win[1] + 1e-9:
            ax.set_facecolor((0.0, 0.5, 0.0, 0.05))
        if abs(r["Nu"] - NU_REF) < 1e-9:
            for s in ax.spines.values():
                s.set_color("green"), s.set_linewidth(1.6)
        lo = min(lo, np.nanmin(I_I))
        hi = max(hi, np.nanmax(I_E))
        drawn.append(ax)

    for ax in axes[len(res):]:
        ax.axis("off")
    pad = 0.05 * (hi - lo)
    drawn[0].set_ylim(lo - pad, hi + pad)      # shared
    for ax in drawn[-ncol:]:
        ax.set_xlabel("time (ms)")
    for ax in drawn[::ncol]:
        ax.set_ylabel(r"$\mu$A/cm$^2$")
    fig.legend(handles=[
        plt.Line2D([], [], color="#c0392b", lw=1.2, label=r"$I_E=-(g_f{+}g_{se})(v-0)$"),
        plt.Line2D([], [], color="#2471a3", lw=1.2, label=r"$I_I=-g_{si}(v+80)$"),
        plt.Line2D([], [], color="k", lw=1.2, label=r"net $I_E{+}I_I{+}I_{leak}$"),
        plt.Line2D([], [], color="#7d3c98", lw=0, marker="|", ms=7, label="spike"),
    ], loc="lower center", ncol=4, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"E and I input currents of E neuron #{cell}, spikes blanked "
        f"(last {SCAN_PLOT_MS:.0f} ms, shared axes)\n"
        + (f"green frame = operating point; tinted = meets the balance criteria "
           f"($N_u\\in$[{win[0]:.1f}, {win[1]:.1f}] kHz)"
           if win is not None else "green frame = operating point"),
        fontsize=11)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(FIG_DIR / "nu_scan_IE_II.png", dpi=160)
    print(f"saved {FIG_DIR / 'nu_scan_IE_II.png'}")
    return cell


def nu_scan(n_proc=8):
    res = scan_nu(n_proc=n_proc)
    coef, r2 = figure_nu_scan_rate(res)
    cell = figure_nu_scan_currents(res)
    print(f"\nexample neuron #{cell}   "
          f"rate = {coef[0]:.2f}*Nu {coef[1]:+.2f} Hz,  R^2 = {r2:.4f}")
    hdr = ("Nu", "rate_all", "rate_E", "rate_I", "isi_cv", "pair_corr",
           "I_E_mean", "I_I_mean", "net_mean", "g_E", "g_si", "v_eff")
    print("  ".join(f"{h:>9}" for h in hdr))
    for r in res:
        print("  ".join(f"{r[h]:>9.4g}" for h in hdr))
    return res


if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "figs"):
        figure_balance()                               # balanced point, blanked
        figure_balance(nu=2.0, blank_spikes=False)     # strong drive, spikes kept
        figure_nu_transfer()
    if what in ("all", "nu-scan"):
        nu_scan()
