# Benchmark harness (`scripts/benchmark/`)

Common-interface harness that runs every causality-estimation method against the same set of
synthetic ground-truth networks, so their reconstruction accuracy (AUC) and runtime can be
compared apples-to-apples. Downstream figures (`fig_nc/fig5.py`, `fig_nc/figR4-6.py`) read the
`.pkl` files this harness produces.

Each method has its own script (`PTD-TE.py`, `DDC.py`, `STE.py`, `GLMCC.py`, `CCM.py`), runnable
standalone via CLI or subprocess (see `benchmark.sh`). `run_method.py` wraps all five behind one
function/CLI (`run_method(method, key, val, ...)` / `--method <name>`) — see
[Unified dispatcher: `run_method.py`](#unified-dispatcher-run_methodpy) below; prefer it over
importing the individual scripts directly when calling them from other Python code.

## Pipeline

```
gen_data.py                 simulate networks with bin/sim* or simulate_*.py, dump voltage .npy
  -> binarization.py        (optional) pick spike threshold from the 90th-pct voltage histogram
  -> measurement_noise.py   (optional) generate noise-corrupted spike trains for robustness tests
  -> {PTD-TE,DDC,STE,GLMCC,CCM}.py --key <dataset> --cfg-file <config.yml>
       (or run_method.py --method <name> --key <dataset> --cfg-file <config.yml>, same options)
       each writes  <root>/results/<METHOD>/recon_df_noise_<level>_<key>_<fullnet|shuffle_id>.pkl
  -> fig_nc/fig5.py, fig_nc/figR4-6.py   read the .pkl files, compute AUC, make comparison figures
```

`benchmark.sh` (repo root) is the canonical full sweep — loops all 8 datasets through PTD-TE, DDC,
STE, GLMCC, then CCM/FDCCM/SCCM, using `benchmark10_causal.yml` (the N=10 config).

## Datasets (the `--key` values)

Defined per-network in the `*_causal.yml` files. The 8 keys used in the main sweep:

| key | model | pulse/continuous | notes |
|---|---|---|---|
| `HHEE` | Hodgkin-Huxley, all-excitatory | pulse | `bin/simHH` |
| `HHEI` | HH, excitatory+inhibitory | pulse | |
| `HHconEE` | HH continuous-voltage variant | continuous | `bin/simHHcon` |
| `HHconEI` | HHcon, E+I | continuous | |
| `Lorenz` | Lorenz system, pulse-thresholded | pulse | `bin/simLorenz` |
| `Logistic` | coupled logistic maps | continuous | `bin/simLogistic` |
| `Rcon` | Rössler, continuous | continuous | `bin/simRcon` |
| `RNN` | rate RNN (`simulate_RNN.py`) | continuous | Python simulator, not C++ |

Extra keys exist in the configs but aren't part of the standard sweep: `HHII`, `HHconII` (all-
inhibitory), `Lcon` (continuous Lorenz), `Gaussian` (OU-type Python simulator).

## Config files (which one to use)

- `benchmark10.yml` / `benchmark100.yml` — **simulation** params (consumed by `gen_data.py` via
  `utils.run_simulation`) for the N=10 vs. N=100 node networks. Same keys, differ in `NE`/`NI`,
  `seed`, `record_path` (`benchmark/N10/...` vs `benchmark/N100/...`).
- `benchmark10_causal.yml` / `benchmark100_causal.yml` — **causality-estimation** params (order,
  `dt`, `delay`, `spk_fname`, `conn_file`, `path`) matching the N10/N100 simulations. Note
  `conn_file` extension differs: `.npy` for N10, `.dat` for N100 (raw connectivity matrix format
  written by the simulator differs by node count — always check which one a config expects).
- `benchmark_causal.yml` — the **default** `--cfg-file` for all five method scripts and what
  `fig_nc/fig5.py` reads. It's an N=100-flavored config but trimmed to just the 8 main-sweep keys
  (no `HHII`/`HHconII`/`Lcon`/`Gaussian`) — this is what "the benchmark" means unless you pass
  `--cfg-file benchmark10_causal.yml` explicitly for a quick N=10 smoke test.
- `binarization.yaml` — per-key `threshold` / `refractory` / `dt` for converting continuous
  voltage traces to spike trains (`causal4.utils.binarize`); used by `binarization.py`,
  `measurement_noise.py`, `GLMCC.py`, `PTD-TE.py` (noisy-spike variant).

## Method scripts

All five share the same CLI: `--key <dataset> [--idx <shuffle_id>] [--noise_level <float>]
[--T <duration>] [--cfg-file <config.yml>]`, and a `core_function(key, val, shuffle_id,
noise_level, T)` doing the actual work (importable, so can be called directly instead of via
CLI/subprocess).

`--T` truncates how much data is fed into the estimator (data-length scaling runs), overriding
the config's `T`/duration. It's `None` by default (full recorded duration, identical to prior
behavior and output filenames). When given, each script converts it to whatever unit it natively
needs:
- `PTD-TE.py` / `GLMCC.py`: overrides `val['T']` directly before the C++ estimator / GLMCC's own
  `T/1e3`-or-`/1e4` scaling runs — same unit as the config's `T` field (ms).
- `DDC.py`: passed straight through to `causal4.ddc.DDC_long(..., T=T)` (native time units, same
  as the voltage file's own `dt`) — see below.
- `STE.py` / `CCM.py`: converted to a raw sample count `L = int(T/dt)` using the voltage file's
  own native `dt` (read live from the file, not from the causal config), then the array is
  truncated to `[:L]` before estimation. `CCM.py`'s `--T` overrides its hardcoded `L_dict[key]`
  default sample count.

Truncated runs get a `_T=<value>` tag inserted into the output `.pkl` filename (GLMCC already had
this via its own `T={T:.0f}` tag) so they don't collide with full-length results.

| script | measure | output column(s) | extra dependency |
|---|---|---|---|
| `PTD-TE.py` | pointwise transfer entropy (via `bin/calCausality`) | `TE`, `Delta_p` | none (uses `causal4.Causality.CausalityEstimator`) |
| `DDC.py` | Dynamic Differential Covariance | `ddc`, `ddc_abs`, `log-ddc_abs` | none (`causal4.ddc.DDC_long`) |
| `STE.py` | symbolic transfer entropy | `ste`, `log-ste` | **`smite`** package (`smite.symbolic_transfer_entropy_matrix`) — not in `requirements.txt`, install separately |
| `GLMCC.py` | GLM-based cross-correlogram | `glmcc`, `glmcc_abs`, `log-glmcc_abs` | **`glmcc`** package (`glmcc.Est_Data.Est_Data`) — not in `requirements.txt` |
| `CCM.py` | convergent cross mapping, 3 variants via `--ccm {CCM,FDCCM,SCCM}` | `ccm`, `log-ccm` | in-repo `crossmap_indices.py` |

`DDC.py` always goes through `causal4.ddc.DDC_long(fname, N, indices, n_blocks=20, preprocess,
T=None, max_memory_gb=None, ram_fraction=None)` — it no longer has a separate single-shot
`DDC()` path for truncated runs; `--T` (any value, including `None`) is passed straight through,
so full-length and truncated runs both go through the same block-averaged (`n_blocks=20`)
estimator. (This is a behavior change from earlier: truncated `DDC.py` runs used to call the
single-shot `DDC()` directly, bypassing block-averaging — any old `_T=...` DDC `.pkl` results
predate this and are numerically not comparable to new ones.)

`DDC_long` itself grew three new params on top of its original signature:
- `T:float=None` — truncate to the first `T` (native time units, same as the voltage file's own
  `dt`) of data before splitting into `n_blocks`. `None` (default) uses the full file, bit-identical
  to the pre-`T` behavior.
- `max_memory_gb:float=None` — `DDC_long` always prints an estimated peak per-block memory
  (`causal4.ddc.estimate_ddc_block_memory`) so you can judge whether a given `n_blocks` will fit in
  RAM before a long run. If set and the estimate for the current `n_blocks` exceeds it, `n_blocks`
  is auto-increased (`causal4.ddc.suggest_n_blocks`) to fit, with a printed note. `None` (default)
  is a no-op beyond the info line.
- `ram_fraction:float=None` — same auto-bump behavior as `max_memory_gb`, but the budget is
  computed as `ram_fraction * causal4.ddc.get_system_ram_bytes()` (requires `psutil`, now in
  `requirements.txt`) instead of a fixed value, e.g. `ram_fraction=0.1` caps a block at ~10% of the
  machine's total RAM. Mutually exclusive with `max_memory_gb` (raises `ValueError` if both given).

`crossmap_indices.py` implements the actual CCM/FDCCM(frequency-domain)/SCCM(symbolic) embedding
and cross-mapping math (`ClassicalCCM`, `FrequencyCCM`, `SymbolicCCM` in `cm`). `CCM.py` hardcodes
per-dataset `L_dict` (samples read) and `tau_dict` (downsample stride / FDCCM lag) — **these were
tuned by hand per dataset**, so if you add a new dataset key you must add entries there or CCM
silently uses a `KeyError`.

`GLMCC.py` scales `T` differently by dataset family: `T/1e3` for `{HHEE, HHEI, HHconEE, HHconEI,
Lorenz}`, `T/1e4` for `{Gaussian, Rcon, Logistic, RNN}` — this reflects each family's native time
unit, not a bug; matches the `WIN`/`DELTA` bin sizes in `binarization.yaml`'s `dt` entry.

### Default data length per dataset (when `--T` is omitted)

Two different "dt"s are in play here and it's easy to conflate them:
- `benchmark100.yml`'s `T_step` is the **numerical integration step** the simulator actually
  advances by — i.e. the true time resolution of the recorded voltage/state trace (every
  simulated step gets written to the `.npy` file). This is confirmed by `T_Max/T_step` matching
  `CCM.py`'s hardcoded `L_dict` "total available samples" comments (`CCM.py:45-54`) exactly for
  every dataset (e.g. `HHconEE`: `1.0e7/0.05 = 2.0e8`, matching that file's `# total 2e8`).
  `DDC_long`/`STE.py` read this value live off the file itself (`dt = dat[1,0]-dat[0,0]`), never
  off a config field.
- `benchmark_causal.yml`'s `dt` is a coarser, *downstream* analysis bin size — the TE bin width
  PTD-TE's `order`/`delay` are expressed in, and (via `STE.py`'s `stride =
  int(val['dt']/dt_from_file)`) the stride STE decimates the raw `T_step`-resolution trace to
  before calling `smite`. It is **not** the recording's sampling interval.

Also note `benchmark_causal.yml`'s `T` field is numerically identical to `benchmark100.yml`'s
`T_Max` for every dataset in the main sweep — i.e. it already represents that dataset's *full*
simulated duration, not some smaller default.

| dataset | `T_step` (ms/sample, true recording resolution) | causal-config `dt` (ms, analysis bin only) | `T_Max` = full recorded duration (ms) | total raw samples = `T_Max/T_step` |
|---|---|---|---|---|
| `HHEE`     | 0.2  | 0.5 | 1.0×10⁷ ms (10,000 s ≈ 2.8 h)  | 5×10⁷ |
| `HHEI`     | 0.2  | 0.5 | 1.0×10⁷ ms (10,000 s ≈ 2.8 h)  | 5×10⁷ |
| `HHconEE`  | 0.05 | 0.5 | 1.0×10⁷ ms (10,000 s ≈ 2.8 h)  | 2×10⁸ |
| `HHconEI`  | 0.05 | 0.5 | 1.0×10⁷ ms (10,000 s ≈ 2.8 h)  | 2×10⁸ |
| `Lorenz`   | 0.01 | 0.02| 1.0×10⁶ ms (1000 s ≈ 16.7 min) | 1×10⁸ |
| `Logistic` | 1    | 1   | 1.0×10⁸ (a.u., discrete map)   | 1×10⁸ |
| `Rcon`     | 0.01 | 3.0 | 1.0×10⁷ ms (10,000 s ≈ 2.8 h)  | 1×10⁹ |
| `RNN`      | 1    | 5   | 1.0×10⁸ ms (100,000 s ≈ 27.8 h)| 1×10⁸ |

With `T=None`, **`PTD-TE.py`, `GLMCC.py`, `DDC.py`, and `STE.py` all default to the same full
duration, `T_Max`** — they just consume it at different resolutions (see below). Only
`CCM.py`'s three variants default to something much shorter, via their own hardcoded `L_dict`
(sample count, at `T_step` resolution — **not** scaled by the causal-config `dt`):

| dataset | `CCM`/`SCCM`/`FDCCM` default span = `L_dict × T_step` | % of `T_Max` | `CCM`/`SCCM` points used (`L_dict/tau`, after decimation) | `FDCCM` points used (`L_dict`, full resolution) |
|---|---|---|---|---|
| `HHEE`     | 2×10⁵ ms (200 s)   | 2% | 1×10⁵ (τ=10)     | 1×10⁶ |
| `HHEI`     | 2×10⁵ ms (200 s)   | 2% | 1×10⁵ (τ=10)     | 1×10⁶ |
| `HHconEE`  | 1×10⁵ ms (100 s)   | 1% | 66,667 (τ=30)    | 2×10⁶ |
| `HHconEI`  | 1×10⁵ ms (100 s)   | 1% | 66,667 (τ=30)    | 2×10⁶ |
| `Lorenz`   | 1×10⁴ ms (10 s)    | 1% | 1×10⁵ (τ=10)     | 1×10⁶ |
| `Logistic` | 1×10⁶ (a.u.)       | 1% | 1×10⁶ (τ=1, no decimation) | 1×10⁶ |
| `Rcon`     | 1×10⁵ ms (100 s)   | 1% | 1×10⁵ (τ=100)    | 1×10⁷ |
| `RNN`      | 1×10⁶ ms (1000 s ≈ 16.7 min) | 1% | 5×10⁵ (τ=2) | 1×10⁶ |

Key takeaways:
- Duration-wise, `PTD-TE`/`GLMCC`/`DDC`/`STE` are all apples-to-apples by default (full `T_Max`);
  `CCM`/`SCCM`/`FDCCM` are the outlier, seeing only ~1-2% of the recording by default because CCM
  is far more expensive per point.
- Point-density is *not* apples-to-apples even among the four full-duration methods: `DDC.py`
  reads every raw sample (`T_Max/T_step` points, the finest of all five — e.g. 2×10⁸ for
  `HHconEE`), while `STE.py` strides down to roughly `T_Max`/(causal `dt`) points
  (`int(causal_dt/T_step)`-step decimation), and PTD-TE bins spikes at the causal-config
  `order`/`dt` scale rather than working sample-by-sample.
- Pass `--T` explicitly (same value in ms, converted per-method as described above) if you need a
  true apples-to-apples data-*length* comparison across methods for a given dataset — it still
  won't equalize point density, only duration.

### Unified dispatcher: `run_method.py`

`run_method.py` gives all five method scripts one common call signature, so you don't need to know
each script's own import quirks to run any of them:

```python
from run_method import run_method, load_config

pm_causal_set = load_config('benchmark_causal.yml')   # or benchmark10_causal.yml
run_method('PTD-TE', 'HHEE', pm_causal_set['HHEE'], shuffle_id=None, noise_level=None, T=None)
run_method('FDCCM', 'HHEE', pm_causal_set['HHEE'])     # CCM's 3 variants: CCM/FDCCM/SCCM
```

or from the CLI: `python run_method.py --method DDC --key HHEE --T 5e4 --cfg-file benchmark_causal.yml`
(same `--key`/`--idx`/`--noise_level`/`--T`/`--cfg-file` flags as the individual scripts, plus a
required `--method`).

Why this exists rather than just `import PTD-TE`: filenames like `PTD-TE.py` have a `-`, which
isn't valid in a Python `import` statement, and `STE.py`/`GLMCC.py` import their optional
third-party deps (`smite`/`glmcc`) at module scope — a naive up-front import of all five would
break dispatch to the other three on any machine missing those packages. `run_method.py` instead
loads each script lazily by file path (`importlib.util.spec_from_file_location`, cached per
method) the first time that specific method is requested, so a `ModuleNotFoundError` for `smite`/
`glmcc` only affects calls to `STE`/`GLMCC`, never `PTD-TE`/`DDC`/`CCM`.

## Robustness axes: subnetwork resampling and measurement noise

- `gen_data.py`'s "Part 3" generates `benchmark/N100/subnet_indices.npy` — 10 random 10-node
  subsets of the 100-node network (seed 42), used as `--idx <shuffle_id>` so every method can also
  be tested on N=10 subnetworks sampled out of the larger N=100 simulation (not just the dedicated
  N10 run).
- `--noise_level <float>` adds Gaussian measurement noise (σ = `noise_level` × std of the first
  10000 samples) to the continuous trace before re-estimating; `measurement_noise.py` /
  `binarization.py` are the exploratory notebooks used to pick noise σ / threshold values (produce
  `PDF_level_0.9.pdf`, `measurement_noise_std=0.4.pdf`) and to pre-generate noisy binarized spike
  trains via `get_spk_fname(..., f'_noisy{noise_level:.1f}', ...)`.
- Each method script's commented-out tail block (e.g. bottom of `PTD-TE.py`) shows the full sweep
  that was actually run for the noise-robustness figures: `noise_level in [0.1, 0.2, 0.3, 0.4]` ×
  all keys × all 10 subnet shuffles — that's what produced the `recon_df_noise_<level>_<key>_<i>.pkl`
  files (as opposed to the `_fullnet.pkl` ones from the plain full-network run with `--idx` unset).

## Output layout

All method scripts write to `<repo_root>/results/<METHOD>/`:
- `recon_df_noise_0_<key>_fullnet.pkl` — full N-node network, no shuffle, no noise.
- `recon_df_noise_0_<key>_<shuffle_id>.pkl` — 10-node subnetwork `shuffle_id` (0-9), no noise.
- `recon_df_noise_<level>_<key>_<shuffle_id>.pkl` — subnetwork + noise level.
- GLMCC additionally embeds `T={T:.0f}` in the filename (its `T` is measure-specific, see above).

Each pickle is a `pandas.DataFrame` (`pre_id`, `post_id`, measure column(s), `connection` = ground
truth) with `.attrs['wall_time']` / `.attrs['cpu_time']` for the runtime comparison plots.

## Consumers

- `fig_nc/fig5.py` — loads `benchmark_causal.yml`, sweeps `net_keys × {PTD-TE, STE, GLMCC, DDC,
  CCM, FDCCM, SCCM} × {fullnet, noisy-subnet-0.3}`, computes `roc_auc_score(connection, <measure>)`
  and reconstruction figures. The `labels` dict there (`{'PTD-TE':'TE', 'STE':'ste', 'GLMCC':
  'glmcc_abs', 'DDC':'ddc_abs', 'CCM'/'FDCCM'/'SCCM':'ccm'}`) is the canonical mapping from method
  name to the DataFrame column to score — check it first if a new method's column isn't picking up.
- `fig_nc/figR4-6.py` — related supplementary figures, same `.pkl` inputs.

## Gotchas / things to remember

- `smite` and `glmcc` are **not** in `requirements.txt` / README install docs — install them
  manually before running `STE.py` / `GLMCC.py`, or expect `ModuleNotFoundError`. (`psutil` *is*
  now in `requirements.txt` — only used by `DDC_long`'s `ram_fraction` option.)
- `benchmark/` (simulation output) and presumably `results/` are gitignored — nothing under them
  ships in the repo; re-run `gen_data.py` + the method scripts from scratch on a fresh checkout.
  Careful: the `.gitignore` pattern is `benchmark/` with no leading `/`, so it also matches
  `scripts/benchmark/` itself — new files added directly in this directory (e.g. `run_method.py`,
  this README) are silently untracked by plain `git add` unless force-added (`git add -f`) or the
  pattern is scoped to `/benchmark/`.
- `conn_file` extension convention differs between N10 (`.npy`) and N100 (`.dat`) configs — the
  method scripts often do `.with_suffix('.npy')` explicitly, so don't assume the extension in the
  yml is what actually gets loaded.
- Root-level `test_benchmarks.py` is an older, unrelated ad hoc exploration script (different path
  convention: `benchmark/HH/EE/N=100/...`) — not part of this harness, don't confuse the two.
- (fixed) `PTD-TE.py` used to eagerly `np.load(.../N100/subnet_indices.npy)` at module scope —
  dead code (immediately shadowed inside `core_function`) that crashed the script on import
  whenever that file didn't exist yet, even for runs that never touch subnetting. Removed.
