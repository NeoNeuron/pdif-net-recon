# PDIF network reconstruction

Code for the in-review journal paper (submitted to a Science-partner journal) on **PDIF
(Pairwise Delayed Information Flow)** — a generalization of transfer entropy (TE) to general
continuous time series (`scripts/benchmark/PDIF.py`, formerly named `PTD-TE.py`). All of this
paper's network reconstructions use PDIF; the other methods already present in
`scripts/benchmark/` — `DDC.py` (Dynamic Differential Covariance), `STE.py` (symbolic transfer
entropy), `GLMCC.py` (GLM-based cross-correlogram), `CCM.py` (convergent cross mapping, incl.
FDCCM/SCCM variants) — are baseline methods PDIF is benchmarked against, not alternative
reconstruction methods used for this paper's own results.

Network systems covered (broader than the companion PNAS paper's HH-only main text): `HH`,
`HHcon`, `Lorenz`, `Lcon`, `Logistic`, `Rcon`, plus two pure-Python simulators
(`simulate_gaussian.py`, `simulate_RNN.py`) and real Allen Institute Neuropixels
(visual coding) / two-photon (visual behavior) recordings.

This repo was split out of the original `causal4-dev` monorepo with `git filter-repo`, so file
history predates this repo's creation — `git log` on any file still shows its real authorship
history.

## Setup

### 1. C/C++ dependencies (needed to build `simHH`, `simHHcon`, `simLorenz`, `simLcon`, `simRcon`, `simLogistic`)

- [Eigen](https://eigen.tuxfamily.org), [Boost](http://www.boost.org/users/download/) (only
  `program_options` is needed), OpenMP.

**Linux (Ubuntu):**
```bash
sudo apt-get update && sudo apt-get install -y libeigen3-dev libboost-all-dev
```

**macOS:**
```bash
brew install boost eigen libomp
```
The `makefile` auto-detects macOS and adds Homebrew's include/lib paths automatically; it uses
Apple Clang, not Homebrew GCC (mixing Homebrew GCC/libstdc++ with Homebrew Boost/libc++ risks ABI
mismatches).

**Windows (MSYS2/MinGW-w64):**
```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make mingw-w64-x86_64-boost mingw-w64-x86_64-eigen3
```
Build with `mingw32-make all -j` from an MSYS2 MINGW64 shell. Native MSVC isn't supported.

```bash
make all -j     # builds bin/simHH bin/simHHcon bin/simLorenz bin/simLcon bin/simRcon bin/simLogistic
```

### 2. Python

```bash
conda create -n pdif-net-recon python=3.11 -y
conda activate pdif-net-recon
pip install -r requirements.txt
```
This installs [`PDIF`](https://github.com/NeoNeuron/PDIF) (the shared
causality-estimation package — `pip install`-ing it also builds its `calCausality` binary, no
separate step needed) plus this repo's own Python dependencies.

`scripts/benchmark/STE.py` and `scripts/benchmark/GLMCC.py` additionally need the `smite` and
`glmcc` packages respectively — install those manually; they aren't on PyPI under obvious names
and aren't vendored here (see "Known gaps" below).

## Usage

### Benchmark harness (`scripts/benchmark/`)

Runs every causality-estimation method against the same synthetic ground-truth networks so
reconstruction accuracy (AUC) and runtime are comparable apples-to-apples:

```
gen_data.py                 simulate networks with bin/sim* or simulate_*.py, dump voltage .npy
  -> binarization.py        (optional) pick spike threshold from the voltage histogram
  -> measurement_noise.py   (optional) generate noise-corrupted spike trains
  -> {PDIF,DDC,STE,GLMCC,CCM}.py --key <dataset> --cfg-file <config.yml>
       each writes  results/<METHOD>/recon_df_noise_<level>_<key>_<fullnet|shuffle_id>.pkl
  -> fig_nc/fig5.py, fig_nc/figR4-6.py   read the .pkl files, compute AUC, make figures
```

- Quick local sweep: `./benchmarkN10.sh` (N=10, all 5 methods).
- Full N=100 sweep on a SLURM cluster: see `slurms/` — run `sbatch slurms/bm_noisy.slurm` first
  (pre-generates noisy spike trains), then `./slurms/batch_bm_{PDIF,DDC,STE,GLMCC,CCM}.sh` from
  the repo root (paths in the `.slurm` files are resolved relative to the submission directory).
- Full details, dataset keys, config file roles, and per-method gotchas: see
  `scripts/benchmark/README.md`.

### Figures (`fig_nc/`)

Each `fig_nc/fig*.py` / `fig_nc/figR*.py` produces one paper figure or supplementary figure,
sharing plotting config from `fig_nc/figrc.py` / `fig_nc/matplotlibrc`. Most read `.pkl` results
already produced by the benchmark harness or the Allen-data pipeline below.

### Allen Institute real-data pipeline

Visual coding (Neuropixels):
```
extract_visualcoding_data2pkl.py  ->  allen_data_causality_estimation.py  ->  test_visualcoding.py  ->  test_visualcoding_postprocess.py
```
Visual behavior (two-photon), same shape:
```
extract_visualbehavior_data2pkl.py  ->  allen_data_causality_estimation.py  ->  test_visualbehavior.py  ->  test_visualbehavior_postprocess.py
```
`data/download_visual_coding.ipynb` / `data/download_visual_behavior.ipynb` are the AllenSDK
download tutorials — run these first to fetch the raw session data (requires `allensdk`,
installed separately). Output feeds `fig_nc/figR1-4.py` and `fig_nc/fig6.py`.

## Known gaps (pre-existing, not introduced by splitting this repo out)

- `fig_nc/figR4-1.py` reads real-connectome adjacency matrices from `data/connnet/*.npy`
  (N=178/152/609 physical networks) — no generator script for these exists anywhere in this
  repo's history; they're externally-sourced/hand-placed data.
- `smite` / `glmcc` (used by `STE.py` / `GLMCC.py`) aren't packaged anywhere in this repo or
  pinned in `requirements.txt` — install them manually per their own upstream instructions.
