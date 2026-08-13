#!/bin/bash
# Run from the repo root: ./slurms/batch_bm_PDIF.sh
# logs/ and results/ are written relative to the repo root (both this script's
# sbatch submissions and the .slurm jobs' `python scripts/benchmark/...` calls
# assume cwd == repo root, which SLURM inherits from wherever sbatch was run).
mkdir -p logs

# REGEN=1 ./slurms/batch_bm_PDIF.sh recomputes everything: jobs whose .pkl
# already exists are submitted again, and each re-runs its causality estimation
# instead of reusing the one on disk. Default (0) resumes where the sweep left
# off, which is also the only mode that reports timing from the original run.
REGEN=${REGEN:-0}

keys=(HHEE HHEI HHconEE HHconEI Lorenz Logistic Rcon RNN)
# Ts=(1e7 1e7 1e7 1e7 1e6 1e8 1e7 1e8) full data length
# Control study: every method sees the SAME T per dataset -- keep this array identical to the Ts
# array in batch_bm_DDC.sh / batch_bm_STE.sh / batch_bm_GLMCC.sh / batch_bm_CCM.sh.
# HHEE/HHEI: 2e5 (2% of T_Max, matches CCM.py's own L_dict default span for those two datasets).
# All others: 1% of T_Max.
Ts=(2e5 2e5 1e5 1e5 1e4 1e6 1e5 1e6)
for i in "${!keys[@]}"
do
    for idx in {0..9}
    do
        for noise_level in 0 0.1 0.2 0.3 0.4
        do
            # Skip if the output .pkl this job would produce already exists.
            [ "$noise_level" = "0" ] && noise_tag="0" || noise_tag=$(printf "%.1f" "$noise_level")
            T_tag=$(printf "%.2e" "${Ts[i]}")
            outfile="results/PDIF/recon_df_noise_${noise_tag}_${keys[i]}_T=${T_tag}_${idx}.pkl"
            if [ "$REGEN" != "1" ] && [ -f "$outfile" ]; then
                echo "Skip (exists): $outfile"
                continue
            fi
            sbatch --export=ALL,KEY=${keys[i]},IDX=${idx},NOISE_LEVEL=${noise_level},T=${Ts[i]},REGEN=${REGEN} slurms/bm_PDIF.slurm
        done
    done
done