#!/bin/bash

#SBATCH --job-name="BM-PDIF"
#SBATCH --time=168:00:00        # task walltime
#SBATCH --nodes 1              # number of tasks
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH --cpus-per-task=40
#SBATCH -p cpu
#SBATCH --exclusive
#SBATCH --output logs/log_%J.log
#SBATCH --error logs/err_%J.log
#SBATCH --mail-type=end
#SBATCH --mail-user=kchen513@sjtu.edu.cn

# REGEN=1 forces the causality estimation to be recomputed; unset or 0 reuses
# an existing one (the default).
REGEN_FLAG=""
if [ "${REGEN:-0}" = "1" ]; then
    REGEN_FLAG="--regen"
fi

python scripts/benchmark/PDIF.py --key $KEY --idx $IDX --noise_level $NOISE_LEVEL --T $T $REGEN_FLAG