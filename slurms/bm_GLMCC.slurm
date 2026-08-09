#!/bin/bash

#SBATCH --job-name="BM-GLMCC"
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

python scripts/benchmark/GLMCC.py --key $KEY --idx $IDX --noise_level $NOISE_LEVEL --T $T