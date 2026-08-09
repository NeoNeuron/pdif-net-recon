#!/bin/bash

#SBATCH --job-name="BM-generate-noisy-data"
#SBATCH --time=168:00:00        # task walltime
#SBATCH --nodes 1              # number of tasks
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH --cpus-per-task=10
#SBATCH -p cpu
#SBATCH --output logs/log_%J.log
#SBATCH --error logs/err_%J.log
#SBATCH --mail-type=end
#SBATCH --mail-user=kchen513@sjtu.edu.cn

python scripts/benchmark/measurement_noise.py