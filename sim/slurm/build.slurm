#!/bin/bash
# Compile every target on a compute node (-march=native matches the nodes of the training jobs) and run the unit tests.
# Submit from whoop-airfoil/sim after slurm/setup.sh:  sbatch slurm/build.slurm
#SBATCH --job-name=whoop-build
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm-build-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 cmake/3.27.7 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
cmake --build build -j "$SLURM_CPUS_PER_TASK"
build/whoop_test
