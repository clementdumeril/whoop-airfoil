#!/bin/bash
# Training with the L2F setup (SAC by default, L2F intervals and batch, 64 x 64), 10 seeds of each variant:
#   tasks 0-9:   hover, no plate (reproduces the L2F task on the whoop)
#   tasks 10-19: hover, plate
#   tasks 20-29: velocity task (1-4 m/s), no plate
#   tasks 30-39: velocity task, plate
# Each run writes runs/<variant>_<job>_seed<seed>/ (config.json, progress.csv, evaluation_progress.csv,
# evaluation.json, flights.csv, runner_checks.json, actor.h). A 300k-step run took about 1-1.5 min on the laptop.
# Submit from whoop-airfoil/sim after slurm/build.slurm:
#   sbatch slurm/train.slurm                                  (SAC, 300k steps: the L2F paper)
#   sbatch --export=ALL,STEPS=1000000 slurm/train.slurm       (longer: more curriculum stages)
#   sbatch --export=ALL,ALGORITHM=td3 slurm/train.slurm       (TD3, the algorithm of the paper)
#SBATCH --job-name=whoop-train
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-39
#SBATCH --output=runs/slurm-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MKL_NUM_THREADS=1
ALGORITHM=${ALGORITHM:-sac}
STEPS=${STEPS:-300000}
SEED=$((SLURM_ARRAY_TASK_ID % 10))
case $((SLURM_ARRAY_TASK_ID / 10)) in
    0) VARIANT=hover_noplate ;;
    1) VARIANT=hover_plate ;;
    2) VARIANT=velocity_noplate ;;
    3) VARIANT=velocity_plate ;;
esac
build/whoop_train_${ALGORITHM}_${VARIANT} "runs/${ALGORITHM}_${VARIANT}_${STEPS}_${SLURM_ARRAY_JOB_ID}_seed${SEED}" "$SEED" "$STEPS"
