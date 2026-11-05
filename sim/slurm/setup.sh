#!/bin/bash
# Configure the simulator build on the Savio login node (light: compiles nothing). Then sbatch slurm/build.slurm, then
# sbatch slurm/train.slurm. Run from whoop-airfoil/sim:  source slurm/setup.sh
#
# RLtools: header-only, commit b32d9985 (the one tested here). Point RL_TOOLS_DIR to a checkout of that commit, for
# example the copy the Fixed-Wing build fetched (Fixed-Wing/cpp/build/_deps/rl_tools-src, same commit). It is only read.
module load gcc/11.4.0 cmake/3.27.7 python/3.11.6-gcc-11.4.0
source ~/.venvs/fixed-wing/bin/activate   # the Fixed-Wing venv: Python for the parameter header, Intel MKL
RL_TOOLS_DIR=${RL_TOOLS_DIR:-$HOME/Fixed-Wing/cpp/build/_deps/rl_tools-src}
if [ ! -f "$RL_TOOLS_DIR/include/rl_tools/version.h" ]; then
    echo "RLtools not found in $RL_TOOLS_DIR: export RL_TOOLS_DIR=<checkout of rl-tools at b32d9985> and source again"
    return 1 2>/dev/null || exit 1
fi
if [ "$(git -C "$RL_TOOLS_DIR" rev-parse --short=8 HEAD 2>/dev/null)" != "b32d9985" ]; then
    echo "warning: $RL_TOOLS_DIR is not at commit b32d9985 (the tested one)"
fi
MKL_CONFIG=$(find "$VIRTUAL_ENV" -name MKLConfig.cmake | head -n 1)
if [ -n "$MKL_CONFIG" ]; then
    MKL_OPTIONS="-DWHOOP_MKL=ON -DMKL_DIR=$(dirname "$MKL_CONFIG") -DMKL_ROOT=$VIRTUAL_ENV"
else
    echo "no MKL in $VIRTUAL_ENV (pip install mkl-devel): building with RLtools' own matrix code"
    MKL_OPTIONS=""
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_DIR="$RL_TOOLS_DIR" -DPython3_EXECUTABLE="$(which python)" $MKL_OPTIONS
mkdir -p runs
