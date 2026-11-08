#!/bin/bash
# One task of a 1-node training started with srun from this folder's .venv (make train-direct):
# the environment of the JUPITER job script, without the new venv that launch-slurm.py builds
# for every job (which needs GitHub).
#   srun -n 4 ... scripts/train_task.sh train --run-id <id> --base-config ... --config ...
ulimit -c 0
echo "rank $SLURM_PROCID: $(nproc) CPUs"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# compile_blocks: keep the compiled kernels across jobs
export TORCHINDUCTOR_CACHE_DIR="/e/scratch/weatherai/$USER/inductor-cache"
export UCX_TLS="^cma" UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
# NSYS=<name>: Nsight Systems trace of each rank, from NSYS_DELAY s after the start for
# NSYS_DURATION s (the job then ends), to profiling/nsys/<name>_<rank>.nsys-rep
if [ -n "${NSYS:-}" ]; then
    mkdir -p profiling/nsys
    exec nsys profile -w true --trace=cuda,nvtx,osrt --force-overwrite=true \
        --delay="${NSYS_DELAY:-120}" --duration="${NSYS_DURATION:-120}" \
        -o "profiling/nsys/${NSYS}_${SLURM_PROCID}" .venv/bin/python -u src/weathergen/run_train.py "$@"
fi
exec .venv/bin/python -u src/weathergen/run_train.py "$@"
