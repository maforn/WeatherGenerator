#!/bin/bash
# One task of a 1-node training run started with srun: rank 0 runs under scalene (time per
# line), the other ranks run normally. Environment as in the JUPITER job script.
#   srun -n 4 ... scripts/scalene_task.sh <profile.json> train --base-config ... --config ...
# CPU time only by default. SCALENE_FLAGS=--gpu adds scalene's torch.profiler (GPU time per
# line), started after SCALENE_TORCH_PROFILER_DELAY seconds (scripts/scalene_launch.py) since
# it crashes the model setup; but in the 4-GPU training it then still corrupts the heap (with
# fsdp_cpu_offload) or hangs rank 0, so use Nsight Systems (make profile) for the GPU side.
out=$1
shift
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UCX_TLS="^cma" UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
export SCALENE_TORCH_PROFILER_DELAY=${SCALENE_TORCH_PROFILER_DELAY:-180}
if [ "$SLURM_PROCID" = 0 ]; then
    exec .venv/bin/python scripts/scalene_launch.py run ${SCALENE_FLAGS:---cpu-only} \
        --profile-only weathergen --program-path . -o "$out" src/weathergen/run_train.py --- "$@"
fi
exec .venv/bin/python src/weathergen/run_train.py "$@"
