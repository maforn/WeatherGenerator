# Running on JSC JUPITER

The real commands live in the `Makefile` and run on a JUPITER login node.
[minion](https://github.com/andres-nav/minion) (`.minion.json`) syncs your local changes and runs those targets from your laptop.

## One-time setup

1. In JuDoor, join the projects `weatherai`, `slmet` (data) and `e-ext-2025e01-128` (compute), and set up MFA.
2. Create a key (`ssh-keygen -t ed25519 -a 100 -f ~/.ssh/jsc_ed25519`) and add it in JuDoor for `jupiter` and `judac` with a `from=` clause that covers the address JSC sees, for example:

   ```
   from="195.176.96.0/19,129.132.0.0/16,2001:67c:10ec::/48" ssh-ed25519 AAAA...
   ```

3. Add this to `~/.ssh/config`, above any `Host *` block (SSH keeps the first value it finds, so a later `ControlPersist` would be ignored).
   Keep `ProxyJump` only if you are outside the ETH network without VPN.

   ```
   Host jupiter judac
       User YOUR_JSC_USER
       IdentityFile ~/.ssh/jsc_ed25519
       IdentitiesOnly yes
       ProxyJump jumphost.inf.ethz.ch
       ControlMaster auto
       ControlPath ~/.ssh/control-%C
       ControlPersist 8h

   Host jupiter
       HostName login.jupiter.fz-juelich.de

   Host judac
       HostName judac.fz-juelich.de
   ```

4. Clone `WeatherGenerator-private` next to this repo, then log in once and initialize:

   ```
  ssh jupiter exit    # MFA; the session is reused for 8 h
   minion run init    # syncs both repos, installs uv, creates .venv
   minion run check
   ```

Both repos are synced to `/e/scratch/weatherai/$USER/` and uv goes to `~/.local/bin`.
Scratch is fine for these: the laptop is the source of truth and everything there can be rebuilt.

## Daily use

Log in once with `ssh jupiter` (MFA), then:

```
minion run train                                    # baseline: BASE + CONFIG from the Makefile, 1 node, 30 min
minion run train --arg "CONFIG=config/my_test.yml"  # your own overlay on the same base
minion run train --arg "NODES=2 TIME=01:00:00"
minion run profile                                  # same, with Nsight Systems profiling
minion run timing                                   # same, with per-stage data-loading timers
minion run timing --arg "OPTIONS=data_loading.num_workers=6"   # config overrides, comma-separated
minion run test                                     # unit tests on the login node
minion run queue
```

`timing` logs, per step and per rank (see `src/weathergen/utils/performance.py`):
- `performance.data.rank<r>.<stage>_s` (`DataTimingTracker`): `wait`, `queued`, `pin`, `h2d` and `step` in the training loop,
  and `read`, `preprocess`, `tokenize`, `assemble` and `batch` per worker batch; `performance.data.<stage>_s` is the max over ranks.
- `performance.host_memory.rank<r>.<key>_gib` (`HostMemoryTracker`): `main_pss`, `workers_pss`, and the cgroup's
  `cgroup_anon` and `cgroup_shmem`, which on JUPITER are totals for the node, not per rank.

It syncs the GPU around `h2d` and `step` and reads every process's memory maps, so use it for measurement runs only.
`minion run profile-data` profiles the data loading alone on a compute node (cProfile, plus the worker-to-main hand-off).
`profile` writes one Nsight trace per GPU to `shared_work/profiling/<run_id>/`; Nsight stops the program after 240 s.

`minion run train` prints the run id. Outputs are in `/e/scratch/weatherai/shared_work/`:
`logs/<run_id>/output.<job>.txt` (training log) and `results/<run_id>/<run_id>_train_metrics.json`
(one record per step: losses, `performance.memory.*`, throughput).
After jobs finish, `experiments/save.py` copies every new run into [experiments](../experiments/) and updates its table
(`experiments/save.py <run_id> "note"` adds a note).

Without minion, run the same targets on JUPITER: `cd /e/scratch/weatherai/$USER/WeatherGenerator && make train`.

## GPU memory options

All off by default; turn one on with `minion run train --arg "OPTIONS=<option>=True"`.
`minion run snapshot` records a CUDA allocator snapshot (3 steps); `scripts/analyze_memory_snapshot.py` breaks it down.

| Option | What it changes | Results |
|---|---|---|
| `fe_checkpoint_per_block` | forecast engine: one checkpoint per attention block and the layers after it | same math |
| `decoder_gather_after_kv_projection` | decoder: project keys/values of the cells' tokens, then select the 9-cell neighbourhoods | same math |
| `ae_local_checkpoint_per_block` | local assimilation: one checkpoint per attention + MLP pair | same math |
| `pred_checkpoint_per_layer` | decoder: one checkpoint per cross-attention and the blocks after it (raises the peak on the operan config) | same math |
| `source_tokens_in_mixed_precision` | source tokens kept on the GPU in the mixed-precision dtype (the embedding casts to it anyway) | same |
| `fe_checkpoint_bf16` | forecast engine: checkpoint inputs saved in bf16 | rounds the residual stream |

## Notes

- Python comes from `Stages/2025` (3.12.3), loaded by the `Makefile`. Running `module --force purge` before loading it breaks the load.
- Pass the operan config as `--base-config` (the `Makefile` does). As `--config` it is merged on top of `default_config.yml`, which turns on `memory_pinning` among others; that run crashed with GPU and host out-of-memory errors.
- Slurm jobs read `BUDGET_ACCOUNTS` from `~/.bashrc` on JUPITER; `make init` adds it.
- `launch-slurm.py` copies only the files listed by `git ls-files` into the job directory, so `make train` first indexes the synced trees (`git init && git add -A`; nothing is committed on JUPITER).
  This skips files that match `.gitignore` even if they are tracked, which is why `config/**/*.json` is excluded from the `*.json` rule (`config/profiling/annotations.json` is needed for profiling).
- Files on `$SCRATCH` are deleted after 90 days without use.
