# Run WeatherGenerator on JSC JUPITER (see docs/jupiter.md).
# All targets run on a JUPITER login node, directly or through `minion run <target>`.

SHELL := /bin/bash

# BASE replaces config/default_config.yml (merging on top of it changes e.g. memory_pinning).
# CONFIG: overlays, comma-separated; dphpc_measure.yml only shortens the run and adds logging.
BASE    ?= config/config_operan_georing_avhrr_forecasting_lowres.yml
CONFIG  ?= config/dphpc_measure.yml
NODES   ?= 1
TIME    ?= 00:30:00
# config overrides, comma-separated: OPTIONS=data_loading.num_workers=4,train_logging.data_timing=True
OPTIONS ?=
ARGS    ?=
PRIVATE := ../WeatherGenerator-private

export PATH            := $(HOME)/.local/bin:$(PATH)
export BUDGET_ACCOUNTS := $(or $(BUDGET_ACCOUNTS),e-ext-2025e01-128)
export UV_CACHE_DIR    := $(or $(UV_CACHE_DIR),/e/scratch/weatherai/$(USER)/uv-cache)
# build the venv on the module Python, not on a uv-downloaded one
export UV_PYTHON_PREFERENCE := only-system
# Python must be the Stages/2025 module (3.12.3); `module --force purge` before the load breaks it.
MODULES := { type module >/dev/null 2>&1 || . "$$LMOD_ROOT/lmod/init/bash"; } && \
	module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3 &&
comma := ,

.PHONY: help init check test train profile snapshot timing profile-data

help: ## show this help
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*##"}; {printf "  %-13s %s\n", $$1, $$2}'

# Slurm jobs only get the variables launch-slurm.py exports, and the job script reads
# BUDGET_ACCOUNTS from ~/.bashrc (platform-env.py fails without it).
init: ## install uv, set BUDGET_ACCOUNTS in ~/.bashrc and create .venv; safe to rerun
	[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
	grep -q BUDGET_ACCOUNTS ~/.bashrc || echo 'export BUDGET_ACCOUNTS=$(BUDGET_ACCOUNTS)' >> ~/.bashrc
	$(MODULES) ./scripts/actions.sh sync

check: ## show Python, uv, budget account and location
	$(MODULES) python3 --version && uv --version && echo "BUDGET_ACCOUNTS=$$BUDGET_ACCOUNTS" && pwd

test: ## run the unit tests on the login node, with the existing .venv
	$(MODULES) uv run --no-sync pytest -q src/

# launch-slurm.py copies only the files listed by `git ls-files`. The laptop is the source of
# truth, so indexing the synced trees is enough (nothing is committed here).
train: ## submit BASE + CONFIG (+ OPTIONS) on NODES nodes for TIME (extra launcher flags in ARGS)
	git init -q && git add -A
	git -C $(PRIVATE) init -q && git -C $(PRIVATE) add -A
	$(MODULES) $(PRIVATE)/hpc/launch-slurm.py --time $(TIME) --nodes $(NODES) --no-register \
		--base-config $(BASE) --config $(subst $(comma), ,$(CONFIG)) \
		$(if $(OPTIONS),--options $(subst $(comma), ,$(OPTIONS))) $(ARGS)

profile: ## like train, with Nsight Systems profiling
	$(MAKE) train ARGS="--nsys-profiling $(ARGS)"

snapshot: ## like train, but only 3 steps with a CUDA allocator snapshot (GPU memory breakdown)
	$(MAKE) train CONFIG="$(CONFIG)$(comma)config/dphpc_memory_snapshot.yml"

TIMING := train_logging.data_timing=True,train_logging.host_memory_tracking=True
timing: ## like train, with per-stage data-loading timers and per-process host memory
	$(MAKE) train OPTIONS="$(TIMING)$(if $(OPTIONS),$(comma)$(OPTIONS))"

BATCHES ?= 3
profile-data: ## cProfile BATCHES batches of BASE + CONFIG data loading on a compute node, no training
	$(MODULES) srun -A $(BUDGET_ACCOUNTS) -p booster -N 1 -n 1 --time 00:20:00 \
		.venv/bin/python scripts/profile_data_loading.py --base-config $(BASE) --config $(subst $(comma), ,$(CONFIG)) \
		--batches $(BATCHES) $(if $(OPTIONS),--options $(subst $(comma), ,$(OPTIONS)))
