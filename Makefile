# Run WeatherGenerator on JSC JUPITER (see docs/jupiter.md).
# All targets run on a JUPITER login node, directly or through `minion run <target>`.

SHELL := /bin/bash

# BASE replaces config/default_config.yml (merging on top of it changes e.g. memory_pinning).
# CONFIG: overlays, comma-separated, e.g. CONFIG=config/dphpc_memory_baseline_100steps.yml,config/optimizations.yml
BASE    ?= config/config_operan_georing_avhrr_forecasting_lowres.yml
CONFIG  ?= config/dphpc_memory_baseline_100steps.yml
NODES   ?= 1
TIME    ?= 00:30:00
# config overrides, comma-separated: OPTIONS=embed_num_chunks=2,decoder_num_chunks=4
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

.PHONY: help init check test train profile snapshot scalene-data scalene-train read-stress sp-loader

help: ## show this help
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*##"}; {printf "  %-14s %s\n", $$1, $$2}'

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

snapshot: ## like train, but 3 steps with a CUDA allocator snapshot (scripts/analyze_memory_snapshot.py)
	$(MAKE) train CONFIG="$(CONFIG)$(comma)config/config_memory_profiling.yml"

# Scalene: line-level CPU (Python and native), memory and GPU profiles, run on a compute node.
# View: `.venv/bin/python -m scalene view --cli -r <json>`, or `--standalone` for one HTML file.
SCALENE := uv pip install --python .venv/bin/python -q scalene &&
SCALENE_DIR := profiling/scalene
BATCHES ?= 3
STEPS ?= 30
SCALENE_CONFIGS := --base-config $(BASE) --config $(subst $(comma), ,$(CONFIG))

scalene-data: ## Scalene CPU + memory profile of building BATCHES batches (no workers, no GPU)
	mkdir -p $(SCALENE_DIR)
	$(MODULES) $(SCALENE) srun -A $(BUDGET_ACCOUNTS) -p booster -N 1 -n 1 --time 00:40:00 \
		.venv/bin/python -m scalene run --memory --profile-only weathergen --program-path . \
		-o $(SCALENE_DIR)/data.json scripts/profile_data_loading.py --- $(SCALENE_CONFIGS) \
		--batches $(BATCHES) --checksum $(if $(OPTIONS),--options $(subst $(comma), ,$(OPTIONS)))

# Storage stress of the read stage: PROCS concurrent reader processes, READ_SECONDS each, on all
# 288 CPUs of one node and no GPU (scripts/read_stress.py). Results: READ_DIR/summary.json.
READ_DIR ?= profiling/read_stress
PROCS ?= 1 16 64 128 192 256
READ_SECONDS ?= 90
read-stress: ## read-only storage stress test: PROCS concurrent readers, READ_SECONDS per level
	mkdir -p $(READ_DIR)
	$(MODULES) srun -A $(BUDGET_ACCOUNTS) -p booster -N 1 -n 1 --cpus-per-task=288 --time 00:30:00 \
		.venv/bin/python -u scripts/read_stress.py $(SCALENE_CONFIGS) --out $(READ_DIR) \
		--procs $(PROCS) --seconds $(READ_SECONDS) \
		$(if $(OPTIONS),--options $(subst $(comma), ,$(OPTIONS)))

# Loader work per sample for each rank of a spatial group vs data parallel
# (scripts/sp_loader_cost.py): per level in SP_LEVELS, 1 data-parallel process and SP_SIZE
# spatial ranks, all at once on one node, no GPU. Results: SP_DIR/summary.json.
SP_DIR ?= profiling/sp_loader
SP_LEVELS ?= 5 7
SP_SIZE ?= 4
SP_BATCHES ?= 3
sp-loader: ## loader stage times per sample, spatial ranks vs data parallel, levels SP_LEVELS
	mkdir -p $(SP_DIR)
	$(MODULES) srun -A $(BUDGET_ACCOUNTS) -p booster -N 1 -n 1 --cpus-per-task=288 --time 00:40:00 \
		.venv/bin/python -u scripts/sp_loader_cost.py $(SCALENE_CONFIGS) --out $(SP_DIR) \
		--levels $(SP_LEVELS) --size $(SP_SIZE) --batches $(SP_BATCHES) \
		$(if $(OPTIONS),--options $(subst $(comma), ,$(OPTIONS)))

scalene-train: ## Scalene CPU + GPU profile of rank 0 of a 1-node training of STEPS steps
	mkdir -p $(SCALENE_DIR)
	$(MODULES) $(SCALENE) srun -A $(BUDGET_ACCOUNTS) -p booster -N 1 -n 4 --cpus-per-task=72 \
		--gres=gpu:4 --time $(TIME) scripts/scalene_task.sh $(SCALENE_DIR)/train.json train \
		$(SCALENE_CONFIGS) --options training_config.samples_per_mini_epoch=$(shell echo $$(($(STEPS) * 4))) \
		$(subst $(comma), ,$(OPTIONS))
