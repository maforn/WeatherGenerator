# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utilities for measuring training throughput metrics."""

import logging
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import psutil
import torch

logger = logging.getLogger(__name__)

_GIB = 1024**3


class ThroughputTracker:
    """Tracks training throughput metrics.

    Accumulates per-batch sample and source-byte counts across ranks.

    Note the counts include the first steps of a run, which carry one-off startup
    costs (kernel autotuning, allocator growth, NCCL buffer setup, cold page
    cache). Since the counters are cumulative and never reset, only the first
    logging window is affected; discard it when reading throughput off a short
    run rather than filtering here, to keep the per-step path branch-free.
    """

    def __init__(
        self,
        device: torch.device,
        batch_size_per_gpu: int,
    ) -> None:
        self._device = device
        self.batch_size_per_gpu = batch_size_per_gpu
        self._total_batches: int = 0
        self._total_samples: int = 0
        self._total_mb: float = 0.0

    def step(self, batch) -> None:
        """Accumulate one training step's counts. No synchronization or collectives.

        Call on every step from the training loop. Metrics are computed separately
        via ``compute_metrics`` at the logging interval, so the hot path stays free
        of device syncs and cross-rank collectives.

        Args:
            batch: The current training batch (must expose ``get_source_samples()``).
        """
        source_mb = compute_source_bytes(batch.get_source_samples()) / 1e6
        self.update(source_mb)

    def update(self, source_mb: float) -> None:
        """Record one training step.

        Purely local bookkeeping: no device synchronization. The cumulative
        counts are turned into throughput (and reduced across ranks) only when
        ``compute_metrics`` runs.

        Args:
            source_mb: Source tensor megabytes for this batch. Should be computed
                       fresh each step via ``compute_source_bytes`` as batch sizes
                       can vary across samples.
        """
        self._total_batches += 1
        self._total_samples += self.batch_size_per_gpu
        self._total_mb += source_mb

    def compute_metrics(self) -> dict[str, float]:
        """Return throughput metrics, empty if no step has been recorded yet.

        Collective: performs a single SUM all-reduce of the per-device throughput
        to obtain the global throughput, so it must be called on every rank at the
        same point in the training loop. The returned dict is identical on all
        ranks. Global throughput is the sum of the per-device rates across ranks.

        Note the empty-dict early return is not collective, but it is taken on
        every rank simultaneously: it depends only on the step count, and all
        ranks run the training loop in lockstep.

        Returns:
            Dict of ``"performance.<key>": value`` pairs; empty if no step has
            been recorded.
        """
        if self._total_batches == 0:
            return {}

        device_batches = self._total_batches
        device_samples = self._total_samples
        device_mb = self._total_mb

        # Global throughput: sum the per-device rates across ranks with a single
        # reduce, done here (once per logging interval) rather than per step.
        global_batches, global_samples, global_mb = device_batches, device_samples, device_mb
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rates = torch.tensor(
                [device_batches, device_samples, device_mb],
                dtype=torch.float64,
                device=self._device,
            )
            torch.distributed.all_reduce(rates, op=torch.distributed.ReduceOp.SUM)
            global_batches, global_samples, global_mb = rates.tolist()

        return {
            "performance.throughput.global.batches": global_batches,
            "performance.throughput.global.samples": global_samples,
            "performance.throughput.global.mb": global_mb,
        }


class MemoryTracker:
    """Tracks the global (max across all ranks) peak GPU memory per window.

    Reads the CUDA caching allocator's high-water marks (``max_memory_allocated``
    and ``max_memory_reserved``) at each ``collect()`` call, reduces them across
    all ranks with MAX, and resets the peak stats so every call reports the peak
    since the previous one.

    On top of the per-window peak, ``collect()`` also reports running maxima over
    all windows closed so far: ``<window>_global`` (the high-water mark of that
    window label alone, e.g. the worst training interval up to now) and ``global``
    (the run-wide high-water mark across every window label). Both are accumulated
    from the already-reduced values, so they are identical on all ranks. They are
    not derivable downstream from a single record — the windows are logged into
    separate records — which is why they are emitted here.

    ``max_memory_allocated`` is the peak of memory occupied by live tensors
    (parameters, gradients, optimizer states, activations) — the model's actual
    demand, comparable to analytic memory estimates. ``max_memory_reserved`` is
    the peak of memory the caching allocator has claimed from the device via
    cudaMalloc; it also counts cached blocks that are currently free but not
    returned to CUDA, so reserved >= allocated and the gap measures
    fragmentation / caching slack. An OOM is raised when a new cudaMalloc
    fails, so reserved (plus non-PyTorch usage such as NCCL buffers and cuBLAS
    workspaces, which neither counter sees) is what determines headroom against
    the device's capacity.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        # running (allocated, reserved) high-water marks, per window label and run-wide
        self._window_peaks: dict[str, tuple[int, int]] = {}
        self._run_peak: tuple[int, int] = (0, 0)
        # start with a clean window so the first step reports its own peak
        torch.cuda.reset_peak_memory_stats(device)

    def collect(self, window: str = "step") -> dict[str, float]:
        """Return peak-memory metrics for the window since the last call.

        Collective: must be called on every rank at the same point in the
        training loop (the cross-rank MAX reduction and the peak-stat reset run
        on every rank). The returned dict is identical on all ranks; callers
        merge it into whatever metrics record they log on the root rank.

        Args:
            window: Label naming what the window since the last call covers
                    (e.g. ``"train"``, ``"save_model"``, ``"validation"``); becomes
                    part of the metric key.

        Returns:
            Dict of ``"performance.memory.<label>.max_{allocated,reserved}_gib"``
            pairs, for ``<label>`` in ``<window>`` (this window), ``<window>_global``
            (running max of this window label) and ``global`` (run-wide running max).
        """
        max_allocated = torch.cuda.max_memory_allocated(self._device)
        max_reserved = torch.cuda.max_memory_reserved(self._device)
        torch.cuda.reset_peak_memory_stats(self._device)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            peaks = torch.tensor(
                [max_allocated, max_reserved], dtype=torch.int64, device=self._device
            )
            torch.distributed.all_reduce(peaks, op=torch.distributed.ReduceOp.MAX)
            max_allocated, max_reserved = peaks.tolist()

        # Accumulate after the reduction, so the running maxima stay identical on
        # all ranks without any further collective.
        window_peak = self._window_peaks.get(window, (0, 0))
        window_peak = (max(window_peak[0], max_allocated), max(window_peak[1], max_reserved))
        self._window_peaks[window] = window_peak
        self._run_peak = (
            max(self._run_peak[0], max_allocated),
            max(self._run_peak[1], max_reserved),
        )

        return {
            f"performance.memory.{window}.max_allocated_gib": max_allocated / _GIB,
            f"performance.memory.{window}.max_reserved_gib": max_reserved / _GIB,
            f"performance.memory.{window}_global.max_allocated_gib": window_peak[0] / _GIB,
            f"performance.memory.{window}_global.max_reserved_gib": window_peak[1] / _GIB,
            "performance.memory.global.max_allocated_gib": self._run_peak[0] / _GIB,
            "performance.memory.global.max_reserved_gib": self._run_peak[1] / _GIB,
        }


def _gather(values: list[float], device: torch.device) -> list[list[float]]:
    """``values`` of every rank, in rank order. Collective when distributed."""
    local = torch.tensor(values, dtype=torch.float64, device=device)
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return [local.tolist()]
    gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, local)
    return [g.tolist() for g in gathered]


class DataTimingTracker:
    """Tracks wall time per training step, split into data-loading stages.

    Training-loop stages: ``wait`` (for the next batch from the DataLoader), ``queued``
    (from the worker finishing the batch to the training loop receiving it: time spent
    waiting in the queue plus the hand-off), ``pin`` (pinning it in host memory), ``h2d``
    (copy to the device) and ``step`` (the whole step, including ``pin`` and ``h2d``).
    Worker stages arrive with each batch in ``batch.timings``: ``read`` (dataset readers),
    ``preprocess`` (shuffle, NaN removal, normalization), ``tokenize``, ``assemble`` (the
    rest of building the batch) and ``batch`` (all of it, including retries of empty
    samples). Workers prefetch, so worker time only slows training when it shows up as
    ``wait``.

    ``h2d`` and ``step`` synchronize the device to time the actual copy and compute;
    this removes some CPU/GPU overlap, so enable it for measurement runs only.
    """

    STAGES = (
        "wait",
        "queued",
        "pin",
        "h2d",
        "step",
        "read",
        "preprocess",
        "tokenize",
        "assemble",
        "batch",
    )

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._sums = dict.fromkeys(self.STAGES, 0.0)
        self._steps = 0
        self._t_step = 0.0

    def add(self, stage: str, seconds: float) -> None:
        self._sums[stage] += seconds

    def received(self, batch, wait: float) -> None:
        """Record a batch arriving in the training loop after waiting ``wait`` seconds."""
        self.add("wait", wait)
        # wall clock: the batch was stamped in another process on the same node
        self.add("queued", time.time() - getattr(batch, "t_ready", time.time()))

    @contextmanager
    def timed(self, stage: str):
        """Time a block, synchronizing the device around it."""
        torch.cuda.synchronize(self._device)
        t = time.perf_counter()
        yield
        torch.cuda.synchronize(self._device)
        self.add(stage, time.perf_counter() - t)

    def start_step(self) -> None:
        self._t_step = time.perf_counter()

    def end_step(self, batch) -> None:
        """Close the step started by ``start_step`` and add the batch's worker timings."""
        torch.cuda.synchronize(self._device)
        self.add("step", time.perf_counter() - self._t_step)
        for stage, seconds in getattr(batch, "timings", {}).items():
            self.add(stage, seconds)
        self._steps += 1

    def collect(self) -> dict[str, float]:
        """Mean seconds per step for each stage since the last call.

        Returns each rank's own values (``performance.data.rank<r>.<stage>_s``, which add
        up within a rank) and the max over ranks (``performance.data.<stage>_s``, taken per
        stage, so these can come from different ranks).

        Collective: call on every rank at the same point in the training loop.
        """
        if self._steps == 0:
            return {}
        per_rank = _gather([self._sums[s] / self._steps for s in self.STAGES], self._device)
        self._sums, self._steps = dict.fromkeys(self.STAGES, 0.0), 0
        metrics = {
            f"performance.data.{s}_s": max(rank[i] for rank in per_rank)
            for i, s in enumerate(self.STAGES)
        }
        for r, values in enumerate(per_rank):
            metrics |= {
                f"performance.data.rank{r}.{s}_s": v
                for s, v in zip(self.STAGES, values, strict=True)
            }
        return metrics


class NullDataTimingTracker:
    """No-op data timing tracker used when data timing is disabled."""

    def add(self, stage: str, seconds: float) -> None:
        pass

    def received(self, batch, wait: float) -> None:
        pass

    def timed(self, stage: str):
        return nullcontext()

    def start_step(self) -> None:
        pass

    def end_step(self, batch) -> None:
        pass

    def collect(self) -> dict[str, float]:
        return {}


class HostMemoryTracker:
    """Host memory of this rank's processes, sampled at each ``collect()``.

    ``main_pss`` is the training process and ``workers_pss`` its child processes (the
    DataLoader workers), as proportional set sizes, so pages shared after the fork are
    split between processes instead of counted in each. ``cgroup_anon`` and
    ``cgroup_shmem`` come from the cgroup the rank runs in (cgroup v2). Slurm on JUPITER
    uses one cgroup per job step and node, so these are totals for all ranks on the
    rank's node, not per rank. ``cgroup_shmem`` holds the batches that workers hand to
    the main process through /dev/shm. Reading the memory maps of every process takes
    time, so enable it for measurement runs only.
    """

    KEYS = ("main_pss", "workers_pss", "cgroup_anon", "cgroup_shmem")

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._process = psutil.Process()
        # cgroup v2 has one line "0::<path>"
        cgroup = Path("/proc/self/cgroup").read_text().split("::", 1)[-1].strip()
        self._stat = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "memory.stat"

    @staticmethod
    def _pss(process: psutil.Process) -> int:
        try:
            return process.memory_full_info().pss
        except psutil.Error:  # e.g. a worker that exited in the meantime
            return 0

    def collect(self) -> dict[str, float]:
        """Current host memory of every rank, ``performance.host_memory.rank<r>.<key>_gib``.

        Collective: call on every rank at the same point in the training loop.
        """
        workers = sum(self._pss(p) for p in self._process.children(recursive=True))
        stat = {}
        if self._stat.exists():
            stat = dict(line.split() for line in self._stat.read_text().splitlines())
        values = [self._pss(self._process), workers]
        values += [int(stat.get(k, 0)) for k in ("anon", "shmem")]
        metrics = {}
        for r, rank in enumerate(_gather([v / _GIB for v in values], self._device)):
            metrics |= {
                f"performance.host_memory.rank{r}.{k}_gib": v
                for k, v in zip(self.KEYS, rank, strict=True)
            }
        return metrics


class NullHostMemoryTracker:
    """No-op host memory tracker used when host memory tracking is disabled."""

    def collect(self) -> dict[str, float]:
        return {}


class NullMemoryTracker:
    """No-op memory tracker used when memory tracking is disabled.

    Implements the same interface as ``MemoryTracker`` so call sites in the
    training loop need no ``if`` guards.
    """

    def collect(self, window: str = "step") -> dict[str, float]:
        return {}


class NullThroughputTracker:
    """No-op throughput tracker used when performance tracking is disabled.

    Implements the same interface as ``ThroughputTracker`` so call sites in the
    training loop need no ``if`` guards.
    """

    def step(self, batch) -> None:
        pass

    def compute_metrics(self) -> dict[str, float]:
        return {}


def compute_source_bytes(source_samples) -> int:
    """Count total bytes of all source token tensors in a batch.

    Args:
        source_samples: Result of sample_batch.get_source_samples(), containing
                        a list of samples each with per-stream source token cells.

    Returns:
        Total byte count across all streams and cells in the batch.
    """
    total = 0
    for sample in source_samples.samples:
        for stream_data in sample.streams_data.values():
            for t in stream_data.source_tokens_cells:
                total += t.nbytes
    return total


@contextmanager
def nvtx_range(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _nvtx_push(name: str):
    torch.cuda.nvtx.range_push(name)


def _nvtx_pop():
    torch.cuda.nvtx.range_pop()


def register_nvtx_hooks(model, scope: str = "global"):
    torch.nn.modules.module.register_module_forward_pre_hook(
        lambda m, args: _nvtx_push(f"{m.__class__.__name__}.forward")
    )
    torch.nn.modules.module.register_module_forward_hook(
        lambda m, input, output: _nvtx_pop(), always_call=True
    )
    torch.nn.modules.module.register_module_full_backward_pre_hook(
        lambda m, args: _nvtx_push(f"{m.__class__.__name__}.backward")
    )
    torch.nn.modules.module.register_module_full_backward_hook(lambda m, input, output: _nvtx_pop())
