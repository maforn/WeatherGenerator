"""Opt-in CPU loader timing and rank memory sampling; never imported by CUDA workers eagerly."""

import csv
import json
import logging
import os
import shutil
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def profile_output_dir(cf):
    """Resolve the shared run directory only when profiling is enabled."""
    configured = cf.data_loading.get("profile_output_dir")
    if configured:
        return Path(configured)
    from weathergen.common.config import get_path_run

    return get_path_run(cf) / "dataloader_profile"


def payload_bytes(obj, max_groups=12):
    """Bytes of torch tensors and numpy arrays reachable from ``obj``, grouped by attribute path.

    Torch tensors are what a DataLoader worker moves into shared memory (/dev/shm) when it
    hands a sample to the main process; numpy arrays are pickled through the pipe instead.
    Views of one storage are counted once. Groups use the first two path components.
    """
    seen_storage, seen_obj = set(), set()
    totals = {"torch_bytes": 0, "numpy_bytes": 0, "tensors": 0}
    groups = {}
    stack = [(obj, ())]
    while stack:
        item, path = stack.pop()
        if item is None or isinstance(item, (str, bytes, int, float, bool)):
            continue
        kind, nbytes = None, 0
        module = type(item).__module__
        if module.startswith("torch") and hasattr(item, "untyped_storage"):
            storage = item.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in seen_storage:
                seen_storage.add(key)
                kind, nbytes = "torch_bytes", storage.nbytes()
            else:
                continue
        elif module == "numpy" and hasattr(item, "nbytes") and hasattr(item, "dtype"):
            if id(item) in seen_obj:
                continue
            seen_obj.add(id(item))
            if item.dtype == object:
                stack.extend((x, path) for x in item.ravel())
                continue
            kind, nbytes = "numpy_bytes", int(item.nbytes)
        if kind:
            totals[kind] += nbytes
            totals["tensors"] += kind == "torch_bytes"
            group = ".".join(path[:2]) or "<root>"
            groups[group] = groups.get(group, 0) + nbytes
            continue
        if id(item) in seen_obj:
            continue
        seen_obj.add(id(item))
        if isinstance(item, dict):
            children = item.items()
        elif isinstance(item, (list, tuple, set, frozenset)):
            children = ((None, x) for x in item)
        elif hasattr(item, "__dict__") or hasattr(item, "__slots__"):
            names = list(getattr(item, "__dict__", {}))
            names += [n for n in getattr(type(item), "__slots__", ()) if hasattr(item, n)]
            children = ((n, getattr(item, n, None)) for n in names)
        else:
            continue
        for name, child in children:
            stack.append((child, path if name is None else path + (str(name),)))
    top = sorted(groups.items(), key=lambda kv: -kv[1])[:max_groups]
    totals["groups"] = dict(top)
    return totals


def shm_usage(path="/dev/shm"):
    """(used, total) bytes of the shared-memory filesystem, or (None, None) if absent."""
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None, None
    return usage.used, usage.total


class SampleProfiler:
    """One JSONL record per attempted sample, including samples rejected as invalid.

    Construct inside a worker, not before forking. File handles are opened only
    when writing a record, so repeated mini-epochs can append safely. Timings are
    wall-clock durations and RSS snapshots do not capture intermediate peaks.
    """

    def __init__(self, output_dir, rank, worker_id, stage, mini_epoch, sample_id):
        import psutil

        self.process = psutil.Process()
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.worker_id = int(worker_id)
        self.stage = str(stage)
        self.mini_epoch = int(mini_epoch)
        self.sample_id = str(sample_id)
        self.valid = None
        self.streams = {}

    def __enter__(self):
        self.started = time.perf_counter()
        self.timestamp = time.time()
        self.rss_before = self.process.memory_info().rss
        return self

    @contextmanager
    def measure(self, stream, operation):
        started = time.perf_counter()
        try:
            yield
        finally:
            values = self.streams.setdefault(str(stream), {})
            key = f"{operation}_seconds"
            values[key] = values.get(key, 0.0) + time.perf_counter() - started

    def add_stream(self, stream, source_bytes=0, source_tokens=0, target_tokens=0):
        values = self.streams.setdefault(str(stream), {})
        for key, value in (
            ("source_mb", float(source_bytes) / 1e6),
            ("source_tokens", int(source_tokens)),
            ("target_tokens", int(target_tokens)),
        ):
            values[key] = values.get(key, 0) + value

    def __exit__(self, exc_type, exc_value, traceback):
        record = {
            "schema_version": 1,
            "timestamp": self.timestamp,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "rank": self.rank,
            "worker_id": self.worker_id,
            "stage": self.stage,
            "mini_epoch": self.mini_epoch,
            "sample_id": self.sample_id,
            "valid": self.valid,
            "total_seconds": time.perf_counter() - self.started,
            "get_batch_seconds": getattr(self, "get_batch_seconds", None),
            "payload": getattr(self, "payload", None),
            "rss_before_bytes": self.rss_before,
            "rss_after_bytes": self.process.memory_info().rss,
            "streams": self.streams,
            "error": None if exc_type is None else exc_type.__name__,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"samples_rank{self.rank:04d}_worker{self.worker_id:03d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        return False


class MemoryTimeline:
    """Sample one rank and its loader workers without CUDA synchronization.

    Start after DataLoader iterator creation (fork), stop before the next fork.
    Pass a callable returning worker PIDs for exact attribution. Otherwise only
    descendants named ``pt_data_worker`` are counted, excluding unrelated child
    processes. Summed RSS double-counts shared memory and is not node physical RAM.
    CUDA is queried only if torch is already imported and CUDA already initialized.

    ``full_memory=True`` adds honest host totals: USS (pages only this process owns) and PSS
    (shared pages split between the processes mapping them) for the rank and its workers, read
    from ``/proc/<pid>/smaps_rollup`` via psutil, plus node-wide used memory
    (``MemTotal - MemAvailable``, which includes ``/dev/shm``). Columns are added only then.
    """

    fields = (
        "schema_version",
        "timestamp",
        "elapsed_seconds",
        "hostname",
        "rank",
        "pid",
        "stage",
        "mini_epoch",
        "step",
        "rank_rss_bytes",
        "worker_rss_bytes",
        "workers_rss_bytes",
        "total_rss_bytes",
        "gpu_allocated_bytes",
        "gpu_reserved_bytes",
        "shm_used_bytes",
        "shm_total_bytes",
    )
    full_memory_fields = (
        "rank_uss_bytes",
        "rank_pss_bytes",
        "workers_uss_bytes",
        "workers_pss_bytes",
        "node_used_bytes",
        "node_total_bytes",
    )

    def __init__(
        self,
        output_dir,
        rank,
        device=None,
        interval_seconds=0.5,
        worker_pids=None,
        full_memory=False,
    ):
        import psutil

        if interval_seconds <= 0:
            raise ValueError("profile_memory_interval_seconds must be positive")
        self.psutil = psutil
        self.process = psutil.Process()
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.device = device
        self.interval_seconds = float(interval_seconds)
        self.worker_pids = worker_pids
        self.full_memory = bool(full_memory)
        if self.full_memory:
            self.fields = self.fields + self.full_memory_fields
        self._stop = threading.Event()
        self._thread = None
        self._position = (-1, "initializing", 0)
        self._started = time.perf_counter()
        self._hostname = socket.gethostname()

    def set_step(self, step, stage="train", mini_epoch=0):
        self._position = (int(step), str(stage), int(mini_epoch))

    def _memory(self, process):
        """psutil memory record; the slower smaps-based one only when full_memory is on."""
        return process.memory_full_info() if self.full_memory else process.memory_info()

    def snapshot(self):
        rank_mem = self._memory(self.process)
        rank_rss = rank_mem.rss
        children = {child.pid: child for child in self.process.children(recursive=True)}
        if self.worker_pids is None:
            requested = []
            for pid, child in children.items():
                try:
                    if child.name() == "pt_data_worker":
                        requested.append(pid)
                except self.psutil.Error:
                    continue
        else:
            requested = list(self.worker_pids())
        workers = {}
        worker_mems = []
        for pid in set(requested):
            # Never sample a recycled PID belonging to an unrelated process.
            if pid not in children:
                continue
            try:
                mem = self._memory(children[pid])
            except self.psutil.Error:
                continue
            workers[str(pid)] = mem.rss
            worker_mems.append(mem)
        allocated, reserved = None, None
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
        step, stage, mini_epoch = self._position
        shm_used, shm_total = shm_usage()
        row = {
            "schema_version": 1,
            "timestamp": time.time(),
            "elapsed_seconds": time.perf_counter() - self._started,
            "hostname": self._hostname,
            "rank": self.rank,
            "pid": self.process.pid,
            "stage": stage,
            "mini_epoch": mini_epoch,
            "step": step,
            "rank_rss_bytes": rank_rss,
            "worker_rss_bytes": json.dumps(workers, sort_keys=True),
            "workers_rss_bytes": sum(workers.values()),
            "total_rss_bytes": rank_rss + sum(workers.values()),
            "gpu_allocated_bytes": allocated,
            "gpu_reserved_bytes": reserved,
            "shm_used_bytes": shm_used,
            "shm_total_bytes": shm_total,
        }
        if self.full_memory:
            row.update(self._full_memory_columns(rank_mem, worker_mems))
        return row

    def _full_memory_columns(self, rank_mem, worker_mems):
        def total(mems, attr):
            values = [getattr(mem, attr, None) for mem in mems]
            return None if any(v is None for v in values) else sum(values)

        try:
            vm = self.psutil.virtual_memory()
            node_used, node_total = vm.total - vm.available, vm.total
        except (AttributeError, self.psutil.Error):
            node_used, node_total = None, None
        return {
            "rank_uss_bytes": getattr(rank_mem, "uss", None),
            "rank_pss_bytes": getattr(rank_mem, "pss", None),
            "workers_uss_bytes": total(worker_mems, "uss"),
            "workers_pss_bytes": total(worker_mems, "pss"),
            "node_used_bytes": node_used,
            "node_total_bytes": node_total,
        }

    def start(self):
        if self._thread is not None:
            raise RuntimeError("MemoryTimeline is already started")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="loader-memory-profile", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        path = self.output_dir / f"memory_rank{self.rank:04d}.csv"
        try:
            needs_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fields)
                if needs_header:
                    writer.writeheader()
                while True:
                    writer.writerow(self.snapshot())
                    handle.flush()
                    if self._stop.wait(self.interval_seconds):
                        break
        except Exception:
            logger.exception("Memory profiling stopped unexpectedly for rank %s", self.rank)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
