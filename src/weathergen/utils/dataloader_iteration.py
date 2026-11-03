"""Profile parent-side waiting and memory without forking with our sampler thread running."""

import json
import time
from contextlib import contextmanager

from weathergen.utils.dataloader_profile import MemoryTimeline, profile_output_dir


def _worker_pids(iterators):
    # PyTorch has no public worker-PID API. Keep this compatibility dependency in one place.
    return sorted(
        {
            worker.pid
            for iterator in iterators
            if iterator is not None
            for worker in getattr(iterator, "_workers", ())
            if worker.pid is not None and worker.is_alive()
        }
    )


@contextmanager
def profile_loader_iteration(cf, iterator, device, stage, mini_epoch, other_iterators=()):
    """Call *after* iter(loader), and exit before creating any other loader iterator.

    Both outputs are opt-in. The original iterator is returned untouched when disabled.
    Explicit iterator handles identify loader workers without counting unrelated children.
    """
    timing = cf.data_loading.get("profile_timing", False)
    memory = cf.data_loading.get("profile_memory_timeline", False)
    if not timing and not memory:
        yield iterator
        return
    directory = profile_output_dir(cf)
    directory.mkdir(parents=True, exist_ok=True)
    rank = cf.get("rank", 0)
    wait_file = None
    timeline = None
    try:
        if timing:
            wait_file = (directory / f"loader_wait_rank{rank:04d}.jsonl").open("a", buffering=1)
        if memory:
            timeline = MemoryTimeline(
                directory,
                rank=rank,
                device=device,
                interval_seconds=cf.data_loading.get("profile_memory_interval_seconds", 0.5),
                worker_pids=lambda: _worker_pids((iterator, *other_iterators)),
                full_memory=cf.data_loading.get("profile_memory_full", False),
            )
            timeline.set_step(0, stage=stage, mini_epoch=mini_epoch)
            timeline.start()

        def batches():
            step = 0
            while True:
                if timeline:
                    timeline.set_step(step, stage=stage, mini_epoch=mini_epoch)
                start = time.perf_counter() if timing else None
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                if wait_file:
                    wait_file.write(
                        json.dumps(
                            {
                                "timestamp": time.time(),
                                "rank": rank,
                                "stage": stage,
                                "mini_epoch": mini_epoch,
                                "step": step,
                                "wait_seconds": time.perf_counter() - start,
                            }
                        )
                        + "\n"
                    )
                yield batch
                del batch
                step += 1

        yield batches()
    finally:
        if timeline:
            timeline.stop()
        if wait_file:
            wait_file.close()
