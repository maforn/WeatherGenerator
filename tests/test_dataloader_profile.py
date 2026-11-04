"""CPU-only profile schema, worker attribution, and aligned-memory regression checks."""

import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load_module("loader_profile", "src/weathergen/utils/dataloader_profile.py")
analyze = load_module("analyze_loader", "scripts/analyze_dataloader_profile.py")


class ProcessError(Exception):
    pass


class FakeProcess:
    def __init__(self, pid=1, rss=100, name="python", children=(), gone=False):
        self.pid = pid
        self.rss = rss
        self._name = name
        self._children = children
        self.gone = gone

    def memory_info(self):
        if self.gone:
            raise ProcessError("gone")
        return types.SimpleNamespace(rss=self.rss)

    def memory_full_info(self):
        if self.gone:
            raise ProcessError("gone")
        return types.SimpleNamespace(rss=self.rss, uss=self.rss // 4, pss=self.rss // 2)

    def children(self, recursive):
        return self._children

    def name(self):
        return self._name


def fake_psutil(process):
    return types.SimpleNamespace(Process=lambda: process, Error=ProcessError)


def memory_row(timestamp, rank_rss, worker_rss, gpu=10):
    return {
        "timestamp": timestamp,
        "rank_rss_bytes": rank_rss,
        "workers_rss_bytes": worker_rss,
        "total_rss_bytes": rank_rss + worker_rss,
        "gpu_allocated_bytes": gpu,
        "gpu_reserved_bytes": gpu * 2 if gpu is not None else None,
    }


class SampleTests(unittest.TestCase):
    def test_attempt_metadata_and_accumulated_streams(self):
        proc = FakeProcess()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"psutil": fake_psutil(proc)}):
                with profile.SampleProfiler(directory, 2, 3, "train", 4, 5) as sample:
                    with sample.measure("ERA5", "read"):
                        pass
                    with sample.measure("ERA5", "read"):
                        pass
                    sample.add_stream("ERA5", 2_000_000, 10, 20)
                    sample.add_stream("ERA5", 1_000_000, 2, 3)
                    sample.valid = False
                    sample.get_batch_seconds = 0.1
                    proc.rss = 200
            row = json.loads(next(Path(directory).glob("*.jsonl")).read_text())
        self.assertEqual((row["rank"], row["worker_id"], row["mini_epoch"]), (2, 3, 4))
        self.assertEqual(row["sample_id"], "5")
        self.assertFalse(row["valid"])
        self.assertEqual(row["streams"]["ERA5"]["source_mb"], 3.0)
        self.assertEqual(row["streams"]["ERA5"]["source_tokens"], 12)
        self.assertEqual(row["streams"]["ERA5"]["target_tokens"], 23)
        self.assertGreaterEqual(row["streams"]["ERA5"]["read_seconds"], 0)
        self.assertEqual((row["rss_before_bytes"], row["rss_after_bytes"]), (100, 200))
        self.assertEqual(row["get_batch_seconds"], 0.1)

    def test_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"psutil": fake_psutil(FakeProcess())}):
                with self.assertRaisesRegex(ValueError, "sample failed"):
                    with profile.SampleProfiler(directory, 0, -1, "train", 0, 1):
                        raise ValueError("sample failed")
            row = json.loads(next(Path(directory).glob("*.jsonl")).read_text())
        self.assertEqual(row["error"], "ValueError")


class TimelineTests(unittest.TestCase):
    def test_worker_selection_and_uninitialized_cuda(self):
        children = [
            FakeProcess(2, 20, "pt_data_worker"),
            FakeProcess(3, 900),
            FakeProcess(4, 99, "pt_data_worker", gone=True),
        ]
        proc = FakeProcess(children=children)

        def forbidden(*args):
            raise AssertionError("CUDA allocator must not be queried before initialization")

        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_initialized=lambda: False, memory_allocated=forbidden, memory_reserved=forbidden
            )
        )
        with patch.dict(sys.modules, {"psutil": fake_psutil(proc), "torch": torch}):
            timeline = profile.MemoryTimeline("unused", 1)
            timeline.set_step(7, "train", 3)
            row = timeline.snapshot()
        self.assertEqual(json.loads(row["worker_rss_bytes"]), {"2": 20})
        self.assertEqual(row["total_rss_bytes"], 120)
        self.assertIsNone(row["gpu_allocated_bytes"])
        self.assertEqual((row["step"], row["mini_epoch"]), (7, 3))

    def test_explicit_pids_deduplicate_and_exclude_unrelated(self):
        proc = FakeProcess(children=[FakeProcess(2, 20), FakeProcess(3, 90)])
        with patch.dict(sys.modules, {"psutil": fake_psutil(proc)}):
            timeline = profile.MemoryTimeline("unused", 0, worker_pids=lambda: [2, 2, 999])
            row = timeline.snapshot()
        self.assertEqual(json.loads(row["worker_rss_bytes"]), {"2": 20})

    def test_csv_append_single_header_and_thread_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"psutil": fake_psutil(FakeProcess())}):
                for _ in range(2):
                    timeline = profile.MemoryTimeline(directory, 0)
                    timeline.start()
                    timeline.stop()
                    self.assertIsNone(timeline._thread)
            path = next(Path(directory).glob("*.csv"))
            rows = list(csv.DictReader(path.open()))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["schema_version"], "1")

    def test_full_memory_off_keeps_original_columns(self):
        proc = FakeProcess(children=[FakeProcess(2, 20, "pt_data_worker")])
        with patch.dict(sys.modules, {"psutil": fake_psutil(proc)}):
            timeline = profile.MemoryTimeline("unused", 0)
            row = timeline.snapshot()
        self.assertEqual(tuple(row), profile.MemoryTimeline.fields)
        self.assertNotIn("rank_pss_bytes", timeline.fields)

    def test_full_memory_adds_uss_pss_and_node_columns(self):
        proc = FakeProcess(
            children=[FakeProcess(2, 40, "pt_data_worker"), FakeProcess(3, 80, "pt_data_worker")]
        )
        fake = fake_psutil(proc)
        fake.virtual_memory = lambda: types.SimpleNamespace(total=1000, available=300)
        with patch.dict(sys.modules, {"psutil": fake}):
            timeline = profile.MemoryTimeline("unused", 0, full_memory=True)
            row = timeline.snapshot()
        self.assertEqual(tuple(row), timeline.fields)
        self.assertEqual(row["total_rss_bytes"], 220)
        self.assertEqual((row["rank_uss_bytes"], row["rank_pss_bytes"]), (25, 50))
        self.assertEqual((row["workers_uss_bytes"], row["workers_pss_bytes"]), (30, 60))
        self.assertEqual((row["node_used_bytes"], row["node_total_bytes"]), (700, 1000))

    def test_full_memory_without_pss_support_reports_none(self):
        proc = FakeProcess(children=[FakeProcess(2, 40, "pt_data_worker")])
        proc.memory_full_info = lambda: types.SimpleNamespace(rss=100, uss=25)
        with patch.dict(sys.modules, {"psutil": fake_psutil(proc)}):
            row = profile.MemoryTimeline("unused", 0, full_memory=True).snapshot()
        self.assertIsNone(row["rank_pss_bytes"])
        self.assertIsNone(row["node_used_bytes"])

    def test_numa_memory_splits_cpu_and_hbm_nodes(self):
        with tempfile.TemporaryDirectory() as root:
            for node, cpus, total, free, files, shmem in (
                (0, "0-71", 1000, 100, 300, 200),
                (1, "", 500, 400, 0, 0),
            ):
                path = Path(root) / f"node{node}"
                path.mkdir()
                (path / "cpulist").write_text(cpus + "\n")
                (path / "meminfo").write_text(
                    f"Node {node} MemTotal: {total} kB\nNode {node} MemFree: {free} kB\n"
                    f"Node {node} FilePages: {files} kB\nNode {node} Shmem: {shmem} kB\n"
                )
            numa = profile.numa_memory(root)
        self.assertEqual(numa["cpu"], {"used": 800 * 1024, "total": 1000 * 1024})
        self.assertEqual(numa["hbm"], {"used": 100 * 1024, "total": 500 * 1024})

    def test_numa_memory_without_sysfs(self):
        self.assertIsNone(profile.numa_memory("/nonexistent/sysfs/node"))

    def test_bad_interval_rejected(self):
        with patch.dict(sys.modules, {"psutil": fake_psutil(FakeProcess())}):
            with self.assertRaises(ValueError):
                profile.MemoryTimeline("unused", 0, interval_seconds=0)


class AnalysisTests(unittest.TestCase):
    def test_alignment_does_not_add_unrelated_peaks_or_fill_long_gaps(self):
        traces = {
            ("node", 0): [memory_row(0, 100, 0), memory_row(1, 10, 0), memory_row(10, 10, 0)],
            ("node", 1): [memory_row(0, 10, 0), memory_row(1, 100, 0)],
        }
        aligned = analyze.align_memory(traces, max_gap_seconds=0.5)
        self.assertEqual([row["timestamp"] for row in aligned], [0, 1])
        self.assertEqual(max(row["total_rss_bytes"] for row in aligned), 110)

    def test_alignment_requires_every_rank_and_handles_missing_cuda(self):
        traces = {
            ("node", 0): [memory_row(0, 100, 0, gpu=None), memory_row(1, 10, 0, gpu=None)],
            ("node", 1): [memory_row(1, 10, 0, gpu=None)],
        }
        aligned = analyze.align_memory(traces)
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0]["timestamp"], 1)
        self.assertIsNone(aligned[0]["gpu_allocated_bytes"])

    def test_metrics_intervals_and_stall_definition(self):
        metrics = [
            {
                "elapsed_training_time_seconds": time,
                "loss_avg_mean": loss,
                "performance.memory.train.max_allocated_gib": 5,
            }
            for time, loss in [(100, 0.8), (103, 0.7), (110, 0.6)]
        ]
        result = analyze.training_summary(metrics)
        self.assertEqual(result["median_step_seconds"], 5)
        self.assertEqual(result["stalls"], 1)
        self.assertEqual(result["loss_end"], 0.6)
        self.assertIsNone(result["mean_after_10_steps_seconds"])

    def test_cli_reads_samples_and_writes_machine_readable_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "example"
            run.mkdir()
            sample = {
                "stage": "train",
                "hostname": "node",
                "rank": 0,
                "worker_id": 1,
                "valid": True,
                "total_seconds": 2,
                "rss_before_bytes": 100,
                "rss_after_bytes": 200,
                "streams": {"ERA5": {"read_seconds": 1, "source_tokens": 42}},
            }
            (run / "samples_rank0000_worker001.jsonl").write_text(json.dumps(sample) + "\n")
            analyze.main(["--run", str(run), "--no-plots"])
            summary = json.loads((run / "plots/summary.json").read_text())[0]
        self.assertEqual(summary["slowest_stream_by_mean_stage_time"], "ERA5")
        self.assertEqual(summary["attempted_samples"], 1)
        self.assertEqual(summary["mean_per_stream"]["ERA5"]["source_tokens"], 42)


class PayloadTests(unittest.TestCase):
    def test_counts_shared_storage_once_and_groups_by_path(self):
        import numpy as np
        import torch

        class Sample:
            def __init__(self):
                self.source = {"era5": torch.zeros(256, dtype=torch.float32)}
                base = torch.zeros(100, dtype=torch.float64)
                self.target = [base, base[10:20]]  # view of the same storage
                self.coords = np.zeros(50, dtype=np.float32)
                self.name = "x"

        sample = Sample()
        sample.self_ref = sample  # cycles must not loop
        result = profile.payload_bytes(sample)
        self.assertEqual(result["torch_bytes"], 256 * 4 + 100 * 8)
        self.assertEqual(result["numpy_bytes"], 200)
        self.assertEqual(result["tensors"], 2)
        self.assertEqual(result["groups"]["source.era5"], 1024)
        self.assertEqual(result["groups"]["target"], 800)

    def test_shm_usage_missing_path(self):
        self.assertEqual(profile.shm_usage("/nonexistent/shm"), (None, None))


if __name__ == "__main__":
    unittest.main()
