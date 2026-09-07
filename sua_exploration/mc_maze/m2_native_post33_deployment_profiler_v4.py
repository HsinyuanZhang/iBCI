"""CUDA-synchronized deployment phase profiler for the score-sealed evaluator."""
from __future__ import annotations

from contextlib import contextmanager
import resource
import math
import time
from typing import Iterator

import torch


class DeploymentProfilerV4:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase-C deployment profiler requires CUDA")
        self.rows = {
            "support_calibration": {
                "wall_time_ns": 0, "invocations": 0,
                "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0,
                "host_max_rss_bytes": 0,
            },
            "streaming_inference": {
                "wall_time_ns": 0, "invocations": 0,
                "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0,
                "host_max_rss_bytes": 0,
            },
        }
        self.online_b1_microbenchmark = None

    def benchmark_online_b1(self, forward, neural_cpu: torch.Tensor) -> None:
        if self.online_b1_microbenchmark is not None:
            raise RuntimeError("B=1 online microbenchmark already executed")
        if neural_cpu.shape != (1, 50, 96) or neural_cpu.device.type != "cpu":
            raise ValueError("B=1 online microbenchmark requires one CPU neural window")
        neural = neural_cpu.to("cuda")
        samples = []
        with torch.no_grad():
            for _ in range(5):
                forward(neural)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                output = forward(neural)
                torch.cuda.synchronize()
                if output.shape != (1, 50, 2):
                    raise ValueError("B=1 cached deployment output shape drift")
                samples.append(time.perf_counter_ns() - start)
        ordered = sorted(samples)
        self.online_b1_microbenchmark = {
            "real_outer_neural_window": True,
            "behavior_target_read": False,
            "batch_size": 1,
            "input_residency": "cuda_preloaded",
            "warmup_repeats": 5,
            "timed_repeats": 20,
            "cuda_synchronize_before_and_after": True,
            "timing_ns": {
                "median": int((ordered[9] + ordered[10]) // 2),
                "p95": int(ordered[math.ceil(0.95 * len(ordered)) - 1]),
                "samples": samples,
            },
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        if phase not in self.rows:
            raise ValueError(f"unknown deployment cost phase {phase}")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter_ns()
        yield
        torch.cuda.synchronize()
        row = self.rows[phase]
        row["wall_time_ns"] += time.perf_counter_ns() - start
        row["invocations"] += 1
        row["peak_cuda_allocated_bytes"] = max(
            row["peak_cuda_allocated_bytes"], int(torch.cuda.max_memory_allocated())
        )
        row["peak_cuda_reserved_bytes"] = max(
            row["peak_cuda_reserved_bytes"], int(torch.cuda.max_memory_reserved())
        )
        row["host_max_rss_bytes"] = max(
            row["host_max_rss_bytes"],
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        )

    def payload(
        self, *, arm: str, fold: int, seed: int, outer_session: str,
        resolved_config: dict, cache_evidence: dict, descriptor_fit: dict,
        integrity_audit: dict,
    ) -> dict:
        if any(row["invocations"] <= 0 for row in self.rows.values()):
            raise RuntimeError("deployment profiler did not observe both phases")
        if self.rows["support_calibration"]["invocations"] != 1:
            raise RuntimeError("deployment support identity must be computed exactly once")
        if (
            cache_evidence.get("support_identity_computations") != 1
            or cache_evidence.get("query_decode_invocations")
            != self.rows["streaming_inference"]["invocations"]
        ):
            raise RuntimeError("deployment cache evidence/profiler invocation mismatch")
        if self.online_b1_microbenchmark is None:
            raise RuntimeError("deployment B=1 production microbenchmark is missing")
        batch_sizes = cache_evidence.get("query_batch_sizes")
        query_windows = cache_evidence.get("query_window_count")
        if (
            not isinstance(batch_sizes, list) or not batch_sizes
            or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in batch_sizes)
            or len(batch_sizes) != self.rows["streaming_inference"]["invocations"]
            or sum(batch_sizes) != query_windows
        ):
            raise RuntimeError("deployment query batch/window accounting failed")
        expected_side_shape = [1, 96, 4] if arm == "t4" else None
        expected_bytes_per_scan = 33 * 100 * 96 * 4 + (96 * 4 * 4 if arm == "t4" else 0)
        expected_audit = {
            "execution_device", "full_tensor_scan_invocations", "wall_time_ns",
            "scanned_bytes", "support_shape", "side_feature_shape", "sha256",
            "included_in_streaming_latency",
        }
        if not isinstance(integrity_audit, dict) or set(integrity_audit) != expected_audit:
            raise RuntimeError("deployment calibration integrity-audit keys mismatch")
        if (
            integrity_audit["execution_device"] != "cpu"
            or integrity_audit["full_tensor_scan_invocations"] != 2
            or not isinstance(integrity_audit["wall_time_ns"], int)
            or integrity_audit["wall_time_ns"] <= 0
            or integrity_audit["scanned_bytes"] != 2 * expected_bytes_per_scan
            or integrity_audit["support_shape"] != [1, 33, 100, 96]
            or integrity_audit["side_feature_shape"] != expected_side_shape
            or integrity_audit["sha256"] != cache_evidence.get(
                "support_and_side_sha256" if arm == "t4" else "support_sha256"
            )
            or integrity_audit["included_in_streaming_latency"] is not False
        ):
            raise RuntimeError("deployment calibration integrity-audit contract failed")
        sources = [
            session for index, session in {
                0: "ses-2020-10-19-Run1", 1: "ses-2020-10-19-Run2",
                2: "ses-2020-10-20-Run1", 3: "ses-2020-10-20-Run2",
                4: "ses-2020-10-27-Run1", 5: "ses-2020-10-27-Run2",
                6: "ses-2020-10-28-Run1",
            }.items() if index != fold
        ]
        return {
            "schema": "m2_post33_phase_c_deployment_cost_evidence_v4",
            "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
            "phase_id": "PHASE_C_V4", "arm": arm, "fold": fold, "seed": seed,
            "outer_session": outer_session, "source_sessions": sources,
            "score_value_disclosed": False, "formal_data_accessed": False,
            "profiler_semantics": (
                "CUDA-synchronized cached evaluator: calibration once; batched query timing is "
                "throughput-only; separate post-score B=1 microbenchmark uses one real neural "
                "window and reads no behavior target"
            ),
            "phases": self.rows,
            "query_execution": {
                "batch_sizes": batch_sizes,
                "query_window_count": query_windows,
                "batched_total_wall_time_ns": self.rows["streaming_inference"]["wall_time_ns"],
                "batched_mean_wall_time_per_window_ns": (
                    self.rows["streaming_inference"]["wall_time_ns"] / query_windows
                ),
                "batched_throughput_windows_per_second": (
                    query_windows * 1e9 / self.rows["streaming_inference"]["wall_time_ns"]
                ),
                "batched_latency_claim_permitted": False,
            },
            "online_b1_microbenchmark": self.online_b1_microbenchmark,
            "cache_evidence": cache_evidence,
            "descriptor_fit": descriptor_fit,
            "integrity_audit": integrity_audit,
            "resolved_config": resolved_config,
        }
