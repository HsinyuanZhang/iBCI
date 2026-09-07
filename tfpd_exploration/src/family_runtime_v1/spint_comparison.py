"""Measured M1 original-SPINT / Transformer public-call comparison.

Original SPINT caches only calibration identity and the fixed output queries.
Its finite raw W100 window and neural forward are fully recomputed each call.
All candidates consume exactly the same continuous source-bin stream on CPU.
This is a local original-teacher comparison, not an official/container result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m1_runtime_v3.public_benchmark import material
from tfpd_exploration.src.m1_runtime_v3 import HeterogeneousCurrentQueryStream
from .m1 import FiveTokenCurrentQueryStream
from .m1_lifted import LiftedFiveTokenCurrentQueryStream
from .benchmark import sha, stats


class CachedOriginalSpintStream:
    def __init__(self, model, calibration):
        self.model = model.eval()
        self.calibration = calibration
        with torch.no_grad():
            values = model.fc_id_in(calibration.permute(0, 1, 3, 2)).mean(dim=1)
            self.identity = model.fc_id_out(values)
            self.query = model.fc_in(model.rep).expand(len(calibration), -1, -1).contiguous()
        self.raw = torch.zeros(len(calibration), model.window_size, calibration.shape[-1])
        self._versions = tuple((p.data_ptr(), p._version) for p in model.parameters())

    @torch.no_grad()
    def predict(self, value):
        if self.model.training or self._versions != tuple((p.data_ptr(), p._version) for p in self.model.parameters()):
            raise RuntimeError("SPINT immutable runtime model changed")
        if value.dtype != np.float32 or value.shape != (len(self.raw), self.raw.shape[-1]) or not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValueError("finite contiguous float32 whole-batch observations required")
        self.raw = torch.cat((self.raw[:, 1:], torch.from_numpy(value)[:, None]), dim=1)
        source = self.model.fc_in(self.raw.permute(0, 2, 1) + self.identity)
        hidden, _ = self.model.transformer(self.query, source)
        output = self.model.fc_out(hidden)[..., -1]
        return output.numpy().astype(np.float32, copy=True)


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, bank, _, values, selected, cache, names = material(4, args.calls + args.warmup)
    from tfpd_exploration.src.m1_family_v1.original_teacher_replay import CKPT
    from tfpd_exploration.src.m1_optimized_v2.data import build_source_only_datamodule
    from tfpd_exploration.src.m1_optimized_v2.plan import REPO_ROOT
    sys.path.insert(0, str(REPO_ROOT / "streaming_calibration_exp"))
    from src.models.components.spint import SpintModel
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    teacher = SpintModel(1024, 16, 100, num_heads=8, num_layers=1, num_id_layers=3)
    teacher.load_state_dict({k[4:]: v for k, v in payload["state_dict"].items() if k.startswith("net.")}, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    dm = build_source_only_datamodule()
    calibration = torch.stack([torch.as_tensor(dm.train_dataset.base.calib_trialized_neural_features[n][:10], dtype=torch.float32) for n in names])
    start = time.perf_counter_ns()
    spint = CachedOriginalSpintStream(teacher, calibration)
    cold_spint = (time.perf_counter_ns() - start) / 1e6
    with torch.no_grad():
        full = teacher(spint.raw, calib_trialized_neural_features=calibration)[:, -1].numpy()
        cached = spint.predict(np.zeros((4, 64), dtype=np.float32))
        np.testing.assert_allclose(cached, full, atol=1e-5, rtol=1e-5)
    spint_max_error = float(np.abs(cached - full).max())
    engines = {"spint_original_cached": spint, "transformer_v3": HeterogeneousCurrentQueryStream(model, bank),
               "transformer_five_token": FiveTokenCurrentQueryStream(model, bank),
               "transformer_lifted": LiftedFiveTokenCurrentQueryStream(model, bank)}
    timing = {key: [] for key in engines}
    keys = list(engines)
    max_error = {key: 0. for key in keys if key.startswith("transformer")}
    with torch.no_grad():
        for index, value in enumerate(values):
            output = {}
            shift = index % len(keys)
            order = keys[shift:] + keys[:shift]
            for name in order:
                start = time.perf_counter_ns()
                output[name] = engines[name].predict(value)
                duration = (time.perf_counter_ns() - start) / 1e6
                if index >= args.warmup:
                    timing[name].append(duration)
            for name in max_error:
                diff = np.abs(output[name] - output["transformer_v3"])
                max_error[name] = max(max_error[name], float(diff.max()))
                np.testing.assert_allclose(output[name], output["transformer_v3"], atol=1e-5, rtol=1e-5)
            if index in (0, 3, 4, 98, 99, 100):
                full = teacher(spint.raw, calib_trialized_neural_features=calibration)[:, -1].numpy()
                np.testing.assert_allclose(output["spint_original_cached"], full, atol=1e-5, rtol=1e-5)
                spint_max_error = max(spint_max_error, float(np.abs(output["spint_original_cached"] - full).max()))
    metrics = {key: stats(value) for key, value in timing.items()}
    result = {"schema": "m1_original_spint_same_host_source_b4_latency_v1", "batch": 4, "threads": args.threads,
              "calls": args.calls, "warmup": args.warmup, "rows": names, "fourth_lane": "repeat first source",
              "public_host_to_numpy_output": True, "p95_ratio_to_spint": {k: v["p95_ms"] / metrics["spint_original_cached"]["p95_ms"] for k, v in metrics.items()},
              "timing": metrics, "transformer_native_parity_max_abs": max_error,
              "spint_cached_vs_original_full_max_abs": spint_max_error, "spint_reset_identity_query_ms": cold_spint,
              "spint_checkpoint": str(CKPT), "spint_checkpoint_sha256": sha(CKPT),
              "spint_operator": "Original SpintModel d1024/layer1/IDlayers3, M10 calibration, no B3/T4/rSyn3 extension",
              "transformer_state_sha256": selected["plain_ema_model_state_sha256"], "runtime_cache_sha256": cache["npz_sha256"],
              "calibration_tensor_sha256": __import__("hashlib").sha256(calibration.numpy().tobytes()).hexdigest(),
              "affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__,
              "code_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
              "not_container_or_official_latency": True, "not_generalization_or_quality_test": True,
              "parameter_updates": 0, "target_arrays_used_by_latency_loop": False,
              "source_loader_disclosure": "source-only datamodule opens/materializes source neural and behavior objects; behavior tensors are not used by this latency loop; no outer query loader"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--calls", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
