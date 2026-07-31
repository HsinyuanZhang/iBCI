#!/usr/bin/env python3
"""No-data CUDA latency audit for coupled and cached-K decoder paths.

This is a deployment-cost diagnostic only.  It does not load neural data,
calibration labels, validation sessions or formal sessions, and its result is
never used for checkpoint or architecture selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[2]
_SCE_ROOT = _ROOT / "streaming_calibration_exp"
sys.path.insert(0, str(_SCE_ROOT))

from src.models.decoupled_kv_v2_module import (  # noqa: E402
    TeacherReadinDecoupledLitModule,
)
from src.models.head_oracle_module import (  # noqa: E402
    TeacherHeadOracleLitModule,
)
from src.models.streaming_calibration_module import (  # noqa: E402
    StreamingCalibrationLitModule,
)


DEFAULT_TEACHER = (
    _ROOT
    / "sua_exploration/checkpoints/teacher_mc_maze/"
    "best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _common_kwargs(teacher: Path) -> dict:
    return {
        "task": "mc_maze",
        "variant": "B3S",
        "teacher_ckpt_path": str(teacher),
        "window_size": 50,
        "trial_length": 100,
        "id_hidden_dim": 128,
        "hidden_dim": 64,
        "pad_value": -1.0,
        "freeze_decoder": False,
        "freeze_encoder_base": False,
        "loss_mode": "task_only",
        "lambda_y": 1.0,
        "lambda_E": 0.1,
        "decode_last_timestep_only": True,
        "predict_scaled_behavior": True,
        "behavior_scaling_factor": 5.0,
        "identity_mode": "calibrated",
        "fixed_slot_count": 0,
        "fixed_slot_dim": 32,
        "fixed_slot_mode": "soft",
        "fixed_slot_fusion": "film",
        "fixed_slot_temperature": 1.0,
        "decoder_mode": "coupled",
        "side_dim": 4,
        "electrode_embed_dim": 0,
        "num_electrodes": 0,
        "encoder_warmstart_path": None,
        "optimizer": partial(torch.optim.Adam, lr=1.0e-4),
        "scheduler": None,
        "compile": False,
    }


def _make_model(
    path: str, teacher: Path
) -> StreamingCalibrationLitModule:
    common = _common_kwargs(teacher)
    if path == "coupled":
        model = StreamingCalibrationLitModule(**common)
    elif path == "v2_cached_k":
        model = TeacherReadinDecoupledLitModule(
            **common,
            v2_key_mode="e_t4",
            v2_key_dim=48,
            v2_value_dim=64,
            v2_key_permutation_seed=None,
        )
    elif path == "exact_head_cached_k":
        model = TeacherHeadOracleLitModule(
            **common,
            oracle_key_mode="e_t4",
            oracle_key_permutation_seed=None,
        )
    else:
        raise ValueError(f"unsupported decoder path: {path}")
    model.setup("fit")
    if model.student is None:
        raise RuntimeError(f"{path}: setup did not construct a student")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def _elapsed_ms(
    fn: Callable[[], torch.Tensor | object],
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, int]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    milliseconds = start.elapsed_time(end) / iterations
    peak_increment = max(
        0, torch.cuda.max_memory_allocated() - baseline
    )
    return float(milliseconds), int(peak_increment)


def _benchmark_shape(
    model: StreamingCalibrationLitModule,
    *,
    path: str,
    units: int,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict:
    assert model.student is not None
    student = model.student.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(20260731 + units)
    neural = torch.randn(
        1, 50, units, generator=generator, device=device
    )
    identity = torch.randn(
        1, units, 50, generator=generator, device=device
    )
    t4 = torch.randn(
        1, units, 4, generator=generator, device=device
    )

    with torch.inference_mode():
        if path == "coupled":
            online = lambda: student.decode_with_identity(
                neural, identity
            )
            calibration = None
            state_bytes = identity.numel() * identity.element_size()
        elif path == "v2_cached_k":
            state = student.derive_decoupled_kv_state(identity, t4)
            online = lambda: student.decode_with_decoupled_kv_state(
                neural, state
            )
            calibration = lambda: student.derive_decoupled_kv_state(
                identity, t4
            )
            state_bytes = (
                state.projected_key.numel()
                * state.projected_key.element_size()
            )
        elif path == "exact_head_cached_k":
            state = student.derive_head_oracle_state(identity)
            online = lambda: student.decode_with_head_oracle_state(
                neural, state
            )
            calibration = lambda: student.derive_head_oracle_state(
                identity
            )
            state_bytes = state.nbytes
        else:
            raise AssertionError(path)

        online_ms, online_peak = _elapsed_ms(
            online, warmup=warmup, iterations=iterations
        )
        calibration_result = None
        if calibration is not None:
            calibration_ms, calibration_peak = _elapsed_ms(
                calibration,
                warmup=max(10, warmup // 10),
                iterations=max(100, iterations // 10),
            )
            calibration_result = {
                "milliseconds_per_refresh": calibration_ms,
                "peak_temporary_bytes_above_baseline": (
                    calibration_peak
                ),
            }
    return {
        "units": units,
        "online": {
            "milliseconds_per_window": online_ms,
            "windows_per_second": 1000.0 / online_ms,
            "peak_temporary_bytes_above_baseline": online_peak,
        },
        "calibration_decoder_state_refresh": calibration_result,
        "persistent_decoder_state_bytes_fp32": state_bytes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher_ckpt", default=str(DEFAULT_TEACHER)
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--units", default="32,64,96")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latency audit")
    if args.warmup < 1 or args.iterations < 100:
        raise ValueError("warmup>=1 and iterations>=100 are required")
    try:
        units = tuple(
            int(item.strip())
            for item in args.units.split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise ValueError("units must be comma-separated integers") from exc
    if not units or any(value <= 0 for value in units):
        raise ValueError("unit counts must be positive")
    teacher = Path(args.teacher_ckpt).expanduser().resolve()
    if not teacher.is_file():
        raise FileNotFoundError(teacher)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.manual_seed(20260731)
    torch.backends.cudnn.benchmark = False

    paths = ("coupled", "v2_cached_k", "exact_head_cached_k")
    results: dict[str, list[dict]] = {}
    for path in paths:
        model = _make_model(path, teacher).to(device).eval()
        results[path] = [
            _benchmark_shape(
                model,
                path=path,
                units=count,
                device=device,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            for count in units
        ]
        model.to("cpu")
        del model
        torch.cuda.empty_cache()

    by_units = {
        str(count): {
            path: next(
                item["online"]["milliseconds_per_window"]
                for item in results[path]
                if item["units"] == count
            )
            for path in paths
        }
        for count in units
    }
    relative = {
        count: {
            path: latency / values["coupled"]
            for path, latency in values.items()
        }
        for count, values in by_units.items()
    }
    properties = torch.cuda.get_device_properties(device)
    payload = {
        "schema_version": 1,
        "purpose": "no_data_decoder_online_latency_diagnostic",
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_role": "none",
        "neural_data_opened": False,
        "calibration_labels_opened": False,
        "validation_sessions_opened": False,
        "formal_sessions_opened": False,
        "teacher_checkpoint": str(teacher),
        "teacher_sha256": sha256_file(teacher),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "device": {
            "index": args.gpu,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
        },
        "protocol": {
            "dtype": "float32",
            "batch_size": 1,
            "window_size": 50,
            "unit_counts": list(units),
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "cuda_event_timing": True,
            "inference_mode": True,
            "models_measured_sequentially": True,
        },
        "results": results,
        "online_latency_ratio_vs_coupled_by_units": relative,
        "limitations": [
            "RTX3090 latency is not target-hardware latency",
            "decoder-only; identity encoder and I/O are excluded",
            "synthetic tensors; no accuracy inference is permitted",
            "single-process warm-cache measurement",
        ],
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(out),
        "device": properties.name,
        "milliseconds_per_window": by_units,
        "relative_to_coupled": relative,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
