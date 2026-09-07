#!/usr/bin/env python3
"""Run reviewed v2 synthetic controls; default is no-GPU/no-data preflight."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


os.environ["PYTHONNOUSERSITE"] = "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "cebra_exploration/src", ROOT / "cebra_exploration/third_party/cebra", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import track_b_v2_post_cost_synthetic_controls_v2 as controls  # noqa: E402


def _nvidia_identity(physical: str) -> dict[str, str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id",
         "--format=csv,noheader,nounits", "-i", physical],
        check=True, capture_output=True, text=True,
    )
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    controls.require(len(fields) == 6 and fields[0] == physical, "physical GPU identity drift")
    return dict(zip(("physical_index", "uuid", "name", "total_memory_mib", "driver_version", "pci_bus_id"), fields))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-have-authorization", action="store_true")
    parser.add_argument("--physical-gpu", default=controls.PHYSICAL_GPU_INDEX)
    args = parser.parse_args(argv)
    if args.execute != args.i_have_authorization:
        parser.error("v2 execution requires both --execute and --i-have-authorization")
    try:
        controls.validate_physical_gpu(args.physical_gpu)
        preflight = controls.build_preflight()
        if not args.execute:
            sys.stdout.write(json.dumps(preflight, sort_keys=True, indent=2) + "\n")
            return 0
        launch_closure = preflight["implementation_closure"]
        controls.assert_output_fresh()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.physical_gpu
        import torch
        import torchmetrics
        import cebra
        nvidia = _nvidia_identity(args.physical_gpu)
        controls.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                         "exactly one logical GPU must be visible")
        properties = torch.cuda.get_device_properties(0)
        controls.require(properties.name == nvidia["name"], "torch/nvidia-smi GPU identity drift")
        cuda_identity = nvidia | {
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "logical_device": "cuda:0",
            "logical_device_count": torch.cuda.device_count(),
            "torch_device_name": properties.name,
            "torch_total_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "torch_path": str(Path(torch.__file__).resolve()),
        }
        payload = controls.execute_controls(
            preflight=preflight, torch=torch, torchmetrics=torchmetrics, cebra=cebra,
            cuda_identity=cuda_identity, launch_closure=launch_closure,
        )
        controls.validate_measurement_receipt(payload)
        controls.require(launch_closure == controls.implementation_closure(),
                         "implementation closure drift immediately before publication")
        controls.assert_output_fresh()
        published = controls.route.write_immutable_pair(controls.CANONICAL_OUTPUT, payload)
        sys.stdout.write(json.dumps(published, sort_keys=True, indent=2) + "\n")
        return 0
    except (controls.PostCostSyntheticControlV2Error, controls.route.TrackBV2ActualCpuError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

