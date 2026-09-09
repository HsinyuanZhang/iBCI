#!/usr/bin/env python3
"""Run matched concat/P16/P32 initialization and gradient audits on synthetic banks.

This script intentionally contains no NWB reader.  It is a ROOT-executed model
verification tool: use a fresh output directory and the device requested by the
caller (for example CUDA_VISIBLE_DEVICES=1 with --device cuda:0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKSPACE = ROOT.parent
for entry in (ROOT / "src", ROOT / "scripts", HERE, WORKSPACE / "btransform_unified_v1" / "src"):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from btransform_unified_v1.bank import TaskBank, array_sha256
from fusion_model import audit_matched_initialization, build_static


SEED = 42
FORWARD_TOL = 1.0e-5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_geometry(task: str) -> tuple[int, int, int, int]:
    if task == "m1":
        return 64, 100, 100, 16
    if task == "h1":
        return 176, 700, 300, 7
    raise ValueError(f"unsupported task {task}")


def _synthetic_bank(task: str) -> tuple[TaskBank, torch.Tensor, torch.Tensor]:
    """Produce one legal finite TaskBank and one nontrivial raw sequence."""
    units, e0_dim, context_bins, out_dim = _task_geometry(task)
    rng = np.random.RandomState(SEED + (1 if task == "m1" else 2))
    e0 = np.ascontiguousarray(rng.standard_normal((units, e0_dim)).astype(np.float32))
    carrier = np.ascontiguousarray(rng.standard_normal((units, 4)).astype(np.float32))
    store = np.ascontiguousarray(rng.standard_normal((1, context_bins, units)).astype(np.float32))
    targets = np.zeros((1, out_dim), dtype=np.float32)
    bank = TaskBank(
        session_id=f"synthetic-{task}-fusion-audit",
        E0=e0,
        carrier=carrier,
        unit_mask=np.ones(units, dtype=np.bool_),
        X_store=store,
        target_store=targets,
        window_ids=np.array([0], dtype=np.int64),
        calibration_meta={
            "shape": list(e0.shape),
            "trial_count": 10 if task == "m1" else 3,
            "budget": 10 if task == "m1" else 3,
            "estimator": "synthetic_fusion_audit_no_nwb",
            "array_sha256": array_sha256(e0),
        },
    )
    generator = torch.Generator(device="cpu").manual_seed(SEED + (101 if task == "m1" else 202))
    x = torch.randn((2, context_bins, units), generator=generator, dtype=torch.float32)
    valid = torch.ones((2, context_bins), dtype=torch.bool)
    return bank, x, valid


def _build_triplet(task: str, device: torch.device) -> dict[str, torch.nn.Module]:
    _units, _e0_dim, context_bins, _out_dim = _task_geometry(task)
    return {
        "concat": build_static(task, fusion="concat", proj_dim=16, context_bins=context_bins, seed=SEED, device=device),
        "p16": build_static(task, fusion="proj_add", proj_dim=16, context_bins=context_bins, seed=SEED, device=device),
        "p32": build_static(task, fusion="proj_add", proj_dim=32, context_bins=context_bins, seed=SEED, device=device),
    }


def _all_pairwise_gates(report: dict[str, Any]) -> None:
    for pair, value in report["pairwise"].items():
        _require(value["non_fusion_shared_byte_equal"], f"{pair}: non-fusion parameter bytes differ: {value['non_fusion_unequal_names']}")
        _require(value["effective_fold_shape_equal"], f"{pair}: folded token shape differs")
        _require(value["effective_fold_max_abs"] <= FORWARD_TOL, f"{pair}: folded token mismatch {value['effective_fold_max_abs']}")
    forward = report["forward"]
    _require(forward is not None, "audit forward report missing")
    for name, maximum in forward["max_abs_vs_anchor"].items():
        _require(maximum <= FORWARD_TOL, f"forward {name}: max_abs {maximum} exceeds {FORWARD_TOL}")


def _p32_metadata_gate(model: torch.nn.Module, task: str) -> dict[str, int]:
    frontend = model.frontend  # type: ignore[attr-defined]
    owner = model._frontend_owner  # type: ignore[attr-defined]
    observed = {
        "frontend_token_in": int(frontend.token_in),
        "owner_token_in": int(owner.token_in),
        "owner_proj_out_dim": int(owner.proj_out_dim),
        "owner_proj_groups": int(owner.proj_groups),
        "owner_proj_dim_override": int(owner.proj_dim_override),
        "token_mlp_input_features": int(frontend.token_mlp[0].in_features),
        "e0_proj_output_features": int(frontend.e0_proj.out_features),
    }
    expected = {
        "frontend_token_in": 36,
        "owner_token_in": 36,
        "owner_proj_out_dim": 32,
        "owner_proj_groups": 2,
        "owner_proj_dim_override": 32,
        "token_mlp_input_features": 36,
        "e0_proj_output_features": 32,
    }
    _require(observed == expected, f"{task}: P32 architecture metadata drift: {observed}")
    return observed


def _gradient_gate(model: torch.nn.Module, bank: TaskBank, x: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """Check the intended staged P32 gradient behavior without changing model law."""
    model.train()
    token_weight = model.frontend.token_mlp[0].weight  # type: ignore[attr-defined]
    projection_weight = model.frontend.e0_proj.weight  # type: ignore[attr-defined]
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    optimizer.zero_grad(set_to_none=True)
    first_loss = model(x, bank, input_valid_mask=valid).square().mean()
    first_loss.backward()
    initial_token_extra = token_weight.grad[:, 16:32].detach().abs().max().item()
    initial_projection_extra = projection_weight.grad[16:].detach().abs().max().item()
    _require(initial_token_extra > 0.0, "P32 initial added token columns have zero gradient")
    _require(initial_projection_extra == 0.0, "P32 new projection rows must have zero initial gradient")
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second_loss = model(x, bank, input_valid_mask=valid).square().mean()
    second_loss.backward()
    second_projection_extra = projection_weight.grad[16:].detach().abs().max().item()
    _require(second_projection_extra > 0.0, "P32 new projection rows did not receive gradient after token-column step")
    model.eval()
    return {
        "initial_loss": float(first_loss.detach().item()),
        "initial_added_token_column_grad_max_abs": float(initial_token_extra),
        "initial_new_projection_row_grad_max_abs": float(initial_projection_extra),
        "post_step_loss": float(second_loss.detach().item()),
        "post_step_new_projection_row_grad_max_abs": float(second_projection_extra),
    }


def _rng_gate(task: str, device: torch.device) -> bool:
    """P32 allocation must not consume more global RNG than the same P16 build."""
    _units, _e0_dim, context_bins, _out_dim = _task_geometry(task)
    torch.manual_seed(0xF0510)  # fixed seed used only for the construction-RNG comparison
    initial = torch.get_rng_state().clone()
    build_static(task, fusion="proj_add", proj_dim=16, context_bins=context_bins, seed=SEED, device=device)
    p16_after = torch.get_rng_state().clone()
    torch.set_rng_state(initial)
    build_static(task, fusion="proj_add", proj_dim=32, context_bins=context_bins, seed=SEED, device=device)
    p32_after = torch.get_rng_state().clone()
    equal = bool(torch.equal(p16_after, p32_after))
    _require(equal, f"{task}: P32 construction changed global RNG beyond P16")
    return equal


def _run_task(task: str, device: torch.device) -> dict[str, Any]:
    bank, x_cpu, valid_cpu = _synthetic_bank(task)
    x, valid = x_cpu.to(device), valid_cpu.to(device)
    models = _build_triplet(task, device)
    report = audit_matched_initialization(task, models, actual_banks=bank, x=x, input_valid_mask=valid)
    _all_pairwise_gates(report)
    metadata = _p32_metadata_gate(models["p32"], task)
    counts = report["parameter_counts"]
    _require(counts["p32"] > counts["p16"], f"{task}: P32 parameter count must exceed P16")
    gradients = _gradient_gate(models["p32"], bank, x, valid)
    rng_equal = _rng_gate(task, device)
    return {
        "task": task,
        "synthetic_bank": {"units": bank.E0.shape[0], "e0_dim": bank.E0.shape[1], "context_bins": x.shape[1]},
        "parameter_counts": counts,
        "matched_initialization": report,
        "p32_architecture": metadata,
        "gradient_gate": gradients,
        "global_rng_p16_equals_p32": rng_equal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh receipt directory")
    parser.add_argument("--device", required=True, help="torch device, e.g. cuda:0")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    _require(not output.exists(), f"output directory must be fresh: {output}")
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA device requested but unavailable")
    output.mkdir(parents=True)
    result = {
        "schema": "carrier_v4_fusion_matched_initialization_audit_v1",
        "status": "PASSED",
        "seed": SEED,
        "device": str(device),
        "forward_tolerance": FORWARD_TOL,
        "nwb_reads": 0,
        "tasks": {task: _run_task(task, device) for task in ("m1", "h1")},
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "fusion_model_sha256": _sha256_file(HERE / "fusion_model.py"),
    }
    receipt = output / "fusion_audit_receipt.json"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "receipt": str(receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
