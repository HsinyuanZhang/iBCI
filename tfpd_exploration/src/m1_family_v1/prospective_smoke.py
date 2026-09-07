"""CPU-only gates for the unlaunched M1 chron80 prospective family runs."""
from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tfpd_exploration.src.m1_optimized_v2 import bank as source_bank
from tfpd_exploration.src.m1_optimized_v2 import plan
from tfpd_exploration.src.m1_optimized_v2.data import build_source_only_datamodule, materialize_source_banks
from tfpd_exploration.src.m1_optimized_v2.model import build as build_query
from tfpd_exploration.src.m1_optimized_v2.source_dev import _clone, _keep, _split


def _name(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _query_pair_gate() -> dict[str, object]:
    """Prove shared FLAT state and exact zero-gate equivalence before training."""
    torch.manual_seed(42)
    flat = build_query("flat").eval()
    route = build_query("route").eval()
    flat_state = flat.state_dict()
    route_state = route.state_dict()
    shared = [key for key in flat_state if key in route_state]
    if any(not torch.equal(flat_state[key], route_state[key]) for key in shared):
        raise RuntimeError("new QueryAge16 pair lacks identical shared initialization")
    route_only = sorted(set(route_state) - set(flat_state))
    if not route_only or not all(key.startswith("frontend.attn.") for key in route_only):
        raise RuntimeError("unexpected ROUTE-only parameter inventory")
    if not torch.equal(route.frontend.attn.g, torch.zeros_like(route.frontend.attn.g)):
        raise RuntimeError("ROUTE gate is not zero initialized")
    banks = materialize_source_banks()
    x = torch.zeros(2, 100, 64)
    with torch.inference_mode():
        a = flat.forward_last(x, banks["ses-20120926"])
        b = route.forward_last(x, banks["ses-20120926"])
    error = float((a - b).abs().max())
    if error > 1e-5 + 1e-5 * float(a.abs().max()):
        raise RuntimeError(f"ROUTE(g=0) differs from FLAT: {error}")
    # The fixed per-example [B,64] whole-unit mask is generated once and can
    # be passed to either arm unchanged.
    keep = _keep(2, 1, 0, torch.device("cpu"))
    if keep.shape != (2, 64) or keep.dtype is not torch.bool:
        raise RuntimeError("paired dropout mask topology drift")
    return {"shared_parameter_keys": len(shared), "route_only_keys": route_only,
            "zero_gate_max_abs_error": error, "dropout_keep_shape": list(keep.shape)}


def _b3_fresh_gate() -> dict[str, object]:
    """Load a fresh historical B3 constructor, never the S-Fix student state."""
    experiment = str(plan.REPO_ROOT / "streaming_calibration_exp")
    if experiment not in sys.path:
        sys.path.insert(0, experiment)
    # Reproducible fresh construction; it never restores S-Fix state.
    torch.manual_seed(42)
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import module as sfix_module

    lit = sfix_module.make_module(plan.REPO_ROOT)
    lit.setup("fit")
    student = lit.student.train()
    loaded = source_bank.load()
    dm = build_source_only_datamodule(loaded)
    train_rows, dev_rows, _ = _split(dm)
    if set(train_rows) & set(dev_rows):
        raise RuntimeError("chron80 train/dev rows overlap")
    train = _clone(dm.train_dataset, train_rows, loaded)
    item = next(iter(DataLoader(train, batch_size=2, shuffle=False)))
    neural, target, calibration, sessions, carrier = item
    if any(_name(name) not in plan.SOURCE_SESSIONS for name in sessions):
        raise RuntimeError("non-source training row")
    output, _ = student(neural.float(), calib_trials=calibration.float(), side_features=None, carrier=carrier.float())
    loss = F.mse_loss(output[:, -1, :], target[:, -1, :])
    if not torch.isfinite(loss):
        raise RuntimeError("fresh B3 smoke loss is nonfinite")
    # It is deliberately only a backward smoke: no optimizer step mutates the
    # fresh construction and no checkpoint is written.
    loss.backward()
    return {"train_windows": len(train_rows), "dev_windows": len(dev_rows),
            "loss": float(loss.detach()), "fresh_student_state": True,
            "sfix_epoch011_loaded": False}


def run() -> dict[str, object]:
    return {"schema": "m1_family_v1_prospective_cpu_smoke_v1", "device": "cpu",
            "outer_query_opened": False, "b3_chron80": _b3_fresh_gate(),
            "queryage16_flat_route": _query_pair_gate()}


def seal_cpu_smoke() -> dict[str, object]:
    result = plan.RESULT_ROOT / "family_v1" / "prospective_cpu_smoke_v2.json"
    payload = run()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    result.parent.mkdir(parents=True, exist_ok=True)
    if result.exists() and result.read_text() != encoded:
        raise FileExistsError(f"frozen smoke differs: {result}")
    if not result.exists():
        result.write_text(encoded)
    return payload


if __name__ == "__main__":
    print(json.dumps(seal_cpu_smoke(), indent=2, sort_keys=True))
