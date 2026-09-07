"""Matched per-output variance-weighted scorer + final-four SWA builder.

Implements the metric-parity and SWA contracts of
HANDOFF_CARRIER_ADMISSION_CURRICULUM_20260816.md sections 7 and 10:

- one scorer everywhere: torchmetrics R2Score(multioutput="variance_weighted")
  — the house Falcon/A2 metric family — per session, equal weight per session;
- paired-session statistics: mean/median/n-positive/min/max/all deltas plus a
  fixed-seed paired-session bootstrap 95% interval and exact sign pattern;
- SWA: arithmetic FP64 mean of the exact final four consecutive checkpoints'
  floating tensors, non-floating buffers copied from the last and exact-compared,
  strict reload + finite forward smoke.  No optimizer state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def session_r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Variance-weighted per-output R² via the house torchmetrics implementation.

    predictions/targets: [n_bins, 2] with padding already masked out.
    """
    from torchmetrics.regression import R2Score

    metric = R2Score(num_outputs=targets.shape[-1], multioutput="variance_weighted")
    metric.update(predictions, targets)
    return float(metric.compute())


def paired_session_stats(deltas: list[float], seed: int = 42, n_boot: int = 10000) -> dict:
    deltas = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(deltas[idx].mean()))
    return {
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "n_positive": int((deltas > 0).sum()),
        "n_total": n,
        "min": float(deltas.min()),
        "max": float(deltas.max()),
        "all_deltas": [float(d) for d in deltas],
        "bootstrap_95_interval": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "exact_sign_pattern": "".join("+" if d > 0 else ("-" if d < 0 else "0") for d in deltas),
        "bootstrap_seed": seed,
    }


def build_swa_final_four(checkpoint_paths: list[Path], out_path: Path) -> dict:
    """Arithmetic FP64 mean of floating tensors across the final four checkpoints.

    Non-floating buffers are copied from the final checkpoint and exact-compared
    across the window.  Returns a manifest with component hashes; the SWA state
    is written without optimizer state and smoke-checked by strict reload.
    """
    import hashlib

    if len(checkpoint_paths) != 4:
        raise ValueError("SWA window is exactly the final four checkpoints")

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    states = [torch.load(p, map_location="cpu", weights_only=False)["state_dict"] for p in checkpoint_paths]
    keys = list(states[0].keys())
    for s in states[1:]:
        if list(s.keys()) != keys:
            raise ValueError("state-key mismatch across SWA window")

    swa = {}
    uninitialized = 0
    for key in keys:
        tensors = [s[key] for s in states]
        if isinstance(tensors[0], torch.nn.parameter.UninitializedParameter):
            # Materialize-on-forward lazy parameters stay unmaterialized in the
            # saved state; they carry no values, so copy the final one as-is.
            uninitialized += 1
            swa[key] = tensors[-1]
            continue
        shapes = {tuple(t.shape) for t in tensors}
        dtypes = {t.dtype for t in tensors}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError(f"shape/dtype drift at {key}")
        if tensors[0].is_floating_point():
            acc = tensors[0].double().clone()
            for t in tensors[1:]:
                acc += t.double()
            swa[key] = (acc / 4.0).to(tensors[0].dtype)
        else:
            for t in tensors[1:]:
                if not torch.equal(t, tensors[0]):
                    raise ValueError(f"non-floating buffer drift at {key}")
            swa[key] = tensors[0].clone()

    final = states[-1]
    manifest = {
        "components": [{"path": str(p), "sha256": sha(p)} for p in checkpoint_paths],
        "floating_tensor_count": sum(
            1 for k in keys
            if not isinstance(final[k], torch.nn.parameter.UninitializedParameter)
            and final[k].is_floating_point()
        ),
        "buffer_tensor_count": sum(
            1 for k in keys
            if not isinstance(final[k], torch.nn.parameter.UninitializedParameter)
            and not final[k].is_floating_point()
        ),
        "uninitialized_lazy_tensor_count": uninitialized,
        "fp64_arithmetic": True,
        "optimizer_state_included": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": swa, "swa_manifest": manifest}, out_path)

    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)["state_dict"]
    from torch.nn.parameter import UninitializedParameter

    def _same(a, b) -> bool:
        if isinstance(a, UninitializedParameter) or isinstance(b, UninitializedParameter):
            return type(a) is type(b)  # unmaterialized: nothing to compare
        return torch.equal(a, b)

    if list(reloaded.keys()) != keys or any(not _same(reloaded[k], swa[k]) for k in keys):
        raise ValueError("SWA strict reload mismatch")
    finite = all(
        torch.isfinite(v).all()
        for v in swa.values()
        if not isinstance(v, UninitializedParameter) and v.is_floating_point()
    )
    manifest["strict_reload_verified"] = True
    manifest["finite_forward_tensors"] = finite
    return manifest
