"""Guarded EMA-only scorers for the frozen H1 QueryAge formal surfaces.

This module is intentionally an additive scoring primitive.  It constructs no
cache, writes no artifact, selects no epoch, and performs no optimizer step.
Callers must supply an already validated immutable cache and invoke the
``guard`` callback around the worker's resource policy.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import torch

from tfpd_exploration.src.h1_optimized_v2.score import _windows
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

SELECTION_BINS = 2908
COMPLETE_BINS = 20325
WINDOW = 700
CHUNK = 8
SESSIONS = 13


def _r2_float64(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise RuntimeError("native FP64 prediction/target contract drift")
    denominator = float(np.square(target - target.mean(axis=0, keepdims=True)).sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("native FP64 target variance is invalid")
    value = float(1.0 - np.square(prediction - target).sum() / denominator)
    if not np.isfinite(value):
        raise RuntimeError("native FP64 R2 is nonfinite")
    return value


def _rows(cache: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    rows = cache.get("minival") if isinstance(cache, Mapping) else None
    if not isinstance(rows, Mapping) or len(rows) != SESSIONS:
        raise RuntimeError("formal H1 scorer requires exactly thirteen frozen minival sessions")
    ordered = sorted(rows.items())
    if len({name for name, _ in ordered}) != SESSIONS or any(not isinstance(name, str) or not name for name, _ in ordered):
        raise RuntimeError("formal H1 minival source identifiers drift")
    return ordered


def _ends(row: Mapping[str, Any], *, complete: bool) -> np.ndarray:
    neural, velocity = row.get("neural"), row.get("velocity")
    if not isinstance(neural, np.ndarray) or neural.dtype != np.float32 or neural.ndim != 2:
        raise RuntimeError("cached neural source must be rank-two FP32")
    if not isinstance(velocity, np.ndarray) or velocity.dtype != np.float32 or velocity.ndim != 2 or len(velocity) != len(neural):
        raise RuntimeError("cached native velocity source geometry drift")
    if complete:
        mask = row.get("eval_mask")
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (len(neural),):
            raise RuntimeError("complete eval-mask contract drift")
        result = np.flatnonzero(mask).astype(np.int64, copy=False)
    else:
        starts = row.get("query_starts")
        if not isinstance(starts, np.ndarray) or starts.dtype != np.int64 or starts.ndim != 1:
            raise RuntimeError("selection query-start contract drift")
        result = starts + (WINDOW - 1)
    if not len(result) or np.any(result < 0) or np.any(result >= len(neural)) or len(np.unique(result)) != len(result):
        raise RuntimeError("frozen source endpoint identity drift")
    return result


def _bank(row: Mapping[str, Any], device: torch.device) -> H1Bank:
    value = row.get("bank")
    if not isinstance(value, Mapping) or set(value) != {"E0", "T", "unit_mask"}:
        raise RuntimeError("cached H1 bank topology drift")
    return H1Bank(value["E0"].to(device), value["T"].to(device), value["unit_mask"].to(device))


def _raw_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def _restore_raw(model: torch.nn.Module, raw: Mapping[str, torch.Tensor], was_training: bool) -> bool:
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    same = set(current) == set(raw) and all(torch.equal(current[name].detach(), raw[name].to(current[name])) for name in raw)
    with torch.no_grad():
        for name, parameter in current.items():
            if name in raw:
                parameter.copy_(raw[name].to(device=parameter.device, dtype=parameter.dtype))
    model.train(was_training)
    return same


def _score(*, model: torch.nn.Module, ema: Any, cache: Mapping[str, Any], device: torch.device,
           guard: Callable[[], None], complete: bool) -> dict[str, Any]:
    if not callable(guard):
        raise TypeError("formal scorer requires a resource guard callback")
    rows = _rows(cache)
    raw, was_training = _raw_parameters(model), bool(model.training)
    pieces: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    session_ids: list[np.ndarray] = []
    endpoints: list[np.ndarray] = []
    error: BaseException | None = None
    try:
        with torch.inference_mode():
            def score(candidate: torch.nn.Module) -> None:
                candidate.eval()
                for session, row in rows:
                    guard()  # before source/cache endpoint/window access
                    ends = _ends(row, complete=complete)
                    guard()  # immediately before bank/window materialization
                    bank = _bank(row, device)
                    for offset in range(0, len(ends), CHUNK):
                        guard()  # before every `_windows` cache read and forward
                        x = torch.as_tensor(_windows(row["neural"], ends[offset:offset + CHUNK]), device=device)
                        guard()
                        value = candidate.forward_last(x, bank) / 20.0
                        guard()  # every forward is bounded on both sides
                        piece = value.detach().cpu().numpy().astype(np.float64, copy=False)
                        if piece.shape != (len(x), row["velocity"].shape[1]) or not np.isfinite(piece).all():
                            raise RuntimeError("EMA native prediction contract drift")
                        pieces.append(piece)
                    targets.append(np.asarray(row["velocity"][ends], dtype=np.float64))
                    session_ids.append(np.full(len(ends), session, dtype="U128"))
                    endpoints.append(np.asarray(ends, dtype=np.int64))
                    guard()
            guard()
            ema.score_with_ema(model, score)
            guard()
    except BaseException as exc:
        error = exc
        raise
    finally:
        unchanged = _restore_raw(model, raw, was_training)
        if not unchanged and error is None:
            raise RuntimeError("EMA scoring did not restore RAW parameters")
    prediction, target = np.concatenate(pieces), np.concatenate(targets)
    expected = COMPLETE_BINS if complete else SELECTION_BINS
    if len(prediction) != expected or len(target) != expected:
        raise RuntimeError(f"formal {'complete' if complete else 'selection'} surface cardinality drift")
    result = {"n_bins": int(len(prediction)), "r2_concat_float64": _r2_float64(prediction, target), "finite": True}
    if not complete:
        return result
    sid, end = np.concatenate(session_ids), np.concatenate(endpoints)
    if len(np.unique(sid)) != SESSIONS or len(np.unique(np.char.add(np.char.add(sid, ":"), end.astype(str)))) != len(end):
        raise RuntimeError("complete source session/endpoint identifiers drift")
    per = {name: _r2_float64(prediction[sid == name], target[sid == name]) for name in sorted(set(sid.tolist()))}
    result.update({"equal_session_mean_r2_float64": float(np.mean(list(per.values()))), "worst_session": min(per, key=per.get),
                   "worst_session_r2_float64": float(min(per.values())), "per_session_r2_float64": per,
                   "_prediction": prediction, "_target": target, "_session_id": sid, "_end": end})
    return result


def score_selection_cached_ema(model: torch.nn.Module, ema: Any, cache: Mapping[str, Any], device: torch.device,
                               guard: Callable[[], None]) -> dict[str, Any]:
    """Score exactly the 2,908 frozen selection endpoints under EMA."""
    return _score(model=model, ema=ema, cache=cache, device=device, guard=guard, complete=False)


def score_complete_cached_ema(model: torch.nn.Module, ema: Any, cache: Mapping[str, Any], device: torch.device,
                              guard: Callable[[], None]) -> dict[str, Any]:
    """Score exactly the 20,325 frozen complete endpoints under EMA."""
    return _score(model=model, ema=ema, cache=cache, device=device, guard=guard, complete=True)


__all__ = ("score_selection_cached_ema", "score_complete_cached_ema")
