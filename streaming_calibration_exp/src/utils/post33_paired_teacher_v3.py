"""Local fail-closed resolver for paired SPINT completion receipts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
PHASE_ID = "PHASE_B_V3"
FOLDS = {
    0: "ses-2020-10-19-Run1",
    1: "ses-2020-10-19-Run2",
    2: "ses-2020-10-20-Run1",
    3: "ses-2020-10-20-Run2",
    4: "ses-2020-10-27-Run1",
    5: "ses-2020-10-27-Run2",
    6: "ses-2020-10-28-Run1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_regular_file(raw: str | Path) -> Path:
    path = Path(raw)
    canonical = path.resolve(strict=True)
    if path.is_symlink() or not canonical.is_file() or str(path) != str(canonical):
        raise ValueError(f"receipt requires a canonical regular file: {raw}")
    return canonical


def resolve_paired_spint_teacher(
    receipt_path: str | Path, *, loso_fold: int, seed: int
) -> Path:
    """Resolve only a receipt matching this exact fold/seed pair."""
    if isinstance(loso_fold, bool) or loso_fold not in FOLDS:
        raise ValueError("loso_fold must be in [0, 6]")
    if isinstance(seed, bool) or seed not in {42, 43, 44}:
        raise ValueError("seed must be 42, 43, or 44")
    receipt_file = _canonical_regular_file(receipt_path)
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    sources = tuple(session for fold, session in FOLDS.items() if fold != loso_fold)
    identity = (
        receipt.get("schema"),
        receipt.get("protocol_id"),
        receipt.get("phase_id"),
        receipt.get("arm"),
        receipt.get("fold"),
        receipt.get("seed"),
        receipt.get("outer_session"),
        tuple(receipt.get("source_sessions", [])),
    )
    expected = (
        "m2_post33_spint_completion_receipt_v3",
        PROTOCOL_ID,
        PHASE_ID,
        "spint",
        loso_fold,
        seed,
        FOLDS[loso_fold],
        sources,
    )
    if identity != expected:
        raise ValueError("paired SPINT receipt identity does not match T4 cell")
    records = receipt.get("selector_records")
    if not isinstance(records, list) or len(records) != 35:
        raise ValueError("paired receipt must contain 35 selector records")
    epochs = [record.get("epoch") for record in records]
    if sorted(epochs) != list(range(35)) or len(set(epochs)) != 35:
        raise ValueError("paired receipt epochs must be exactly 0..34")
    for record in records:
        value = record.get("metric_value")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("selector metric must be finite")
        if tuple(record.get("source_sessions", [])) != sources:
            raise ValueError("selector source sessions mismatch")
        totals = record.get("source_totals")
        if not isinstance(totals, Mapping) or set(totals) != set(sources):
            raise ValueError("selector source totals mismatch")
        if any(isinstance(total, bool) or not isinstance(total, int) or total <= 2 for total in totals.values()):
            raise ValueError("selector source totals must all exceed two")
        if record.get("outer_session") != FOLDS[loso_fold] or record.get("outer_total") != 0:
            raise ValueError("outer session entered paired checkpoint selection")
    selected = max(records, key=lambda row: (float(row["metric_value"]), -int(row["epoch"])))
    if receipt.get("selected_epoch") != selected["epoch"]:
        raise ValueError("receipt selected epoch differs from explicit selector")
    if receipt.get("selector_policy") != "max_finite_equal_session_mean_then_earlier_epoch":
        raise ValueError("unexpected selector policy")
    checkpoint_meta = receipt.get("checkpoint")
    config_meta = receipt.get("resolved_config")
    for label, meta in (("checkpoint", checkpoint_meta), ("resolved_config", config_meta)):
        if not isinstance(meta, Mapping):
            raise ValueError(f"missing {label} metadata")
        path = _canonical_regular_file(meta.get("canonical_path", ""))
        if path.stat().st_size != meta.get("size_bytes") or _sha256(path) != meta.get("sha256"):
            raise ValueError(f"{label} differs from paired receipt")
    checkpoint = Path(checkpoint_meta["canonical_path"])
    if checkpoint != Path(selected.get("checkpoint_path", "")):
        raise ValueError("paired checkpoint path differs from selected record")
    return checkpoint

