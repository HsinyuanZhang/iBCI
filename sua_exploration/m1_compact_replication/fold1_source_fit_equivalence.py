#!/usr/bin/env python3
"""CPU-only equivalence audit for the fold-1 source-only-fit data path.

The audit compares the strict v2 source-only class with the pre-existing
Version-B train construction for B3S-Zero4.  The legacy class is allowed to
run only with its left-out ``prepare_session_data`` call replaced by a guard
that returns a source record; no target NWB values are opened.  Thus the
comparison covers the three-source train dataset, sampler order/counts and
the five-tuple Zero4 side contract without making a target metric available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import hydra
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_SOURCE_FIT_EQUIVALENCE_v1.json"
V2_RECEIPT = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_STAGED_RECEIPT_v2.json"
SOURCE_ONLY_TARGET = "src.data.m1_version_b_source_loso_datamodule.M1VersionBSourceOnlyFitDataModule"
LEGACY_TARGET = "src.data.m1_version_b_source_loso_datamodule.M1VersionBSourceLOSODataModule"
SOURCES = ("ses-20120924", "ses-20120927", "ses-20120928")
TARGET = "ses-20120926"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _value_digest(value: Any) -> str:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return hashlib.sha256(
            b"ndarray" + str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes()
        ).hexdigest()
    if isinstance(value, (list, tuple)):
        return canonical([_value_digest(item) for item in value])
    if isinstance(value, str):
        return hashlib.sha256(b"str" + value.encode()).hexdigest()
    if value is None:
        return hashlib.sha256(b"none").hexdigest()
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _compose_config() -> Any:
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "streaming_calibration_exp/configs")):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                "experiment=m1_version_b_c0",
                "data.loso_fold=1",
                "data.source_session_names=[ses-20120924,ses-20120927,ses-20120928]",
                "seed=42",
                "test=false",
                "optimized_metric=null",
                f"data._target_={SOURCE_ONLY_TARGET}",
            ],
        )
    # Compose outside a Hydra ``@hydra.main`` run has no runtime resolver.
    # Resolve only the data subtree after binding the canonical repository
    # root; no output/checkpoint interpolation is needed by this audit.
    cfg.paths.root_dir = str(ROOT)
    cfg.seed = 42
    OmegaConf.resolve(cfg.data)
    return cfg


def _build_pair() -> tuple[Any, Any, list[str]]:
    from hydra.utils import instantiate
    from src.data.m1_version_b_source_loso_datamodule import (  # noqa: PLC0415
        M1VersionBSourceLOSODataModule,
    )

    cfg = _compose_config()
    source_only = instantiate(cfg.data)
    legacy_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    legacy_cfg._target_ = LEGACY_TARGET
    legacy = instantiate(legacy_cfg)
    target_calls: list[str] = []
    source_cache: dict[str, dict[str, Any]] = {}
    original = legacy.prepare_session_data

    def guarded(path, *args, **kwargs):
        session = str(path).split("_ses-")[-1].split("_behavior")[0]
        session = f"ses-{session}"
        if session == TARGET:
            # The old class still invokes this target preparation.  Return a
            # source record instead of opening the target NWB so this audit
            # cannot expose target values.
            target_calls.append(str(path))
            if not source_cache:
                raise RuntimeError("target guard ran before any source record")
            return next(iter(source_cache.values()))
        record = original(path, *args, **kwargs)
        source_cache[session] = record
        return record

    legacy.prepare_session_data = guarded
    source_only.setup("fit")
    legacy.setup("fit")
    return source_only, legacy, target_calls


def build() -> dict[str, Any]:
    from src.data.m1_version_b_source_loso_datamodule import M1VersionBSourceLOSODataModule

    need = lambda condition, message: (_ for _ in ()).throw(RuntimeError(message)) if not condition else None
    source_only, legacy, target_calls = _build_pair()
    need(not hasattr(source_only, "target_path"), "source-only fit resolved target_path")
    need(target_calls == [str((ROOT / "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys.nwb").resolve())], "legacy target guard call mismatch")
    need(tuple(source_only.train_session_names) == SOURCES and tuple(legacy.train_session_names) == SOURCES, "source session list mismatch")
    need(len(source_only.train_dataset) == len(legacy.train_dataset), "train dataset length mismatch")
    need(M1VersionBSourceLOSODataModule._sampler_batch_counts(source_only.train_batch_sampler) == M1VersionBSourceLOSODataModule._sampler_batch_counts(legacy.train_batch_sampler), "per-session batch counts mismatch")
    source_sampler_sha, source_windows = M1VersionBSourceLOSODataModule._sampler_sha256(source_only.train_batch_sampler)
    legacy_sampler_sha, legacy_windows = M1VersionBSourceLOSODataModule._sampler_sha256(legacy.train_batch_sampler)
    need(source_sampler_sha == legacy_sampler_sha and source_windows == legacy_windows, "sampler order mismatch")
    representative = {}
    for index in (0, len(source_only.train_dataset) // 2, len(source_only.train_dataset) - 1):
        left = source_only.train_dataset[index]
        right = legacy.train_dataset[index]
        need(len(left) == len(right) == 5, "B3S Zero4 tuple arity mismatch")
        need(all(_value_digest(a) == _value_digest(b) for a, b in zip(left[:4], right[:4])), f"representative source batch mismatch at {index}")
        need(np.asarray(left[4]).shape[-1] == 4 and np.count_nonzero(left[4]) == 0, "source-only Zero4 side is not exact zero")
        need(np.array_equal(left[4], right[4]), f"Zero4 side mismatch at {index}")
        representative[str(index)] = {
            "source_digest": _value_digest(left),
            "legacy_digest": _value_digest(right),
            "side_exact_zero": bool(np.count_nonzero(left[4]) == 0),
        }
    need(V2_RECEIPT.is_file(), "v2 staged receipt missing")
    return {
        "schema": "m1_compact_b3s_f1_s42_source_fit_equivalence_v1",
        "status": "PASS_M1_COMPACT_B3S_F1_S42_SOURCE_FIT_EQUIVALENT",
        "parent_v2_receipt": {"path": str(V2_RECEIPT.resolve()), "sha256": sha(V2_RECEIPT)},
        "scope": {"task": "m1", "fold": 1, "seed": 42, "arm": "b3s_zero4", "source_sessions": list(SOURCES), "target_session": TARGET, "target_values_opened_by_audit": False, "target_values_used": False},
        "classes": {"source_only_fit": SOURCE_ONLY_TARGET, "legacy_reference": LEGACY_TARGET},
        "train_dataset": {"source_only_len": len(source_only.train_dataset), "legacy_len": len(legacy.train_dataset), "equal": True},
        "sampler": {"source_batch_counts": M1VersionBSourceLOSODataModule._sampler_batch_counts(source_only.train_batch_sampler), "legacy_batch_counts": M1VersionBSourceLOSODataModule._sampler_batch_counts(legacy.train_batch_sampler), "source_sha256": source_sampler_sha, "legacy_sha256": legacy_sampler_sha, "source_windows": source_windows, "legacy_windows": legacy_windows, "equal": True},
        "representative_samples": representative,
        "source_only_guard": {"target_path_attribute_absent": True, "source_prepare_sessions": list(SOURCES), "target_prepare_calls_in_legacy_guarded_reference": target_calls, "target_nwb_opened": False},
    }


def write_once(body: dict[str, Any], output: Path = OUT) -> dict[str, Any]:
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("canonical_content_sha256") != canonical({k: v for k, v in existing.items() if k != "canonical_content_sha256"}):
            raise RuntimeError("existing equivalence receipt canonical drift")
        if canonical(body) != existing["canonical_content_sha256"]:
            raise RuntimeError("existing equivalence receipt differs; refusing overwrite")
        return existing
    value = dict(body)
    value["canonical_content_sha256"] = canonical(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=str(output.parent))
    temporary = Path(name)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.chmod(0o444)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    body = write_once(build(), args.output.resolve())
    print(json.dumps({"path": str(args.output.resolve()), "sha256": sha(args.output.resolve()), "status": body["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
