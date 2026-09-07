#!/usr/bin/env python3
"""Immutable CPU gate for CEBRA-inspired SUA pseudo-session training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import torch


SUA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUA_ROOT.parent
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(SUA_ROOT / "scripts"))

from mc_maze.cebra_pseudosession import (  # noqa: E402
    CebraPseudoSessionDataModule,
    SizeMatchedPseudoSessionDataset,
)
from mc_maze.multisession_datamodule import (  # noqa: E402
    Dandi688MultiSessionDataModule,
    SessionBatchSampler,
)
from mc_maze.unit_side_features import side_feature_stats_sha256  # noqa: E402
from select_gradient_free_protocol_dandi688 import load_frozen_model  # noqa: E402


SCREEN_ID = "cebra_pseudosession_sua_v1"
CONFIG = SUA_ROOT / "configs" / f"{SCREEN_ID}.json"
CONTRACT = SUA_ROOT / "docs" / "CEBRA_PSEUDOSESSION_SUA_CONTRACT_20260814.md"
MODULE = SUA_ROOT / "mc_maze" / "cebra_pseudosession.py"
TEST = SUA_ROOT / "tests" / "test_cebra_pseudosession.py"
MANIFEST = SUA_ROOT / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
A2_PREFLIGHT = SUA_ROOT / "results" / "a2_matched_subject_shift_v2" / "official_cpu_preflight.json"
A2_AGGREGATE = SUA_ROOT / "results" / "a2_matched_subject_shift_v2" / "terminal_aggregate.json"
A2_WITHIN_T4_S42 = (
    SUA_ROOT / "results" / "a2_matched_subject_shift_v2" / "within_subject_source_t4_s42.json"
)
TEACHER = (
    SUA_ROOT / "checkpoints" / "teacher_mc_maze" / "best-epoch=083-val_heldin"
    / "r2_mean=0.9061.ckpt"
)
A2_T4_S42_EPOCH5 = (
    SUA_ROOT / "checkpoints" / "a2_matched_subject_shift_v2_source_t4_dandi688_co_s42"
    / "epoch_ckpts" / "epoch_004.ckpt"
)

EXPECTED = {
    "a2_preflight": "8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd",
    "a2_aggregate": "5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc",
    "manifest": "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9",
    "teacher": "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d",
    "normalizer": "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0",
    "a2_t4_s42_epoch5": "7589bd6230da95d730e2046721f1a36a8f86d490a97666a63caa665ff1eb43a5",
    "a2_within_t4_s42": "588c4123b895878e03b6b4a18d829a8fc9bcc9f1771fc943d9cc06e72c106955",
}


class PreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def verify_readonly_pair(path: Path, expected_sha256: str) -> None:
    sidecar = Path(str(path) + ".sha256")
    require(path.is_file() and sidecar.is_file(), f"missing immutable pair: {path}")
    require(not path.is_symlink() and not sidecar.is_symlink(), f"symlink forbidden: {path}")
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"body mode drift: {path}")
    require(stat.S_IMODE(sidecar.stat().st_mode) == 0o444, f"sidecar mode drift: {sidecar}")
    observed = sha256_file(path)
    require(observed == expected_sha256, f"body SHA drift: {path}")
    require(sidecar.read_text(encoding="ascii").strip().split()[0] == observed,
            f"sidecar SHA drift: {sidecar}")


def write_immutable_pair(path: Path, payload: Mapping[str, Any]) -> str:
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not sidecar.exists(), f"refusing overwrite: {path}")
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    temporary: list[Path] = []
    published_body = False
    try:
        for content, suffix in ((body, ".body"), ((digest + "\n").encode("ascii"), ".sha")):
            descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
            temp = Path(raw)
            temporary.append(temp)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o444)
        os.link(temporary[0], path)
        published_body = True
        os.link(temporary[1], sidecar)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if published_body and not sidecar.exists():
            path.unlink(missing_ok=True)
        raise
    finally:
        for temp in temporary:
            temp.unlink(missing_ok=True)
    verify_readonly_pair(path, digest)
    return digest


def implementation_bindings() -> dict[str, str]:
    paths = {
        "config": CONFIG,
        "contract": CONTRACT,
        "pseudo_session_module": MODULE,
        "focused_test": TEST,
        "preflight": Path(__file__).resolve(),
        "parent_manifest": MANIFEST,
        "parent_a2_preflight": A2_PREFLIGHT,
        "parent_a2_aggregate": A2_AGGREGATE,
        "parent_a2_within_t4_s42": A2_WITHIN_T4_S42,
        "parent_a2_t4_s42_epoch5": A2_T4_S42_EPOCH5,
        "parent_teacher": TEACHER,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_config() -> dict[str, Any]:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(payload.get("screen_id") == SCREEN_ID, "config screen drift")
    intervention = payload.get("intervention", {})
    require(intervention.get("scope") == "source_training_only", "scope drift")
    require(intervention.get("mix_probability") == 0.5, "mix probability drift")
    require(intervention.get("contributors") == 3, "contributor count drift")
    require(intervention.get("max_behavior_residual_per_dimension_rmse") == 0.25,
            "behavior residual gate drift")
    require(payload.get("frozen_source_training", {}).get("formal_test_sessions_opened") is False,
            "formal-test contract drift")
    return payload


def datamodule(group: str) -> CebraPseudoSessionDataModule:
    return CebraPseudoSessionDataModule(
        data_dir=str(SUA_ROOT / "data" / "dandi_000688" / "sub-C"),
        task="CO", split_counts=(27, 6, 6), batch_size=32, window_size=50,
        calibration_n_trials=30, max_trial_length=100, bin_size_ms=20,
        num_workers=0, pin_memory=False, random_calibration=False, seed=42,
        max_units_exclusive=100, cache_dir=str(SUA_ROOT / "cache" / "dandi688_subc_co_v1"),
        signal_view="sua", side_feature_group=group, side_feature_pool_size=30,
        train_val_manifest_path=str(MANIFEST), pseudo_mix_probability=0.5,
        pseudo_contributor_count=3, pseudo_max_behavior_residual=0.25,
    )


def compare_plan(left: Any, right: Any) -> bool:
    if (left.anchor_session, left.anchor_start, left.mix_candidate, left.mixed) != (
        right.anchor_session, right.anchor_start, right.mix_candidate, right.mixed
    ):
        return False
    if not np.array_equal(left.final_permutation, right.final_permutation):
        return False
    return all(
        (a.session, a.start) == (b.session, b.start)
        and np.array_equal(a.unit_indices, b.unit_indices)
        and a.behavior_residual == b.behavior_residual
        for a, b in zip(left.contributors, right.contributors, strict=True)
    )


def run(smoke_examples: int | None) -> dict[str, Any]:
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    config = validate_config()
    verify_readonly_pair(A2_PREFLIGHT, EXPECTED["a2_preflight"])
    verify_readonly_pair(A2_AGGREGATE, EXPECTED["a2_aggregate"])
    verify_readonly_pair(A2_WITHIN_T4_S42, EXPECTED["a2_within_t4_s42"])
    require(sha256_file(MANIFEST) == EXPECTED["manifest"], "parent manifest drift")
    require(sha256_file(TEACHER) == EXPECTED["teacher"], "teacher drift")
    require(sha256_file(A2_T4_S42_EPOCH5) == EXPECTED["a2_t4_s42_epoch5"],
            "A2 T4 seed42 logical-epoch5 checkpoint drift")
    receipt = json.loads(A2_WITHIN_T4_S42.read_text(encoding="utf-8"))
    require(receipt["source_run"]["source_checkpoint_sha256_bundle"]["5"] ==
            EXPECTED["a2_t4_s42_epoch5"], "parent receipt/checkpoint mismatch")
    bindings_before = implementation_bindings()

    t4 = datamodule("t4")
    t4.setup("fit")
    manifest_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))["session_splits"]
    require(t4.session_splits == manifest_payload, "source/dev/formal name roster drift")
    require(len(t4.session_splits["train"]) == 27 and len(t4.session_splits["val"]) == 6,
            "source/dev roster size drift")
    require(len(t4.session_splits["test"]) == 6 and len(t4.session_files["test"]) == 0,
            "formal test paths were resolved")
    assert t4._ordinary_train_dataset is not None
    ordinary = t4._ordinary_train_dataset

    p0 = SizeMatchedPseudoSessionDataset(
        ordinary, seed=42, mix_probability=0.0, contributor_count=3,
        max_behavior_residual=0.25,
    )
    parity_indices = (0, 1, 17, 1000, len(ordinary) // 2, len(ordinary) - 1)
    p0_item_parity = []
    for index in parity_indices:
        base_row, pseudo_row = ordinary[index], p0[index]
        p0_item_parity.append(
            base_row[3] == pseudo_row[3]
            and all(torch.equal(a, b) for a, b in zip(
                base_row[:3] + base_row[4:], pseudo_row[:3] + pseudo_row[4:]
            ))
        )
    require(all(p0_item_parity), "real p=0 item parity failed")
    base_sampler = SessionBatchSampler(ordinary, batch_size=32, shuffle=True, seed=42)
    p0_sampler = SessionBatchSampler(p0, batch_size=32, shuffle=True, seed=42)
    require(next(iter(base_sampler)) == next(iter(p0_sampler)), "p=0 sampler parity failed")

    assert isinstance(t4.train_dataset, SizeMatchedPseudoSessionDataset)
    examples = smoke_examples if smoke_examples is not None else None
    full_audit = t4.pseudo_session_manifest(max_examples=examples)
    require(0.47 <= full_audit["mixed_fraction_observed"] <= 0.51,
            "accepted mix fraction outside frozen CPU gate")
    require(full_audit["rejected_candidate_examples"] / full_audit["examples_audited"] <= 0.01,
            "behavior fallback exceeds one percent")
    require(full_audit["accepted_endpoint_residual"]["max"] <= 0.25,
            "accepted residual exceeds threshold")
    require(full_audit["accepted_endpoint_residual"]["p99"] <= 0.10,
            "accepted residual P99 exceeds source audit gate")
    if smoke_examples is None:
        require(all(count > 0 for count in full_audit["accepted_donor_uses"].values()),
                "at least one source session never contributes accepted units")

    z4 = datamodule("z4")
    z4.setup("fit")
    require(len(z4.session_files["test"]) == 0, "Z4 resolved formal test paths")
    mean_t4, std_t4 = t4._get_side_feature_stats()
    mean_z4, std_z4 = z4._get_side_feature_stats()
    require(np.array_equal(mean_t4, mean_z4) and np.array_equal(std_t4, std_z4),
            "T4/Z4 source normalizer bit drift")
    normalizer_sha = side_feature_stats_sha256(mean_t4, std_t4)
    require(normalizer_sha == EXPECTED["normalizer"], "A2 T4 normalizer drift")
    require(all(np.count_nonzero(row.side_features) == 0
                for row in z4.train_dataset.sessions.values()), "Z4 source rows are not exact zero")
    parity_count = min(10_000, len(t4.train_dataset))
    require(all(compare_plan(t4.train_dataset.plan_for_index(index),
                             z4.train_dataset.plan_for_index(index))
                for index in range(parity_count)), "T4/Z4 mixing-plan drift")
    z4_batch = next(iter(z4.train_dataloader()))
    require(torch.count_nonzero(z4_batch[4]).item() == 0, "mixed Z4 batch is not exact zero")

    production_sampler = SessionBatchSampler(
        t4.train_dataset, batch_size=4, shuffle=False, seed=42
    )
    production_indices = next(iter(production_sampler))
    require(any(t4.train_dataset.plan_for_index(index).mixed for index in production_indices),
            "production smoke batch contains no pseudo-session example")
    t4_loader = torch.utils.data.DataLoader(
        t4.train_dataset,
        batch_sampler=[production_indices],
        num_workers=0,
    )
    batch = next(iter(t4_loader))
    model = load_frozen_model(A2_T4_S42_EPOCH5, TEACHER, "B3S", torch.device("cpu"))
    assert model.student is not None and model.teacher is not None
    for parameter in model.student.parameters():
        parameter.requires_grad_(True)
    for parameter in model.teacher.parameters():
        parameter.requires_grad_(False)
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model.training_step(batch, 0)
    require(bool(torch.isfinite(loss)), "production one-batch loss is non-finite")
    loss.backward()
    trainable = [parameter for parameter in model.student.parameters() if parameter.requires_grad]
    finite_gradients = sum(
        int(parameter.grad is not None and torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )
    require(finite_gradients > 0, "production batch produced no finite student gradients")
    require(all(parameter.grad is None for parameter in model.teacher.parameters()),
            "teacher received source-training gradients")

    bindings_after = implementation_bindings()
    require(bindings_before == bindings_after, "implementation drifted during CPU preflight")
    return {
        "schema_version": 1,
        "screen_id": SCREEN_ID,
        "status": "CPU_PREFLIGHT_PASSED__GPU_NOT_LAUNCHED" if smoke_examples is None
                  else "CPU_SMOKE_PASSED__NO_RECEIPT",
        "mode": "full" if smoke_examples is None else "smoke",
        "formal_subc_test_nwb_opened": False,
        "cuda_used": False,
        "training_started": False,
        "parent": {
            "a2_preflight_sha256": EXPECTED["a2_preflight"],
            "a2_terminal_aggregate_sha256": EXPECTED["a2_aggregate"],
            "a2_t4_s42_epoch5_sha256": EXPECTED["a2_t4_s42_epoch5"],
            "t4_normalizer_sha256": normalizer_sha,
        },
        "roster": {
            "train": t4.session_splits["train"],
            "development_val": t4.session_splits["val"],
            "formal_test_names_only": t4.session_splits["test"],
            "formal_test_paths_resolved": 0,
        },
        "p0_parity": {
            "indices": list(parity_indices),
            "item_bit_parity": p0_item_parity,
            "first_sampler_batch_equal": True,
        },
        "pseudo_session_audit": full_audit,
        "t4_z4": {
            "plan_prefix_compared": parity_count,
            "plan_prefix_exact": True,
            "source_normalizer_bit_equal": True,
            "z4_source_rows_exact_zero": True,
            "z4_mixed_batch_exact_zero": True,
        },
        "production_one_batch": {
            "batch_shapes": [list(value.shape) if hasattr(value, "shape") else len(value)
                             for value in batch],
            "dataset_indices": list(production_indices),
            "mixed_example_count": sum(
                int(t4.train_dataset.plan_for_index(index).mixed)
                for index in production_indices
            ),
            "unique_anchor_sessions": sorted(set(batch[3])),
            "loss": float(loss.detach()),
            "finite_student_gradient_parameter_tensors": finite_gradients,
            "trainable_student_parameter_tensors": len(trainable),
            "teacher_gradient_parameter_tensors": 0,
        },
        "implementation_bindings": bindings_after,
        "implementation_bindings_sha256": canonical_sha256(bindings_after),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-examples", type=int)
    args = parser.parse_args()
    if args.smoke_examples is not None:
        require(args.smoke_examples >= 10_000, "smoke must cover at least 10,000 examples")
        require(args.output is None, "smoke mode never writes a receipt")
    else:
        require(args.output is not None, "full preflight requires --output")
    payload = run(args.smoke_examples)
    if args.output is None:
        print(json.dumps(payload, sort_keys=True))
        return
    digest = write_immutable_pair(args.output, payload)
    print(json.dumps({"status": payload["status"], "path": str(args.output.resolve()),
                      "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
