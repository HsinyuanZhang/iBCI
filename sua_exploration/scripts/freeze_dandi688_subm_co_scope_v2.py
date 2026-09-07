#!/usr/bin/env python3
"""Freeze the v2 sub-M T4-vs-TS4 mechanism endpoint from metadata only.

The script replays and byte-verifies the append-only v1 DANDI scope, then binds
matched shared-T4/shared-TS4 terminal FP32 artifacts and a fully prespecified
external endpoint.  It never downloads or opens NWB content, imports a model,
computes a score, or uses a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import freeze_dandi688_subm_co_scope as v1


REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPE_ID = "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "sua_exploration/manifests/"
    "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json"
)

V1_BINDINGS = {
    "generator": {
        "path": "sua_exploration/scripts/freeze_dandi688_subm_co_scope.py",
        "sha256": "23142a547e6c49cbf6f0e0d723e7c7500b4c8fbc9309424702c21f5522f5a971",
    },
    "document": {
        "path": "sua_exploration/docs/DANDI_000688_SUBM_CO_SCOPE_FREEZE_PREFLIGHT.md",
        "sha256": "28980aebae5b10c7116d47b62a341548dda40636c49b6d8b2519cb9c1b1b8e83",
    },
    "manifest": {
        "path": (
            "sua_exploration/manifests/"
            "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v1.json"
        ),
        "sha256": "65a38eee5f5b13029120978884c8d5dc8c00f8001738a322a114be75a5e1b50c",
    },
}

PRELAUNCH_BINDING = {
    "path": (
        "sua_exploration/results/"
        "t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/receipt.json"
    ),
    "sha256": "8b17c19515fa0e6cc122233fd20cf7e87a616287fa247e5eacb136d6a56c9d85",
}

V2_DOCUMENT_BINDING = {
    "path": "sua_exploration/docs/DANDI_000688_SUBM_CO_MECHANISM_ENDPOINT_V2.md",
    "sha256": "274eea514981ee39968e1b382f839f1bedc6a525aa014a1eb505f5a62628a1e9",
}

CONTROL_CODE_BINDINGS = {
    "side_feature_implementation": {
        "path": "sua_exploration/mc_maze/unit_side_features.py",
        "sha256": "059faefcd766dfc8e25253d9ded2b619a46dea408e6f00a30cfa5b2ecd185ab6",
    },
    "paired_training": {
        "path": "sua_exploration/scripts/train_paired_view_c1_dandi688.py",
        "sha256": "c9284dbb6baef3b858a50b8fe0f66e21ff32292f2dd5586186904a39c7c796a5",
    },
    "guarded_cell_runner": {
        "path": "sua_exploration/scripts/run_t4_paired_view_c1_one_cell.py",
        "sha256": "13de1cf27950611fee7414c4d4aeee2f6790b520a7b6f3108049f55694bbcaa9",
    },
}

ARM_BINDINGS: dict[str, dict[int, dict[str, str | int]]] = {
    "shared_t4": {
        42: {
            "run_metadata_sha256": "98550528e91e6a7c5f637a2acc5112d53ae12412d6077804c30ac950eabfd000",
            "closure_manifest_sha256": "a292548fecbcd169e632039d0d11408eed852f805506ee3b2a477b51e2e6885d",
            "checkpoint_sha256": "ab9df840a07d7aeb6cc417bb684f1f5e0265d50f98168400ac915647cdfd7b9f",
            "checkpoint_bytes": 64_768_898,
        },
        43: {
            "run_metadata_sha256": "f591b4425ce6d39865d2599e96e7211b382fc571c2a7f0b4bfc332577376b83b",
            "closure_manifest_sha256": "abad2b894fca7e6fba88eb289b3cf4a6dbf2e977edb44c179e0fc89acfe57ed2",
            "checkpoint_sha256": "05c05b3ab82a2fba43c55aca523248982a954faf5f0363a0235a29d64e57ab22",
            "checkpoint_bytes": 64_768_898,
        },
        44: {
            "run_metadata_sha256": "d3e84a1c248a0d2fc97dd2d009f4e31ec7800de4f0358161fab23d7c689807d7",
            "closure_manifest_sha256": "7882ba4166d02e36a07696589b63c7828f9d822b4b88a5a2088851137529589d",
            "checkpoint_sha256": "a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6",
            "checkpoint_bytes": 64_769_167,
        },
    },
    "shared_ts4": {
        42: {
            "run_metadata_sha256": "3708ad1db91ccbff2d5cee87e5da523a01d364a9bb9f5d71227fa53f4ddcd9be",
            "closure_manifest_sha256": "9d9d298a3c8268f0f9b374d8ae7f7a7409012b1e7fe9e3abf0772aeb9a40d29f",
            "checkpoint_sha256": "a21da5a72a991bd2665af50572a4132998ac79d7f801879048553efcdc8281b2",
            "checkpoint_bytes": 64_768_898,
        },
        43: {
            "run_metadata_sha256": "a827b0ec6f0c74394484d07c898406055a779140d6d6bdf3a3c94221874214ba",
            "closure_manifest_sha256": "ea1982432e605ac62d80f64be94950ee75cc351ac94807ce7ee9b906dca3d272",
            "checkpoint_sha256": "c8dd22dfadb2bc11555fc21abe464316886d221e2dbcd71bf20a6bffe9cb158e",
            "checkpoint_bytes": 64_768_898,
        },
        44: {
            "run_metadata_sha256": "788c3436b4cabbc6076b42a6566ae555cfdabe011db53d93c798e1703892a906",
            "closure_manifest_sha256": "23c795902300dfcb1d8a7d523613a3ff193d8bb657fddcee2a77d0c9522c5a1c",
            "checkpoint_sha256": "a2d877ac81a4e553e5221c54e465db26eba8592888b8cb5339e9dfc4acd66ced",
            "checkpoint_bytes": 64_769_167,
        },
    },
}

COMMON_SOURCE_SHA256 = {
    "train_val_manifest": "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9",
    "data_manifest": "1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb",
    "teacher": "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d",
}

NORMALIZER_SHA256 = {
    "sua": "ac5156097864110685e0b2fbfe314edcb747e69dc821c10451984a089be8a7a7",
    "pseudo_mua": "92470ad14062af6cb998e06e7696b94bfdfd20ac5e415615302a5dddc7098fcc",
}

EXPECTED_ALLOWED_METADATA_DIFFS = {
    "completed_at",
    "created_at",
    "output_dir",
    "training.epoch_checkpoints_dir",
    "view_configs.pseudo_mua.side_features.group",
    "view_configs.pseudo_mua.side_features.permutation_seed",
    "view_configs.sua.side_features.group",
    "view_configs.sua.side_features.permutation_seed",
    *(f"epoch_checkpoints.{index}" for index in range(12)),
}


class FreezeV2Error(RuntimeError):
    """Fail-closed v2 candidate-freeze error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeV2Error(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_file(binding: dict[str, Any]) -> dict[str, Any]:
    relative = str(binding["path"])
    expected = str(binding["sha256"])
    path = REPO_ROOT / relative
    require(path.is_file(), f"required file missing: {relative}")
    observed = sha256_file(path)
    require(observed == expected, f"file hash drift: {relative}: expected {expected}, got {observed}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": observed}


def load_json(relative: str) -> dict[str, Any]:
    path = REPO_ROOT / relative
    require(path.is_file(), f"required JSON missing: {relative}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeV2Error(f"cannot parse {relative}: {exc}") from exc
    require(isinstance(payload, dict), f"JSON root must be an object: {relative}")
    return payload


def verify_v1_replay(timeout_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    bound = {name: bind_file(row) for name, row in V1_BINDINGS.items()}
    on_disk = load_json(V1_BINDINGS["manifest"]["path"])
    require(on_disk.get("scope_id") == v1.SCOPE_ID, "v1 scope ID drift")
    require(on_disk.get("status") == "scope_frozen_metadata_only_external_runner_blocked", "v1 status drift")
    regenerated = v1.build_manifest(timeout_seconds)
    regenerated_bytes = v1.canonical_bytes(regenerated)
    manifest_path = REPO_ROOT / V1_BINDINGS["manifest"]["path"]
    require(manifest_path.read_bytes() == regenerated_bytes, "v1 is not byte-reproducible now")
    require(len(regenerated.get("selected_assets", [])) == 22, "v1 selected-asset count drift")
    require(
        sum(int(row["size"]) for row in regenerated["selected_assets"]) == 2_312_360_648,
        "v1 selected-asset byte total drift",
    )
    return regenerated, bound


def flatten_scalars(value: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from flatten_scalars(value[key], prefix + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten_scalars(item, prefix + (str(index),))
    else:
        yield ".".join(prefix), value


def arm_paths(arm: str, seed: int) -> dict[str, str]:
    closure_root = (
        "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/"
        f"closure/{arm}_s{seed}"
    )
    checkpoint_root = (
        "sua_exploration/checkpoints/"
        "t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_"
        f"{arm}_s{seed}"
    )
    return {
        "run_metadata": f"{closure_root}/run_metadata.json",
        "closure_manifest": f"{closure_root}/closure_manifest.json",
        "checkpoint": f"{checkpoint_root}/epoch_ckpts/epoch_011.ckpt",
    }


def validate_arm_metadata(arm: str, seed: int, binding: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = arm_paths(arm, seed)
    run_file = bind_file({"path": paths["run_metadata"], "sha256": binding["run_metadata_sha256"]})
    closure_file = bind_file(
        {"path": paths["closure_manifest"], "sha256": binding["closure_manifest_sha256"]}
    )
    checkpoint_file = bind_file({"path": paths["checkpoint"], "sha256": binding["checkpoint_sha256"]})
    require(
        checkpoint_file["bytes"] == binding["checkpoint_bytes"],
        f"checkpoint size drift: {arm}/seed{seed}",
    )

    metadata = load_json(paths["run_metadata"])
    require(metadata.get("status") == "completed", f"run is not completed: {arm}/seed{seed}")
    require(metadata.get("seed") == seed, f"seed mismatch: {arm}/seed{seed}")
    require(metadata.get("training_kind") == "shared_paired_view", f"training kind drift: {arm}/seed{seed}")
    require(metadata.get("held_out_test_evaluated") is False, f"held-out flag drift: {arm}/seed{seed}")
    require(metadata.get("formal_sua_files_opened") is False, f"formal flag drift: {arm}/seed{seed}")
    require(metadata.get("train_val_manifest_sha256") == COMMON_SOURCE_SHA256["train_val_manifest"], "train/val drift")
    require(metadata.get("data_manifest_sha256") == COMMON_SOURCE_SHA256["data_manifest"], "data manifest drift")
    require(metadata.get("teacher_sha256") == COMMON_SOURCE_SHA256["teacher"], "teacher drift")

    objective = metadata.get("paired_objective")
    require(isinstance(objective, dict), f"objective missing: {arm}/seed{seed}")
    require(objective.get("lambda_consistency") == 0, f"lambda drift: {arm}/seed{seed}")
    require(objective.get("sua_task_loss_weight") == 0.5, f"SUA weight drift: {arm}/seed{seed}")
    require(objective.get("pseudo_mua_task_loss_weight") == 0.5, f"pseudo-MUA weight drift: {arm}/seed{seed}")
    require(objective.get("view_specific_heads") is False, f"head sharing drift: {arm}/seed{seed}")

    expected_group = "t4" if arm == "shared_t4" else "ts4"
    expected_permutation = None if arm == "shared_t4" else seed
    views = metadata.get("view_configs")
    require(isinstance(views, dict), f"view configs missing: {arm}/seed{seed}")
    for view in ("sua", "pseudo_mua"):
        config = views.get(view)
        require(isinstance(config, dict) and config.get("signal_view") == view, f"view drift: {arm}/{view}/seed{seed}")
        side = config.get("side_features")
        require(isinstance(side, dict), f"side features missing: {arm}/{view}/seed{seed}")
        require(side.get("group") == expected_group, f"descriptor group drift: {arm}/{view}/seed{seed}")
        require(side.get("permutation_seed") == expected_permutation, f"permutation drift: {arm}/{view}/seed{seed}")
        require(side.get("pool_size") == 50 and side.get("side_dim") == 4, f"descriptor shape drift: {arm}/{view}/seed{seed}")
        require(side.get("normalization_scope") == "source_train_27_only", f"normalizer scope drift: {arm}/{view}/seed{seed}")
        require(side.get("normalization_sha256") == NORMALIZER_SHA256[view], f"normalizer hash drift: {arm}/{view}/seed{seed}")

    epochs = metadata.get("epoch_checkpoints")
    require(isinstance(epochs, list) and len(epochs) == 12, f"epoch list drift: {arm}/seed{seed}")
    require(Path(epochs[-1]).name == "epoch_011.ckpt", f"terminal epoch drift: {arm}/seed{seed}")

    closure = load_json(paths["closure_manifest"])
    require(closure.get("cell") == arm, f"closure arm drift: {arm}/seed{seed}")
    require(closure.get("seed") == seed, f"closure seed drift: {arm}/seed{seed}")
    require(closure.get("status") == "completed", f"closure status drift: {arm}/seed{seed}")
    closure_files = closure.get("files")
    require(isinstance(closure_files, list), f"closure file ledger missing: {arm}/seed{seed}")
    run_rows = [row for row in closure_files if isinstance(row, dict) and row.get("copy_relative", "").endswith("run_metadata.json")]
    require(len(run_rows) == 1, f"closure run-metadata row count drift: {arm}/seed{seed}")
    require(run_rows[0].get("sha256") == binding["run_metadata_sha256"], f"closure run hash drift: {arm}/seed{seed}")

    return metadata, {
        "arm": arm,
        "seed": seed,
        "run_metadata": run_file,
        "closure_manifest": closure_file,
        "terminal_checkpoint": checkpoint_file,
        "terminal_epoch_one_based": 12,
        "checkpoint_filename_zero_based": "epoch_011.ckpt",
        "descriptor_group": expected_group,
        "permutation_seed": expected_permutation,
    }


def normalize_arm_metadata(metadata: dict[str, Any], arm: str) -> dict[str, Any]:
    normalized = json.loads(json.dumps(metadata))
    normalized["created_at"] = "<execution-time>"
    normalized["completed_at"] = "<execution-time>"
    normalized["output_dir"] = "<arm-output-dir>"
    normalized["training"]["epoch_checkpoints_dir"] = "<arm-epoch-dir>"
    normalized["epoch_checkpoints"] = [f"<arm-epoch-{index:03d}>" for index in range(12)]
    for view in ("sua", "pseudo_mua"):
        side = normalized["view_configs"][view]["side_features"]
        side["group"] = "<attachment-arm>"
        side["permutation_seed"] = "<attachment-seed>"
    return normalized


def compare_matched_pair(seed: int, t4: dict[str, Any], ts4: dict[str, Any]) -> dict[str, Any]:
    flat_t4 = dict(flatten_scalars(t4))
    flat_ts4 = dict(flatten_scalars(ts4))
    differences = {path for path in set(flat_t4) | set(flat_ts4) if flat_t4.get(path) != flat_ts4.get(path)}
    require(
        differences == EXPECTED_ALLOWED_METADATA_DIFFS,
        f"unmatched T4/TS4 metadata paths for seed {seed}: {sorted(differences ^ EXPECTED_ALLOWED_METADATA_DIFFS)}",
    )
    for index in range(12):
        left = str(flat_t4[f"epoch_checkpoints.{index}"])
        right = str(flat_ts4[f"epoch_checkpoints.{index}"])
        require(right.replace("shared_ts4", "shared_t4") == left, f"epoch path mismatch at seed {seed}/epoch{index}")
    require(str(flat_ts4["output_dir"]).replace("shared_ts4", "shared_t4") == flat_t4["output_dir"], "output path mismatch")
    require(
        str(flat_ts4["training.epoch_checkpoints_dir"]).replace("shared_ts4", "shared_t4")
        == flat_t4["training.epoch_checkpoints_dir"],
        "checkpoint-directory mismatch",
    )
    for view in ("sua", "pseudo_mua"):
        prefix = f"view_configs.{view}.side_features"
        require(flat_t4[f"{prefix}.group"] == "t4" and flat_ts4[f"{prefix}.group"] == "ts4", "group mismatch")
        require(flat_t4[f"{prefix}.permutation_seed"] is None, "T4 permutation must be null")
        require(flat_ts4[f"{prefix}.permutation_seed"] == seed, "TS4 permutation must equal training seed")
    canonical_t4 = normalize_arm_metadata(t4, "shared_t4")
    canonical_ts4 = normalize_arm_metadata(ts4, "shared_ts4")
    require(canonical_t4 == canonical_ts4, f"canonical T4/TS4 metadata mismatch for seed {seed}")
    canonical_sha = hashlib.sha256(
        json.dumps(canonical_t4, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "seed": seed,
        "strict_match": True,
        "canonical_common_metadata_sha256": canonical_sha,
        "allowed_differences": sorted(differences),
        "scientific_difference": (
            "shared_t4 keeps each normalized four-value T4 row attached to its source unit/channel; "
            "shared_ts4 applies the frozen seed-specific complete-row permutation"
        ),
    }


def verify_control_code() -> dict[str, Any]:
    prelaunch = bind_file(PRELAUNCH_BINDING)
    receipt = load_json(PRELAUNCH_BINDING["path"])
    source_map = receipt.get("source_map")
    require(isinstance(source_map, dict), "prelaunch source map missing")
    code: dict[str, Any] = {}
    for name, binding in CONTROL_CODE_BINDINGS.items():
        row = bind_file(binding)
        source_row = source_map.get(binding["path"])
        require(isinstance(source_row, dict), f"prelaunch source row missing: {binding['path']}")
        require(source_row.get("sha256") == binding["sha256"], f"prelaunch source hash drift: {binding['path']}")
        code[name] = row
    return {"prelaunch_receipt": prelaunch, "code": code}


def endpoint_protocol() -> dict[str, Any]:
    return {
        "scope": {
            "claim": "same-Dandiset cross-animal external-subject confirmation",
            "not_claimed": ["independent dataset", "independent laboratory", "native threshold-crossing MUA"],
            "pseudo_mua_definition": (
                "deterministic within-session electrode pooling of the same sorted-SUA activity; "
                "pooled trial rates are formed before T4 is refit"
            ),
        },
        "eligible_session_symbol": "N",
        "ledger_count": 22,
        "minimum_eligible_sessions": 6,
        "pre_score_rule": (
            "if score-blind compatibility yields N < 6, return ENDPOINT_NO_GO_INSUFFICIENT_ELIGIBLE_SESSIONS "
            "before loading a checkpoint or computing any prediction"
        ),
        "cohort_rule": (
            "one common eligible-session set is frozen before scores and used for both arms, both views, "
            "and all three seeds; all 22 assets remain in the compatibility ledger"
        ),
        "r2_definition": {
            "unit": "one eligible session x one view x one arm x one seed",
            "query": "all valid windows strictly after chronological rewarded trial 50",
            "prediction_target": (
                "last behavior bin of each 50-bin query window; decoder output divided exactly once by "
                "the frozen BEHAVIOR_SCALING_FACTOR=5.0"
            ),
            "reference_implementation": (
                "torchmetrics.regression.R2Score(multioutput='variance_weighted'), updated over all "
                "query-window predictions and targets from that session, then computed once"
            ),
            "formula": (
                "variance-weighted two-output R2 = 1 - sum_d sum_i(y_id-yhat_id)^2 / "
                "sum_d sum_i(y_id-mean_i(y_id))^2, accumulated over all query windows in that session"
            ),
            "undefined_policy": "any undefined/nonfinite arm score fails the endpoint; never drop that session",
        },
        "paired_delta": "delta[view,session,seed] = R2(shared_t4) - R2(shared_ts4)",
        "aggregation_order": [
            "compute one R2 per session/view/arm/seed",
            "pair T4-TS4 within the identical session/view/seed",
            "for each seed, average its N session deltas with equal session weight",
            "grand paired mean is the equal-weight mean of the three seed means",
            "for each session, cross-seed mean delta is the equal-weight mean over seeds 42/43/44",
            "shared-T4 absolute seed mean is the equal-weight mean over N sessions; its grand mean is the equal-weight mean over seeds",
        ],
        "endpoints": {
            "primary": "SUA paired shared_t4 - shared_ts4",
            "key_secondary": "pseudo-MUA paired shared_t4 - shared_ts4",
            "absolute": ["shared_t4 SUA", "shared_t4 pseudo-MUA"],
            "overall_mechanism_pass": (
                "primary and key-secondary must each pass every paired and absolute gate; "
                "pseudo-MUA cannot rescue a failed SUA primary"
            ),
        },
        "gates_applied_separately_to_each_view": {
            "mean_paired_delta": ">= +0.03 R2",
            "seed_consistency": "3/3 equal-session-weight seed mean deltas are strictly > 0",
            "session_consistency": "at least ceil(0.75*N) cross-seed session mean deltas are strictly > 0",
            "bootstrap": "hierarchical session x seed percentile-bootstrap 95% lower bound is strictly > 0",
            "absolute_shared_t4": "grand mean R2 > 0 and 3/3 shared-T4 seed mean R2 values are strictly > 0",
        },
        "bootstrap": {
            "replicates": 100_000,
            "rng": "numpy.random.Generator(numpy.random.PCG64(seed))",
            "seed": 68_820_260_805,
            "resampling": (
                "for each replicate sample N session indices with replacement; for each sampled session "
                "independently sample three seed indices with replacement from [42,43,44]; average the "
                "resulting N*3 paired deltas"
            ),
            "interval": "two-sided percentile 95% interval",
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
            "quantile_method": "linear",
            "pass": "lower_quantile_value > 0",
        },
        "n_dependent_session_threshold": "ceil(0.75*N)",
        "n_dependent_session_threshold_examples": {
            str(n): math.ceil(0.75 * n) for n in range(6, 23)
        },
        "reporting": [
            "N and all 22 compatibility dispositions",
            "all session x seed R2 values for both arms and views",
            "all paired deltas, three seed means, N session means, grand means, and bootstrap interval",
            "absolute shared-T4 results per view; shared-TS4 absolutes as control transparency",
            "TS4 realized row permutation and whether it is identity for every session/view/seed; identity is reported, not excluded",
        ],
    }


def build_manifest(timeout_seconds: float) -> dict[str, Any]:
    v1_payload, v1_files = verify_v1_replay(timeout_seconds)
    v2_document = bind_file(V2_DOCUMENT_BINDING)
    control_code = verify_control_code()
    arm_records: dict[str, list[dict[str, Any]]] = {"shared_t4": [], "shared_ts4": []}
    metadata_by_arm: dict[str, dict[int, dict[str, Any]]] = {"shared_t4": {}, "shared_ts4": {}}
    for arm in ("shared_t4", "shared_ts4"):
        for seed in (42, 43, 44):
            metadata, record = validate_arm_metadata(arm, seed, ARM_BINDINGS[arm][seed])
            metadata_by_arm[arm][seed] = metadata
            arm_records[arm].append(record)
    matches = [
        compare_matched_pair(seed, metadata_by_arm["shared_t4"][seed], metadata_by_arm["shared_ts4"][seed])
        for seed in (42, 43, 44)
    ]

    seed44_ts4 = next(row for row in arm_records["shared_ts4"] if row["seed"] == 44)
    return {
        "schema_version": 2,
        "scope_id": SCOPE_ID,
        "status": "candidate_frozen_metadata_only_external_runner_blocked",
        "supersedes": {
            "scope_id": v1_payload["scope_id"],
            "reason": (
                "v1 froze shared-T4 only and could not support descriptor-row-attachment mechanism attribution; "
                "v2 adds the pre-existing matched shared-TS4 control without changing the DANDI scope"
            ),
            "append_only_v1_files": v1_files,
            "v1_was_modified": False,
        },
        "generation": {
            "deterministic_no_wall_clock": True,
            "generator": {
                "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "bytes": Path(__file__).stat().st_size,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "document": v2_document,
            "write_policy": "atomic hard-link publication; existing destination is never replaced",
        },
        "safety_boundary": {
            "metadata_api_only": True,
            "v1_replayed_and_byte_verified": True,
            "nwb_downloaded": False,
            "nwb_content_opened": False,
            "model_or_behavior_scores_computed": False,
            "gpu_used": False,
            "sub_c_endpoint_opened": False,
            "authorizes_download": False,
            "authorizes_schema_preflight": False,
            "authorizes_external_scoring": False,
        },
        "dandiset": v1_payload["dandiset"],
        "api_inventory": v1_payload["api_inventory"],
        "selection": v1_payload["selection"],
        "selected_assets": v1_payload["selected_assets"],
        "public_metadata_limits": v1_payload["public_metadata_limits"],
        "future_score_blind_schema_preflight": {
            **v1_payload["future_score_blind_schema_preflight"],
            "minimum_eligible_sessions": 6,
            "under_minimum_action": "endpoint NO-GO before checkpoint load or any score",
            "common_cohort_required_for_both_arms_views_and_all_seeds": True,
        },
        "matched_mechanism_arms": {
            "shared_t4": arm_records["shared_t4"],
            "shared_ts4": arm_records["shared_ts4"],
            "common": {
                "seed_set": [42, 43, 44],
                "lambda_consistency": 0,
                "terminal_checkpoint_rule": "one fixed epoch_011 checkpoint per arm and seed; target cannot select epoch",
                "teacher_source_and_normalizers": v1_payload["frozen_c1_candidate"]["source_files"]
                | {"normalizers": v1_payload["frozen_c1_candidate"]["normalizers"]},
                "control_code": control_code,
            },
            "strict_pair_matches": matches,
            "unique_scientific_difference": (
                "descriptor-row attachment: T4 preserves normalized row-to-unit/channel attachment; "
                "TS4 permutes complete normalized four-value rows along the unit/channel axis using "
                "numpy RandomState(training seed), with the same row multiset, normalizer, architecture, "
                "label/rate budget, teacher, source sessions, objective, and parameterization"
            ),
            "ts4_semantics": {
                "base_descriptor": "the identical raw T4 [a,c,m,b] fit",
                "normalization": "the identical view-specific source-train-27 T4 normalizer is applied before shuffle",
                "operation": "features[RandomState(seed).permutation(N), :]",
                "moves": "all four values as one complete row",
                "does_not_change": ["row values", "row multiset", "target labels", "electrode IDs", "width", "network"],
                "permutation_seed": "model training seed 42, 43, or 44",
                "identity_policy": (
                    "the historical implementation does not force a nonidentity permutation; any realized identity "
                    "is reported and retained, never used as a score-blind or score-based exclusion"
                ),
            },
        },
        "artifact_recovery": {
            "arm": "shared_ts4",
            "seed": 44,
            "artifact": seed44_ts4["terminal_checkpoint"],
            "provenance": "hash-qualified copy from the known original remote C1 stage into the previously absent canonical local path",
            "pre_copy_target_existed": False,
            "remote_sha256_reported": ARM_BINDINGS["shared_ts4"][44]["checkpoint_sha256"],
            "local_post_copy_sha256_verified": seed44_ts4["terminal_checkpoint"]["sha256"],
            "expected_bytes_verified": ARM_BINDINGS["shared_ts4"][44]["checkpoint_bytes"],
            "authority": "pre-existing completed closure metadata and terminal checkpoint SHA, not a new run",
            "retraining": False,
            "checkpoint_selection": False,
            "overwrite": False,
        },
        "endpoint_protocol": endpoint_protocol(),
        "external_score_only_runner": {
            "status": "blocked_not_implemented",
            "inherits_v1_blocker": True,
            "required_before_any_score_or_gpu": [
                "separately authorized all-22 score-blind NWB compatibility receipt",
                "N >= 6 before any checkpoint is loaded",
                "one common frozen eligible cohort across arms/views/seeds",
                "exact terminal checkpoints, source-only normalizers, T4/TS4 row semantics, R2, aggregation, and bootstrap frozen here",
                "zero target optimizer/backward/update and no target-driven epoch/session/normalizer/budget selection",
                "fresh single-use output root with no overwrite or retry",
            ],
        },
    }


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.timeout_seconds > 0, "timeout must be positive")
    output = args.output.expanduser().resolve()
    if not args.verify_existing:
        require(not os.path.lexists(output), f"write-once destination already exists: {output}")
    manifest = build_manifest(args.timeout_seconds)
    body = canonical_bytes(manifest)
    if args.verify_existing:
        require(output.is_file(), f"manifest to verify is missing: {output}")
        require(output.read_bytes() == body, f"existing v2 manifest is not byte-reproducible: {output}")
        action = "verified"
    else:
        try:
            v1.write_once_atomic(output, body)
        except v1.FreezeError as exc:
            raise FreezeV2Error(str(exc)) from exc
        action = "created"
    print(
        json.dumps(
            {
                "action": action,
                "manifest": str(output),
                "manifest_sha256": hashlib.sha256(body).hexdigest(),
                "selected_assets": len(manifest["selected_assets"]),
                "matched_cells": 6,
                "metadata_only": True,
                "status": manifest["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except FreezeV2Error as exc:
        raise SystemExit(f"FAIL_CLOSED_V2: {exc}") from exc
