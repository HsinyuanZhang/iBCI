#!/usr/bin/env python3
"""Freeze the DANDI 000688 sub-M center-out scope from metadata only.

This program is deliberately incapable of downloading or opening NWB content.  It
queries only the published DANDI version endpoint, the paginated asset-list
endpoint, and the per-asset metadata endpoint.  It also binds already-frozen C1
files by hashing them as opaque local files; it never imports torch or an NWB
reader and never computes an accuracy metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DANDISET_ID = "000688"
DANDISET_VERSION = "0.250122.1735"
API_ORIGIN = "https://api.dandiarchive.org"
API_PREFIX = f"/api/dandisets/{DANDISET_ID}/versions/{DANDISET_VERSION}/"
VERSION_URL = f"{API_ORIGIN}{API_PREFIX}"
ASSET_LIST_URL = f"{API_ORIGIN}{API_PREFIX}assets/?page_size=100"
ASSET_DETAIL_URL = f"{API_ORIGIN}{API_PREFIX}assets/{{asset_id}}/"

SCOPE_ID = "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v1"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "sua_exploration/manifests/"
    "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v1.json"
)

EXPECTED_DANDISET_NAME = (
    "Long-term recordings of motor and premotor cortical spiking activity "
    "during reaching in monkeys"
)
EXPECTED_GLOBAL_ASSET_COUNT = 111
EXPECTED_GLOBAL_BYTES = 13_179_483_710
EXPECTED_PAGE_LENGTHS = [100, 11]
EXPECTED_SUBM_ASSET_COUNT = 28
EXPECTED_SUBM_BYTES = 2_915_252_180
EXPECTED_CO_ASSET_COUNT = 22
EXPECTED_CO_BYTES = 2_312_360_648
EXPECTED_RT_ASSET_COUNT = 6
EXPECTED_RT_BYTES = 602_891_532

CO_PATH_RE = re.compile(
    r"^sub-M/sub-M_ses-CO-(?P<date>[0-9]{8})_behavior\+ecephys\.nwb$"
)
RT_PATH_RE = re.compile(
    r"^sub-M/sub-M_ses-RT-(?P<date>[0-9]{8})_behavior\+ecephys\.nwb$"
)
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DANDI_ETAG_RE = re.compile(r"^[0-9a-f]{32}(?:-[1-9][0-9]*)?$")

EXPECTED_VARIABLE_MEASURED = {
    "BehavioralTimeSeries",
    "ElectrodeGroup",
    "Position",
    "ProcessingModule",
    "SpatialSeries",
    "Units",
}
EXPECTED_MEASUREMENT_TECHNIQUES = {
    "analytical technique",
    "behavioral technique",
    "spike sorting technique",
    "surgical technique",
}

C1_SOURCE_BINDINGS = {
    "train_val_manifest": {
        "path": "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json",
        "sha256": "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9",
    },
    "data_manifest": {
        "path": (
            "sua_exploration/results/"
            "t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/"
            "c1_train_val_33_manifest.json"
        ),
        "sha256": "1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb",
    },
    "teacher": {
        "path": (
            "sua_exploration/checkpoints/teacher_mc_maze/"
            "best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
        ),
        "sha256": "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d",
    },
}

C1_NORMALIZER_BINDINGS = {
    "behavior": {
        "semantic_scope": "source_train_27_only",
        "procedure": "fit_behavior_stats(source_train_27, bin_size_ms=20)",
        "artifacts": [
            {
                "view": "sua",
                "path": (
                    "sua_exploration/cache/"
                    "t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/"
                    "sua/behavior_stats/be50f588491c004f721e.npz"
                ),
                "sha256": "821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd",
            },
            {
                "view": "pseudo_mua",
                "path": (
                    "sua_exploration/cache/"
                    "t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/"
                    "pseudo_mua/behavior_stats/be50f588491c004f721e.npz"
                ),
                "sha256": "821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd",
            },
        ],
    },
    "t4_by_view": {
        "sua": {
            "semantic_scope": "source_train_27_only",
            "semantic_sha256": "ac5156097864110685e0b2fbfe314edcb747e69dc821c10451984a089be8a7a7",
            "artifact": {
                "path": (
                    "sua_exploration/cache/"
                    "t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/"
                    "sua/side_feature_stats/dd3da1f59700c1b96ab8.npz"
                ),
                "sha256": "32d32a7fe1b80a139571aae0ce3c3a1d802aec21c99b23a4cff72b8a60261701",
            },
        },
        "pseudo_mua": {
            "semantic_scope": "source_train_27_only",
            "semantic_sha256": "92470ad14062af6cb998e06e7696b94bfdfd20ac5e415615302a5dddc7098fcc",
            "artifact": {
                "path": (
                    "sua_exploration/cache/"
                    "t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/"
                    "pseudo_mua/side_feature_stats/e31b40fe6b85b137b5b9.npz"
                ),
                "sha256": "17596b29d90c29ca67efa87437eda909e53dc931fe8f897f513f7ce8e5790236",
            },
        },
    },
}

C1_SEED_BINDINGS = {
    42: {
        "run_metadata_path": (
            "sua_exploration/results/"
            "t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/"
            "shared_t4_s42/run_metadata.json"
        ),
        "run_metadata_sha256": "98550528e91e6a7c5f637a2acc5112d53ae12412d6077804c30ac950eabfd000",
        "checkpoint_path": (
            "sua_exploration/checkpoints/"
            "t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_"
            "shared_t4_s42/epoch_ckpts/epoch_011.ckpt"
        ),
        "checkpoint_sha256": "ab9df840a07d7aeb6cc417bb684f1f5e0265d50f98168400ac915647cdfd7b9f",
    },
    43: {
        "run_metadata_path": (
            "sua_exploration/results/"
            "t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/"
            "shared_t4_s43/run_metadata.json"
        ),
        "run_metadata_sha256": "f591b4425ce6d39865d2599e96e7211b382fc571c2a7f0b4bfc332577376b83b",
        "checkpoint_path": (
            "sua_exploration/checkpoints/"
            "t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_"
            "shared_t4_s43/epoch_ckpts/epoch_011.ckpt"
        ),
        "checkpoint_sha256": "05c05b3ab82a2fba43c55aca523248982a954faf5f0363a0235a29d64e57ab22",
    },
    44: {
        "run_metadata_path": (
            "sua_exploration/results/"
            "t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/"
            "shared_t4_s44/run_metadata.json"
        ),
        "run_metadata_sha256": "d3e84a1c248a0d2fc97dd2d009f4e31ec7800de4f0358161fab23d7c689807d7",
        "checkpoint_path": (
            "sua_exploration/checkpoints/"
            "t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_"
            "shared_t4_s44/epoch_ckpts/epoch_011.ckpt"
        ),
        "checkpoint_sha256": "a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6",
    },
}


class FreezeError(RuntimeError):
    """Fail-closed scope-freeze error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bound_file(relative_path: str, expected_sha256: str) -> dict[str, Any]:
    require(SHA256_RE.fullmatch(expected_sha256) is not None, "invalid expected SHA-256")
    path = REPO_ROOT / relative_path
    require(path.is_file(), f"required frozen file is missing: {relative_path}")
    observed = sha256_file(path)
    require(
        observed == expected_sha256,
        f"frozen file hash drift: {relative_path}: expected {expected_sha256}, got {observed}",
    )
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def load_json_file(relative_path: str) -> dict[str, Any]:
    path = REPO_ROOT / relative_path
    require(path.is_file(), f"required JSON file is missing: {relative_path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"cannot parse required JSON {relative_path}: {exc}") from exc
    require(isinstance(payload, dict), f"JSON root must be an object: {relative_path}")
    return payload


def validate_api_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == "https", f"refusing non-HTTPS API URL: {url}")
    require(parsed.netloc == "api.dandiarchive.org", f"refusing non-DANDI API host: {url}")
    require(parsed.path.startswith(API_PREFIX), f"refusing API path outside frozen version: {url}")
    require("/download" not in parsed.path, f"refusing asset download URL: {url}")
    require("contentUrl" not in url, f"refusing content URL: {url}")


def get_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    validate_api_url(url)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"SPINT-{SCOPE_ID}-metadata-only/1",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            final_url = response.geturl()
            validate_api_url(final_url)
            content_type = response.headers.get_content_type()
            require(
                content_type in {"application/json", "application/vnd.api+json"},
                f"unexpected API content type {content_type!r} for {url}",
            )
            body = response.read(16 * 1024 * 1024 + 1)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FreezeError(f"DANDI metadata request failed for {url}: {exc}") from exc
    require(len(body) <= 16 * 1024 * 1024, f"metadata response too large: {url}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise FreezeError(f"DANDI API returned invalid JSON for {url}: {exc}") from exc
    require(isinstance(payload, dict), f"DANDI API root is not an object: {url}")
    return payload


def validate_version_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("id") == f"DANDI:{DANDISET_ID}/{DANDISET_VERSION}", "Dandiset ID drift")
    require(payload.get("identifier") == f"DANDI:{DANDISET_ID}", "Dandiset identifier drift")
    require(payload.get("version") == DANDISET_VERSION, "published version drift")
    require(payload.get("name") == EXPECTED_DANDISET_NAME, "Dandiset name drift")
    summary = payload.get("assetsSummary")
    require(isinstance(summary, dict), "Dandiset assetsSummary missing")
    require(summary.get("numberOfFiles") == EXPECTED_GLOBAL_ASSET_COUNT, "assetsSummary file count drift")
    require(summary.get("numberOfBytes") == EXPECTED_GLOBAL_BYTES, "assetsSummary byte count drift")
    require(summary.get("numberOfSubjects") == 4, "assetsSummary subject count drift")
    return {
        "id": payload["id"],
        "identifier": payload["identifier"],
        "version": payload["version"],
        "name": payload["name"],
        "doi": payload.get("doi"),
        "date_published": payload.get("datePublished"),
        "schema_version": payload.get("schemaVersion"),
        "assets_summary": {
            "number_of_files": summary["numberOfFiles"],
            "number_of_bytes": summary["numberOfBytes"],
            "number_of_subjects": summary["numberOfSubjects"],
        },
    }


def validate_list_row(row: Any) -> dict[str, Any]:
    require(isinstance(row, dict), "asset-list row is not an object")
    asset_id = row.get("asset_id")
    path = row.get("path")
    size = row.get("size")
    require(isinstance(asset_id, str) and UUID_RE.fullmatch(asset_id) is not None, "invalid asset_id")
    require(isinstance(path, str) and path and not path.startswith("/"), f"invalid path for {asset_id}")
    require(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"invalid size for {path}")
    require(row.get("zarr") is None, f"unexpected Zarr asset in NWB-only Dandiset: {path}")
    require(isinstance(row.get("blob"), str) and row["blob"], f"missing blob ID for {path}")
    return {
        "asset_id": asset_id,
        "path": path,
        "size": size,
        "blob_id": row["blob"],
    }


def list_all_assets(
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[int], list[int | None]]:
    rows: list[dict[str, Any]] = []
    page_lengths: list[int] = []
    page_counts: list[int | None] = []
    seen_urls: set[str] = set()
    url: str | None = ASSET_LIST_URL
    while url is not None:
        require(url not in seen_urls, f"pagination loop detected at {url}")
        seen_urls.add(url)
        page = get_json(url, timeout_seconds)
        page_counts.append(page.get("count"))
        if len(page_lengths) == 0:
            require(page.get("count") == EXPECTED_GLOBAL_ASSET_COUNT, "first-page asset-list count drift")
            require(page.get("previous") is None, "first-page previous URL must be null")
        else:
            # The production API's real pagination schema retains the key but
            # returns JSON null for count after page 1.
            require(page.get("count") is None, "later-page asset-list count must be null")
        results = page.get("results")
        require(isinstance(results, list), f"asset-list results missing at {url}")
        page_lengths.append(len(results))
        rows.extend(validate_list_row(row) for row in results)
        next_url = page.get("next")
        require(next_url is None or isinstance(next_url, str), "asset-list next URL has invalid type")
        if isinstance(next_url, str):
            validate_api_url(next_url)
        url = next_url
        require(len(page_lengths) <= 10, "unexpectedly deep DANDI pagination")

    require(page_lengths == EXPECTED_PAGE_LENGTHS, f"page lengths drifted: {page_lengths}")
    require(page_counts == [EXPECTED_GLOBAL_ASSET_COUNT, None], f"page counts drifted: {page_counts}")
    require(len(rows) == EXPECTED_GLOBAL_ASSET_COUNT, "paginated asset count is incomplete")
    require(sum(row["size"] for row in rows) == EXPECTED_GLOBAL_BYTES, "paginated byte total drift")
    require(len({row["asset_id"] for row in rows}) == len(rows), "duplicate asset_id across pages")
    require(len({row["path"] for row in rows}) == len(rows), "duplicate asset path across pages")
    return rows, page_lengths, page_counts


def select_subm(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subm = sorted((row for row in rows if row["path"].startswith("sub-M/")), key=lambda row: row["path"])
    require(len(subm) == EXPECTED_SUBM_ASSET_COUNT, "sub-M asset count drift")
    require(sum(row["size"] for row in subm) == EXPECTED_SUBM_BYTES, "sub-M byte total drift")

    center_out = [row for row in subm if CO_PATH_RE.fullmatch(row["path"]) is not None]
    random_target = [row for row in subm if RT_PATH_RE.fullmatch(row["path"]) is not None]
    require(len(center_out) == EXPECTED_CO_ASSET_COUNT, "sub-M CO count drift")
    require(sum(row["size"] for row in center_out) == EXPECTED_CO_BYTES, "sub-M CO byte total drift")
    require(len(random_target) == EXPECTED_RT_ASSET_COUNT, "sub-M RT count drift")
    require(sum(row["size"] for row in random_target) == EXPECTED_RT_BYTES, "sub-M RT byte total drift")
    classified = {row["asset_id"] for row in center_out + random_target}
    require(len(classified) == len(subm), "unclassified sub-M asset found")
    return center_out, random_target


def metadata_values(payload: dict[str, Any], key: str, value_key: str) -> set[str]:
    rows = payload.get(key)
    require(isinstance(rows, list), f"asset metadata {key} missing")
    values: set[str] = set()
    for row in rows:
        require(isinstance(row, dict), f"asset metadata {key} row is not an object")
        value = row.get(value_key)
        require(isinstance(value, str) and value, f"asset metadata {key}.{value_key} missing")
        values.add(value)
    return values


def fetch_asset_detail(row: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    asset_id = row["asset_id"]
    payload = get_json(ASSET_DETAIL_URL.format(asset_id=asset_id), timeout_seconds)
    require(payload.get("id") == f"dandiasset:{asset_id}", f"asset detail ID drift: {asset_id}")
    require(payload.get("identifier") == asset_id, f"asset detail identifier drift: {asset_id}")
    require(payload.get("path") == row["path"], f"asset path mismatch: {asset_id}")
    require(payload.get("contentSize") == row["size"], f"asset size mismatch: {asset_id}")
    require(payload.get("encodingFormat") == "application/x-nwb", f"not an NWB asset: {asset_id}")
    require(payload.get("schemaKey") == "Asset", f"unexpected asset schema key: {asset_id}")

    digest = payload.get("digest")
    require(isinstance(digest, dict), f"top-level digest missing: {asset_id}")
    dandi_etag = digest.get("dandi:dandi-etag")
    sha256 = digest.get("dandi:sha2-256")
    require(
        isinstance(dandi_etag, str) and DANDI_ETAG_RE.fullmatch(dandi_etag) is not None,
        f"top-level dandi-etag missing or invalid: {asset_id}",
    )
    require(
        isinstance(sha256, str) and SHA256_RE.fullmatch(sha256) is not None,
        f"top-level SHA-256 missing or invalid: {asset_id}",
    )

    variables = metadata_values(payload, "variableMeasured", "value")
    techniques = metadata_values(payload, "measurementTechnique", "name")
    require(variables == EXPECTED_VARIABLE_MEASURED, f"variableMeasured drift: {asset_id}: {variables}")
    require(
        techniques == EXPECTED_MEASUREMENT_TECHNIQUES,
        f"measurementTechnique drift: {asset_id}: {techniques}",
    )

    participants = [
        item.get("identifier")
        for item in payload.get("wasAttributedTo", [])
        if isinstance(item, dict) and item.get("schemaKey") == "Participant"
    ]
    require(participants == ["M"], f"participant metadata drift: {asset_id}: {participants}")
    expected_session = row["path"].split("/", 1)[1].removesuffix("_behavior+ecephys.nwb")
    sessions = [
        item
        for item in payload.get("wasGeneratedBy", [])
        if isinstance(item, dict) and item.get("schemaKey") == "Session"
    ]
    require(len(sessions) == 1, f"session metadata count drift: {asset_id}")
    require(sessions[0].get("identifier") == expected_session, f"session identifier drift: {asset_id}")
    require(
        sessions[0].get("description") == "Monkey M performing center-out reaching task.",
        f"session task description drift: {asset_id}",
    )

    date_match = CO_PATH_RE.fullmatch(row["path"])
    require(date_match is not None, f"selected row no longer matches CO rule: {row['path']}")
    return {
        "asset_id": asset_id,
        "path": row["path"],
        "session_id": expected_session,
        "session_date": date_match.group("date"),
        "size": row["size"],
        "dandi_etag": dandi_etag,
        "sha256": sha256,
        "encoding_format": payload["encodingFormat"],
        "public_metadata": {
            "participant_identifier": "M",
            "session_description": sessions[0]["description"],
            "variable_measured": sorted(variables),
            "measurement_techniques": sorted(techniques),
        },
    }


def validate_run_metadata(seed: int, binding: dict[str, str]) -> dict[str, Any]:
    metadata_file = bound_file(binding["run_metadata_path"], binding["run_metadata_sha256"])
    payload = load_json_file(binding["run_metadata_path"])
    require(payload.get("status") == "completed", f"C1 seed {seed} is not completed")
    require(payload.get("seed") == seed, f"C1 seed mismatch for {seed}")
    require(payload.get("training_kind") == "shared_paired_view", f"C1 kind mismatch for {seed}")
    require(payload.get("held_out_test_evaluated") is False, f"C1 held-out flag mismatch for {seed}")
    require(payload.get("formal_sua_files_opened") is False, f"C1 formal flag mismatch for {seed}")

    objective = payload.get("paired_objective")
    require(isinstance(objective, dict), f"C1 objective missing for {seed}")
    require(objective.get("lambda_consistency") == 0, f"C1 lambda is not zero for {seed}")
    require(objective.get("sua_task_loss_weight") == 0.5, f"C1 SUA weight drift for {seed}")
    require(objective.get("pseudo_mua_task_loss_weight") == 0.5, f"C1 pseudo-MUA weight drift for {seed}")
    require(objective.get("view_specific_heads") is False, f"C1 head sharing drift for {seed}")

    require(
        payload.get("train_val_manifest_sha256")
        == C1_SOURCE_BINDINGS["train_val_manifest"]["sha256"],
        f"C1 train/validation manifest drift for {seed}",
    )
    require(
        payload.get("data_manifest_sha256") == C1_SOURCE_BINDINGS["data_manifest"]["sha256"],
        f"C1 data manifest drift for {seed}",
    )
    require(payload.get("teacher_sha256") == C1_SOURCE_BINDINGS["teacher"]["sha256"], f"teacher drift for {seed}")

    view_configs = payload.get("view_configs")
    require(isinstance(view_configs, dict), f"C1 view configs missing for {seed}")
    for view in ("sua", "pseudo_mua"):
        view_config = view_configs.get(view)
        require(isinstance(view_config, dict), f"C1 {view} config missing for {seed}")
        require(view_config.get("signal_view") == view, f"C1 {view} signal-view drift for {seed}")
        side = view_config.get("side_features")
        require(isinstance(side, dict), f"C1 {view} side features missing for {seed}")
        expected_normalizer = C1_NORMALIZER_BINDINGS["t4_by_view"][view]
        require(side.get("group") == "t4", f"C1 {view} descriptor group drift for {seed}")
        require(side.get("pool_size") == 50, f"C1 {view} T4 pool drift for {seed}")
        require(side.get("side_dim") == 4, f"C1 {view} side dimension drift for {seed}")
        require(side.get("normalization_scope") == "source_train_27_only", f"C1 {view} scope drift for {seed}")
        require(
            side.get("normalization_sha256") == expected_normalizer["semantic_sha256"],
            f"C1 {view} normalizer drift for {seed}",
        )
    epoch_checkpoints = payload.get("epoch_checkpoints")
    require(isinstance(epoch_checkpoints, list) and len(epoch_checkpoints) == 12, f"C1 epochs drift for {seed}")
    require(Path(epoch_checkpoints[-1]).name == "epoch_011.ckpt", f"C1 terminal epoch drift for {seed}")

    checkpoint_file = bound_file(binding["checkpoint_path"], binding["checkpoint_sha256"])
    return {
        "seed": seed,
        "run_metadata": metadata_file,
        "terminal_checkpoint": checkpoint_file,
        "terminal_epoch_one_based": 12,
        "checkpoint_filename_zero_based": "epoch_011.ckpt",
    }


def build_c1_binding() -> dict[str, Any]:
    sources = {
        name: bound_file(row["path"], row["sha256"])
        for name, row in C1_SOURCE_BINDINGS.items()
    }
    behavior_artifacts = [
        {
            "view": row["view"],
            **bound_file(row["path"], row["sha256"]),
        }
        for row in C1_NORMALIZER_BINDINGS["behavior"]["artifacts"]
    ]
    t4_by_view: dict[str, Any] = {}
    for view, row in C1_NORMALIZER_BINDINGS["t4_by_view"].items():
        t4_by_view[view] = {
            "semantic_scope": row["semantic_scope"],
            "semantic_sha256": row["semantic_sha256"],
            "artifact": bound_file(row["artifact"]["path"], row["artifact"]["sha256"]),
        }
    seeds = [validate_run_metadata(seed, C1_SEED_BINDINGS[seed]) for seed in sorted(C1_SEED_BINDINGS)]
    return {
        "candidate": "fresh_shared_t4_c1_fp32",
        "variant": "shared_t4",
        "weights_shared_across_sua_and_pseudo_mua": True,
        "lambda_consistency": 0,
        "c2_status": "closed_not_an_endpoint_rescue",
        "seed_set": [42, 43, 44],
        "endpoint_checkpoint_rule": (
            "one terminal epoch_011 FP32 checkpoint per seed; do not select an epoch "
            "or reuse the development epoch-window statistic after target access"
        ),
        "source_files": sources,
        "normalizers": {
            "behavior": {
                "semantic_scope": C1_NORMALIZER_BINDINGS["behavior"]["semantic_scope"],
                "procedure": C1_NORMALIZER_BINDINGS["behavior"]["procedure"],
                "artifacts": behavior_artifacts,
            },
            "t4_by_view": t4_by_view,
        },
        "seeds": seeds,
        "calibration_contract": {
            "bin_size_ms": 20,
            "window_size_bins": 50,
            "reward_filter": "result == 'R'",
            "selection_order": "chronological datamodule-usable rewarded trials",
            "activity_support": "[0,30)",
            "t4_label_and_rate_pool": "[0,50)",
            "query_start_trial": 50,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_weight_updates": 0,
        },
    }


def build_manifest(timeout_seconds: float) -> dict[str, Any]:
    version = validate_version_metadata(get_json(VERSION_URL, timeout_seconds))
    all_assets, page_lengths, page_counts = list_all_assets(timeout_seconds)
    center_out, random_target = select_subm(all_assets)
    selected_assets = [fetch_asset_detail(row, timeout_seconds) for row in center_out]
    require(len({row["sha256"] for row in selected_assets}) == len(selected_assets), "duplicate CO SHA-256")
    require(len({row["dandi_etag"] for row in selected_assets}) == len(selected_assets), "duplicate CO dandi-etag")
    c1 = build_c1_binding()

    return {
        "schema_version": 1,
        "scope_id": SCOPE_ID,
        "status": "scope_frozen_metadata_only_external_runner_blocked",
        "generation": {
            "deterministic_no_wall_clock": True,
            "generator": bound_file(
                str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                sha256_file(Path(__file__).resolve()),
            ),
            "write_policy": "atomic hard-link publication; existing destination is never replaced",
        },
        "safety_boundary": {
            "metadata_api_only": True,
            "allowed_hosts": ["api.dandiarchive.org"],
            "nwb_downloaded": False,
            "nwb_content_opened": False,
            "model_or_behavior_scores_computed": False,
            "gpu_used": False,
            "sub_c_endpoint_opened": False,
            "authorizes_download": False,
            "authorizes_schema_preflight": False,
            "authorizes_external_scoring": False,
        },
        "dandiset": version,
        "api_inventory": {
            "version_endpoint": VERSION_URL,
            "asset_list_endpoint": ASSET_LIST_URL,
            "asset_detail_endpoint_template": ASSET_DETAIL_URL,
            "page_size": 100,
            "page_lengths": page_lengths,
            "page_count_fields": page_counts,
            "global_asset_count": len(all_assets),
            "global_bytes": sum(row["size"] for row in all_assets),
            "sub_m_asset_count": len(center_out) + len(random_target),
            "sub_m_bytes": sum(row["size"] for row in center_out + random_target),
        },
        "selection": {
            "subject_prefix": "sub-M/",
            "include_regex": CO_PATH_RE.pattern,
            "exclude_regex": RT_PATH_RE.pattern,
            "rule": (
                "enumerate all 111 published assets across every API page; select every and "
                "only path matching the anchored sub-M center-out NWB regex"
            ),
            "selected_count": len(selected_assets),
            "selected_bytes": sum(row["size"] for row in selected_assets),
            "excluded_sub_m_count": len(random_target),
            "excluded_sub_m_bytes": sum(row["size"] for row in random_target),
            "excluded_sub_m_assets": [
                {
                    "asset_id": row["asset_id"],
                    "path": row["path"],
                    "size": row["size"],
                    "reason": "RT path does not match frozen CO include regex",
                }
                for row in random_target
            ],
        },
        "selected_assets": selected_assets,
        "public_metadata_limits": {
            "proved": [
                "immutable asset identity, path, size, dandi-etag, and SHA-256",
                "participant identifier M and center-out session description",
                "NWB encoding plus Units/Position/ElectrodeGroup metadata categories",
            ],
            "not_proved_without_nwb_content": [
                "manual-SUA row semantics and per-unit electrode attachment",
                "trial-table result/target_dir columns and chronological usable rewarded counts",
                "two-dimensional cursor_vel schema, timestamps, units, and finite values",
                "unit count below the frozen max_units_exclusive=100 gate",
                "first-50 T4 design rank and existence of post-trial-50 query windows",
            ],
        },
        "future_score_blind_schema_preflight": {
            "status": "not_run_requires_separate_authorization_and_local_nwb_content",
            "scope_policy": (
                "all 22 assets remain in the ledger; apply the same frozen gates to every asset, "
                "record every failure, and never remove a session because of a model score"
            ),
            "gates": [
                "downloaded bytes must match this manifest's size and SHA-256 before opening",
                "NWB subject/session/task identity must match the frozen asset row",
                "event-level Units table must contain spike_times and one valid electrodes reference per unit",
                "unit count must satisfy 0 < N < 100; no post-hoc truncation or cap change",
                "trials must contain start_time, stop_time, result, and finite target_dir",
                "trial rows and usable rewarded trials must be chronological",
                "with 20-ms bins and 50-bin windows, at least 50 usable rewarded trials must exist",
                "the first-50 [cos(theta),sin(theta),1] design must have rank 3; imbalance is reported, not tuned",
                "at least one complete valid query window must remain strictly after trial 50",
                "cursor_vel must be finite two-dimensional velocity with strictly increasing timestamps",
                "pseudo-MUA, if reported, must pool raw unit activity by validated electrode then refit T4",
            ],
            "forbidden": [
                "model forward pass or behavior prediction score",
                "checkpoint, lambda, budget, normalizer, component, or session selection from target scores",
                "target optimizer, backward, or weight update",
                "silent exclusion of an incompatible asset",
            ],
        },
        "frozen_c1_candidate": c1,
        "external_score_only_runner": {
            "status": "blocked_not_implemented",
            "reason": (
                "the existing frozen-manifest MultiSessionDataModule path is tied to the sub-C "
                "27/6/6 source/development registry and max_units_exclusive=100; it is not an "
                "authorized sub-M score-only entry point"
            ),
            "required_before_any_gpu_or_score": [
                "consume this exact scope manifest and a separately authorized schema-preflight receipt",
                "load only the three bound terminal checkpoints and bound source-only normalizers",
                "provide no train loader, optimizer, backward path, checkpoint writer, or target normalizer fit",
                "enforce activity support 30, T4 pool 50, and query start 50",
                "emit zero target optimizer/backward/update counters and fail if any are nonzero",
                "write to a fresh single-use output root and refuse retry or overwrite",
            ],
        },
    }


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_once_atomic(path: Path, body: bytes) -> None:
    path = path.expanduser().resolve()
    require(not os.path.lexists(path), f"write-once destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FreezeError(f"write-once destination raced into existence: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="regenerate in memory and require --output to be byte-identical; never write",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.timeout_seconds > 0, "timeout must be positive")
    manifest = build_manifest(args.timeout_seconds)
    body = canonical_bytes(manifest)
    output = args.output.expanduser().resolve()
    if args.verify_existing:
        require(output.is_file(), f"manifest to verify is missing: {output}")
        require(output.read_bytes() == body, f"existing manifest is not reproducible: {output}")
        action = "verified"
    else:
        write_once_atomic(output, body)
        action = "created"
    print(
        json.dumps(
            {
                "action": action,
                "asset_count": len(manifest["selected_assets"]),
                "manifest": str(output),
                "manifest_sha256": hashlib.sha256(body).hexdigest(),
                "metadata_only": True,
                "status": manifest["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except FreezeError as exc:
        raise SystemExit(f"FAIL_CLOSED: {exc}") from exc
