"""Physical three-seed inference executor for the M2 activity-budget screen."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from . import core, plan


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parent(repo_root: Path) -> tuple[dict[tuple[str, str, str], Mapping[str, object]], str]:
    path = repo_root / plan.PARENT_SCORE_RELATIVE
    core.require(path.is_file(), "seed42 parent score missing")
    digest = _sha(path)
    core.require(digest == plan.PARENT_SCORE_SHA256, "seed42 parent score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    core.require(payload.get("schema") == "m2_t4_activity_budget_screen_v1", "parent schema drift")
    core.require(payload.get("status") == "TERMINAL" and payload.get("parameter_updates") == 0,
                 "parent is not an accepted zero-update terminal")
    rows = payload.get("rows")
    core.require(isinstance(rows, list) and len(rows) == 65, "parent row count drift")
    result: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in rows:
        core.require(isinstance(row, Mapping), "malformed parent row")
        key = (str(row.get("surface")), str(row.get("session")), str(row.get("cell")))
        core.require(key not in result, "duplicate parent row")
        result[key] = row
    return result, digest


def _checkpoint(repo_root: Path, seed: int) -> tuple[Path, str]:
    relative, expected = plan.CHECKPOINTS[seed]
    path = repo_root / relative
    core.require(path.is_file(), f"seed{seed} checkpoint missing")
    observed = _sha(path)
    core.require(observed == expected, f"seed{seed} checkpoint drift")
    manifest = json.loads((path.parents[1] / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    core.require(manifest.get("artifact_checkpoint_sha256") == observed,
                 f"seed{seed} checkpoint manifest drift")
    split = json.loads((path.parents[1] / "split_manifest.json").read_text(encoding="utf-8"))
    core.require(split.get("native_t4_normalization", {}).get("sha256") == plan.NORMALIZATION_SHA256,
                 f"seed{seed} T4 normalizer drift")
    return path, observed


def _publish(root: Path, payload: object) -> None:
    core.require(not root.exists(), f"result root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        body = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        (temp / "score.json").write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        (temp / "score.json.sha256").write_text(f"{digest}  score.json\n", encoding="ascii")
        (temp / "score.json").chmod(0o444)
        (temp / "score.json.sha256").chmod(0o444)
        temp.rename(root)
        root.chmod(0o555)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def execute(repo_root: Path, *, gpu_index: int = 1, batch_size: int = plan.DEFAULT_BATCH_SIZE) -> dict[str, object]:
    core.require(isinstance(gpu_index, int) and not isinstance(gpu_index, bool) and gpu_index >= 0,
                 "gpu index invalid")
    core.require(isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0,
                 "batch size invalid")
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu_index), "CUDA_VISIBLE_DEVICES mismatch")
    core.require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER mismatch")
    core.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "deterministic CUBLAS workspace missing")
    root = plan.result_root(repo_root)
    core.require(not root.exists(), f"result root already exists: {root}")
    parent, parent_sha = _load_parent(repo_root)
    checkpoints = {seed: _checkpoint(repo_root, seed) for seed in plan.SEEDS}

    import torch

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as screen_core
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import physical as screen_physical

    core.require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    started = time.monotonic()
    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    core.require(metadata["normalization_sha256"] == plan.NORMALIZATION_SHA256, "runtime normalizer drift")
    datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    rows: list[dict[str, object]] = []
    values: dict[tuple[int, str, str], dict[str, float]] = {}
    for seed in plan.SEEDS:
        if seed == 42:
            for (surface, session, cell), row in sorted(parent.items()):
                copied = {**dict(row), "seed": seed, "source": "reused_seed42_parent"}
                rows.append(copied)
                values.setdefault((seed, surface, cell), {})[session] = float(row["r2"])
            continue
        checkpoint_path, checkpoint_sha = checkpoints[seed]
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state, strict=True)
        student = model.student.to(device).eval()
        for parameter in student.parameters():
            parameter.requires_grad_(False)
        for surface, dataset in datasets.items():
            core.require(dataset is not None, f"{surface}: dataset missing")
            sessions = sorted(dataset.calib_trialized_neural_features)
            core.require(len(sessions) == plan.EXPECTED_SESSIONS[surface], f"{surface}: session count drift")
            for cell in screen_core.CELL_SPECS:
                for session in sessions:
                    row = screen_physical._score_session(
                        torch=torch,
                        student=student,
                        dataset=dataset,
                        session=session,
                        surface=surface,
                        cell=cell,
                        device=device,
                        batch_size=batch_size,
                    )
                    extended = {
                        **row,
                        "seed": seed,
                        "source": "new_frozen_checkpoint_forward",
                        "checkpoint_sha256": checkpoint_sha,
                        "parameter_updates": 0,
                    }
                    rows.append(extended)
                    values.setdefault((seed, surface, cell.name), {})[session] = float(row["r2"])

    expected_rows = len(plan.SEEDS) * sum(plan.EXPECTED_SESSIONS.values()) * 5
    core.require(len(rows) == expected_rows, "three-seed row count drift")
    surface_index = {name: index for index, name in enumerate(datasets)}
    cell_index = {cell.name: index for index, cell in enumerate(screen_core.CELL_SPECS)}
    rows.sort(
        key=lambda row: (
            plan.SEEDS.index(int(row["seed"])),
            surface_index[str(row["surface"])],
            cell_index[str(row["cell"])],
            str(row["session"]),
        )
    )
    summaries = {
        f"seed{seed}:{surface}:{cell}": core.summarize(session_values)
        for (seed, surface, cell), session_values in sorted(values.items())
    }
    contrasts: dict[str, dict[str, object]] = {}
    seed_level_means: dict[tuple[str, int], dict[str, float]] = {}
    for seed in plan.SEEDS:
        for surface in datasets:
            for budget in (10, 4):
                result = core.paired(
                    values[(seed, surface, f"ridge_activity30_m{budget}")],
                    values[(seed, surface, f"ridge_static_m{budget}")],
                )
                contrasts[f"seed{seed}:{surface}:activity30_minus_static_m{budget}"] = result
                seed_level_means.setdefault((surface, budget), {})[f"seed{seed}"] = float(result["mean_delta"])
    across_seed = {
        f"{surface}:activity30_minus_static_m{budget}": core.summarize(seed_means)
        for (surface, budget), seed_means in sorted(seed_level_means.items())
    }
    payload = {
        "schema": plan.SCHEMA,
        "status": "TERMINAL",
        "scientific_role": "frozen_inference_three_seed_replication",
        "seeds": list(plan.SEEDS),
        "cell_order": [cell.name for cell in screen_core.CELL_SPECS],
        "parent_score_sha256": parent_sha,
        "checkpoint_sha256": {str(seed): checkpoints[seed][1] for seed in plan.SEEDS},
        "normalization_sha256": plan.NORMALIZATION_SHA256,
        "row_count": len(rows),
        "reused_rows": 65,
        "new_forward_rows": len(rows) - 65,
        "target_gradients": 0,
        "parameter_updates": 0,
        "query_target_used_for_selection": False,
        "gpu": {
            "cuda_visible_devices": str(gpu_index),
            "logical_device": "cuda:0",
            "name": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "rows": rows,
        "summaries": summaries,
        "paired_contrasts": contrasts,
        "across_seed_delta_summaries": across_seed,
    }
    _publish(root, payload)
    return payload
