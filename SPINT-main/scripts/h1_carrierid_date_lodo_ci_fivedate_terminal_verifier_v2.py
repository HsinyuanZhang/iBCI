#!/usr/bin/env python3
"""Independent, fail-closed terminal verifier for the H1 CI64 five-date grid.

Unlike the historical receipt-only aggregate, this program follows every
terminal-evaluation receipt back through its terminal checker, checkpoint,
resolved config, CI source preflight, source manifest, phase-1 preflight, and
declared code hashes before it recomputes the five-date statistics.  It never
opens target data, constructs a Trainer, or initialises CUDA.  Checkpoint
inspection imports Torch lazily and only after all receipt paths are known.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import uuid
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATES = ("19250108", "19250113", "19250115", "19250119", "19250120")
ARMS = ("CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS")
EVALUATION_SCHEMA = "h1_carrierid_date_lodo_ci_five_arm_terminal_evaluation_v1"
CHECKER_SCHEMA = "h1_carrierid_date_lodo_ci_five_arm_terminal_check_v1"
CHECKER_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_FIVE_ARM_SOURCE_E49_CHECKPOINTS_NO_TARGET"
PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_ci_cpu_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_SOURCE_ONLY_NOT_LAUNCHED"
CHECKPOINT_SCHEMA = "h1_carrierid_date_lodo_ci_terminal_checkpoint_v1"
VERIFIER_SCHEMA = "h1_carrierid_date_lodo_ci_fivedate_terminal_verifier_v2"
VERIFIER_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_FIVEDATE_TERMINAL_VERIFIED_V2"
PRACTICAL_WIDTH_THRESHOLD = 0.03
DEFAULT_EVALUATION_DIR = ROOT / "pilot_artifacts" / "h1_carrierid_date_lodo_ci" / "terminal_evaluations"
DEFAULT_OUTPUT = ROOT / "pilot_artifacts" / "h1_carrierid_date_lodo_ci" / "H1_CARRIERID_DATE_LODO_CI_FIVEDATE_TERMINAL_VERIFIER_v2.json"

# This is deliberately an explicit closure, rather than a glob.  Adding a
# runtime dependency to the terminal evaluator requires updating the verifier.
CHECKER_CLOSURE_FILES = (
    "src/data/h1_carrierid_date_lodo_ci_target.py",
    "src/data/h1_carrierid_date_lodo_target.py",
    "src/data/h1_carrierid_date_lodo_ci.py",
    "src/data/h1_carrierid_date_lodo_source.py",
    "src/data/h1_m4_eb_pilot.py",
    "src/models/components/h1_carrierid_ci_spint.py",
    "src/models/h1_carrierid_date_lodo_ci_module.py",
    "src/models/falcon_module.py",
    "src/models/components/spint.py",
    "src/h1_m4_cce_contract.py",
    "scripts/h1_carrierid_date_lodo_ci_terminal_checker.py",
    "scripts/h1_carrierid_date_lodo_ci_terminal_evaluate.py",
)
PREFLIGHT_CODE_FILES = {
    "data": "src/data/h1_carrierid_date_lodo_ci.py",
    "model": "src/models/h1_carrierid_date_lodo_ci_module.py",
    "component": "src/models/components/h1_carrierid_ci_spint.py",
    "preflight": "scripts/h1_carrierid_date_lodo_ci_preflight.py",
}

# There are three deliberately different configuration artifacts in this
# protocol.  Keeping their roles separate is not a cosmetic provenance detail:
# a resolved run configuration cannot prove which immutable experiment arm was
# selected, while a static experiment file cannot prove how the checkpoint was
# actually launched.
STATIC_EXPERIMENTS = {
    "CI32-FULL": {
        "filename": "h1_carrierid_date_lodo_ci32_full.yaml",
        "experiment": "h1_carrierid_date_lodo_ci32_full",
        "model_choice": "falcon_h1_carrierid_date_lodo_ci32_full",
        "defaults": (
            {"override /data": "falcon_h1_carrierid_date_lodo_ci"},
            {"override /model": "falcon_h1_carrierid_date_lodo_ci32_full"},
            {"override /callbacks": "h1_carrierid_date_lodo_phase2_terminal"},
        ),
    },
    "CI64-FULL": {
        "filename": "h1_carrierid_date_lodo_ci64_full.yaml",
        "experiment": "h1_carrierid_date_lodo_ci64_full",
        "model_choice": "falcon_h1_carrierid_date_lodo_ci64_full",
        "defaults": ("h1_carrierid_date_lodo_ci32_full", {"override /model": "falcon_h1_carrierid_date_lodo_ci64_full"}),
    },
    "CI64-C0": {
        "filename": "h1_carrierid_date_lodo_ci64_c0.yaml",
        "experiment": "h1_carrierid_date_lodo_ci64_c0",
        "model_choice": "falcon_h1_carrierid_date_lodo_ci64_c0",
        "defaults": ("h1_carrierid_date_lodo_ci64_full", {"override /model": "falcon_h1_carrierid_date_lodo_ci64_c0"}),
    },
    "CI64-LS": {
        "filename": "h1_carrierid_date_lodo_ci64_ls.yaml",
        "experiment": "h1_carrierid_date_lodo_ci64_ls",
        "model_choice": "falcon_h1_carrierid_date_lodo_ci64_ls",
        "defaults": ("h1_carrierid_date_lodo_ci64_full", {"override /model": "falcon_h1_carrierid_date_lodo_ci64_ls"}),
    },
    "CI64-RS": {
        "filename": "h1_carrierid_date_lodo_ci64_rs.yaml",
        "experiment": "h1_carrierid_date_lodo_ci64_rs",
        "model_choice": "falcon_h1_carrierid_date_lodo_ci64_rs",
        "defaults": ("h1_carrierid_date_lodo_ci64_full", {"override /model": "falcon_h1_carrierid_date_lodo_ci64_rs"}),
    },
}


class CiTerminalVerifierError(ValueError):
    """A terminal receipt, provenance binding, or frozen decision drifted."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CiTerminalVerifierError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _need_sha(value: Any, label: str) -> str:
    _need(_is_sha(value), f"{label} must be a lowercase SHA-256")
    return str(value)


def _finite(value: Any, label: str) -> float:
    _need(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def parse_path_maps(values: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    """Parse remote=local path-prefix maps, rejecting ambiguous mappings."""

    maps: list[tuple[str, Path]] = []
    for value in values:
        remote, marker, local = value.partition("=")
        _need(marker == "=" and remote and local, "--path-map must be REMOTE_PREFIX=LOCAL_PREFIX")
        _need(remote.startswith("/"), "remote path-map prefix must be absolute")
        maps.append((remote.rstrip("/"), Path(local).expanduser().resolve()))
    _need(len({remote for remote, _local in maps}) == len(maps), "duplicate remote path-map prefix")
    return tuple(sorted(maps, key=lambda item: len(item[0]), reverse=True))


def map_path(value: str | Path, path_maps: Sequence[tuple[str, Path]]) -> Path:
    """Map an embedded remote path onto this host without permitting traversal."""

    raw = str(value)
    _need(raw, "embedded artifact path is empty")
    for remote, local in path_maps:
        if raw == remote or raw.startswith(remote + "/"):
            suffix = raw[len(remote):].lstrip("/")
            candidate = local / suffix
            _need(".." not in candidate.parts, f"path-map traversal rejected: {raw}")
            return candidate.resolve()
    return Path(raw).expanduser().resolve()


def _read_json(path: Path, *, immutable: bool, label: str) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(), f"{label} must be a regular file: {path}")
    if immutable:
        _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} must be mode 0444: {path}")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CiTerminalVerifierError(f"{label} is invalid JSON: {path}") from exc
    _need(isinstance(body, dict), f"{label} must contain a JSON object: {path}")
    return body


def _mapped_reference(row: Mapping[str, Any], *, path_key: str, sha_key: str,
                      path_maps: Sequence[tuple[str, Path]], label: str, immutable: bool) -> tuple[Path, dict[str, Any]]:
    path = map_path(str(row.get(path_key, "")), path_maps)
    _need_sha(row.get(sha_key), f"{label}.{sha_key}")
    _need(path.is_file() and not path.is_symlink(), f"{label} path missing: {path}")
    _need(_sha_file(path) == row[sha_key], f"{label} SHA drift: {path}")
    return path, _read_json(path, immutable=immutable, label=label)


def _get(cfg: Any, dotted: str) -> Any:
    current = cfg
    for token in dotted.split("."):
        if isinstance(current, Mapping):
            current = current.get(token)
        else:
            current = getattr(current, token, None)
    return current


def _expected_arm(arm: str) -> tuple[int, str, bool]:
    _need(arm in ARMS, f"unsupported arm: {arm}")
    return (32 if arm == "CI32-FULL" else 64, arm.split("-", 1)[1].lower(), arm == "CI64-C0")


def _static_relative_parts(arm: str) -> tuple[str, str, str]:
    _need(arm in STATIC_EXPERIMENTS, f"unsupported static CI arm: {arm}")
    return ("configs", "experiment", str(STATIC_EXPERIMENTS[arm]["filename"]))


def _verify_static_experiment_config(configured: Mapping[str, Any], *, arm: str,
                                     path_maps: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    """Verify the immutable arm selector recorded in the source preflight.

    This is purposefully *not* the resolved ``.hydra/config.yaml``.  The
    former is the historical, arm-specific experiment source whose exact bytes
    were bound before training; the latter is only the checkpoint-adjacent
    execution record.
    """

    from omegaconf import OmegaConf

    _need(isinstance(configured, Mapping), f"{arm}: static experiment-config binding malformed")
    declared = map_path(str(configured.get("path", "")), path_maps)
    declared_sha = _need_sha(configured.get("sha256"), f"{arm}: static experiment-config sha256")
    _need(declared.is_file() and not declared.is_symlink(), f"{arm}: static experiment config missing")
    _need(tuple(declared.parts[-3:]) == _static_relative_parts(arm),
          f"{arm}: static experiment config canonical suffix/arm drift")
    _need(_sha_file(declared) == declared_sha, f"{arm}: static experiment config byte SHA drift")
    cfg = OmegaConf.load(declared)
    raw = OmegaConf.to_container(cfg, resolve=False)
    _need(isinstance(raw, Mapping), f"{arm}: static experiment config must be a mapping")
    phase = raw.get("phase_ci")
    _need(isinstance(phase, Mapping), f"{arm}: static experiment phase_ci missing")
    width, intervention, _zero = _expected_arm(arm)
    spec = STATIC_EXPERIMENTS[arm]
    token = arm.lower().replace("-", "_")
    defaults = raw.get("defaults")
    checks = (
        defaults == list(spec["defaults"]),
        raw.get("protocol_id") == f"h1_carrierid_date_lodo_{token}_${{phase_ci.outer_date}}_source_only_v1",
        raw.get("task_name") == f"h1_carrierid_date_lodo_{token}_${{phase_ci.outer_date}}",
        str(phase.get("arm")) == arm,
        str(phase.get("carrier_intervention")) == intervention,
    )
    _need(all(checks), f"{arm}: static experiment config semantic/arm drift")
    # The base static config deliberately contains not-yet-runnable sentinels.
    # Requiring them for CI32 is a role check: it prevents a resolved run file
    # from being laundered as the pre-training experiment specification.
    if arm == "CI32-FULL":
        _need(
            raw.get("seed") == 42 and raw.get("train") is True and raw.get("test") is False
            and raw.get("ckpt_path") is None and phase.get("outer_date") == "__REQUIRED_EXPLICIT_OUTER_DATE__"
            and phase.get("ci_preflight_path") == "__REQUIRED_IMMUTABLE_CI_SOURCE_PREFLIGHT_PATH__"
            and phase.get("five_date_aggregate_path") == "__REQUIRED_IMMUTABLE_FIVE_DATE_AGGREGATE_PATH__"
            and phase.get("warm_start_forbidden") is True
            and isinstance(raw.get("trainer"), Mapping)
            and int(raw["trainer"].get("max_epochs")) == int(raw["trainer"].get("min_epochs")) == 50
            and str(raw["trainer"].get("accelerator")) == "gpu" and str(raw["trainer"].get("devices")) == "1"
            and str(raw["trainer"].get("precision")) == "32-true",
            f"{arm}: static experiment config lost frozen source-only base contract",
        )
    return {
        "path": str(declared),
        "byte_sha256": declared_sha,
        "canonical_relative_suffix": "/".join(_static_relative_parts(arm)),
        "historical_receipt_binding": True,
    }


def _verify_resolved_config(config_path: Path, *, arm: str, date: str, preflight_path: Path,
                            five_date_aggregate_path: Path | None,
                            path_maps: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    # OmegaConf is intentionally imported only when real terminal artifacts are
    # being validated.  Schema-only preflight must not import training stacks.
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    width, intervention, zero = _expected_arm(arm)
    token = arm.lower().replace("-", "_")
    checks = (
        _get(cfg, "protocol_id") == f"h1_carrierid_date_lodo_{token}_{date}_source_only_v1",
        _get(cfg, "train") is True and _get(cfg, "test") is False and _get(cfg, "ckpt_path") is None,
        int(_get(cfg, "seed")) == 42,
        str(_get(cfg, "phase_ci.outer_date")) == date and str(_get(cfg, "phase_ci.arm")) == arm,
        str(_get(cfg, "phase_ci.carrier_intervention")) == intervention,
        map_path(str(_get(cfg, "phase_ci.ci_preflight_path")), path_maps) == preflight_path,
        str(_get(cfg, "data._target_")) == "src.data.h1_carrierid_date_lodo_ci.H1CarrierIdDateLodoCiDataModule",
        str(_get(cfg, "data.ci_arm")) == arm and str(_get(cfg, "data.carrier_intervention")) == intervention,
        map_path(str(_get(cfg, "data.ci_preflight_path")), path_maps) == preflight_path,
        str(_get(cfg, "model._target_")) == "src.models.h1_carrierid_date_lodo_ci_module.H1CarrierIdDateLodoCiLitModule",
        str(_get(cfg, "model.arm")) == arm and str(_get(cfg, "model.outer_date")) == date
        and int(_get(cfg, "model.fixed_seed")) == 42,
        map_path(str(_get(cfg, "model.ci_preflight_path")), path_maps) == preflight_path,
        str(_get(cfg, "model.net._target_")) == "src.models.components.h1_carrierid_ci_spint.H1CarrierIdCiSpint",
        int(_get(cfg, "model.net.carrier_hidden_dim")) == 32 and int(_get(cfg, "model.net.carrier_interface_dim")) == width,
        bool(_get(cfg, "model.net.zero_carrier")) is zero,
        int(_get(cfg, "trainer.max_epochs")) == int(_get(cfg, "trainer.min_epochs")) == 50,
        str(_get(cfg, "trainer.accelerator")) == "gpu" and str(_get(cfg, "trainer.devices")) == "1"
        and str(_get(cfg, "trainer.precision")) == "32-true",
        int(_get(cfg, "trainer.limit_val_batches")) == 0 and int(_get(cfg, "trainer.num_sanity_val_steps")) == 0,
        _get(cfg, "phase_ci.warm_start_forbidden") is True,
    )
    _need(all(checks), f"{date}/{arm}: resolved config violates source-only fixed seed/e49 contract")
    callback = _get(cfg, "callbacks.fixed_epoch50")
    _need(callback is not None and _get(callback, "monitor") is None
          and int(_get(callback, "every_n_epochs")) == 50
          and int(_get(callback, "save_top_k")) == -1 and _get(callback, "save_last") is False,
          f"{date}/{arm}: fixed terminal callback/selection drift")
    if five_date_aggregate_path is not None:
        for dotted in ("phase_ci.five_date_aggregate_path", "data.five_date_aggregate_path", "model.five_date_aggregate_path"):
            _need(map_path(str(_get(cfg, dotted)), path_maps) == five_date_aggregate_path,
                  f"{date}/{arm}: five-date prerequisite path drift at {dotted}")
    return {
        "path": str(config_path),
        "byte_sha256": _sha_file(config_path),
        "checkpoint_adjacent": True,
    }


def _expected_hydra_task_overrides(*, arm: str, date: str, resolved_cfg: Any) -> list[str]:
    spec = STATIC_EXPERIMENTS[arm]
    return [
        f"experiment={spec['experiment']}",
        f"phase_ci.outer_date={date}",
        f"phase_ci.phase1_preflight_path={_get(resolved_cfg, 'phase_ci.phase1_preflight_path')}",
        f"phase_ci.ci_preflight_path={_get(resolved_cfg, 'phase_ci.ci_preflight_path')}",
        f"phase_ci.five_date_aggregate_path={_get(resolved_cfg, 'phase_ci.five_date_aggregate_path')}",
        "seed=42",
        "ckpt_path=null",
        "train=true",
        "test=false",
    ]


def _verify_hydra_lineage(*, resolved_config_path: Path, run_dir: Path, arm: str, date: str,
                           path_maps: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    """Validate current Hydra composition evidence without retroactive claims.

    ``hydra.yaml`` and ``overrides.yaml`` were emitted beside the run and were
    not included in the historical checkpoint/preflight hash fields.  They are
    therefore checked and recorded as *current sidecar evidence*, never
    presented as a retrospective receipt binding.
    """

    from omegaconf import OmegaConf

    hydra_path = resolved_config_path.parent / "hydra.yaml"
    overrides_path = resolved_config_path.parent / "overrides.yaml"
    _need(hydra_path.is_file() and not hydra_path.is_symlink(), f"{date}/{arm}: Hydra sidecar missing")
    _need(overrides_path.is_file() and not overrides_path.is_symlink(), f"{date}/{arm}: Hydra overrides sidecar missing")
    hydra = OmegaConf.load(hydra_path)
    overrides = OmegaConf.load(overrides_path)
    resolved = OmegaConf.load(resolved_config_path)
    expected_task = _expected_hydra_task_overrides(arm=arm, date=date, resolved_cfg=resolved)
    choices = _get(hydra, "hydra.runtime.choices")
    _need(isinstance(choices, Mapping), f"{date}/{arm}: Hydra choices missing")
    expected_choices = {
        "experiment": STATIC_EXPERIMENTS[arm]["experiment"],
        "data": "falcon_h1_carrierid_date_lodo_ci",
        "model": STATIC_EXPERIMENTS[arm]["model_choice"],
        "callbacks": "h1_carrierid_date_lodo_phase2_terminal",
        "trainer": "gpu",
    }
    _need(all(choices.get(key) == value for key, value in expected_choices.items()),
          f"{date}/{arm}: Hydra choice lineage drift")
    _need(map_path(str(_get(hydra, "hydra.run.dir")), path_maps) == run_dir
          and map_path(str(_get(hydra, "hydra.runtime.output_dir")), path_maps) == run_dir,
          f"{date}/{arm}: Hydra run/output lineage drift")
    task = OmegaConf.to_container(_get(hydra, "hydra.overrides.task"), resolve=False)
    override_rows = OmegaConf.to_container(overrides, resolve=False)
    _need(isinstance(task, list) and task == expected_task,
          f"{date}/{arm}: Hydra task overrides missing/extra/wrong")
    _need(isinstance(override_rows, list) and override_rows == expected_task,
          f"{date}/{arm}: Hydra overrides.yaml lineage missing/extra/wrong")
    return {
        "hydra_yaml_path": str(hydra_path),
        "hydra_yaml_current_byte_sha256": _sha_file(hydra_path),
        "overrides_yaml_path": str(overrides_path),
        "overrides_yaml_current_byte_sha256": _sha_file(overrides_path),
        "historical_receipt_binding": False,
        "historical_receipt_binding_note": "Current checkpoint-adjacent sidecar evidence only; neither sidecar SHA was stored in the historical preflight/checkpoint receipts.",
    }


def _verify_source_binding(binding: Mapping[str, Any], *, date: str,
                           path_maps: Sequence[tuple[str, Path]]) -> None:
    _need(binding.get("outer_date") == date and binding.get("seed") == 42 and binding.get("epochs") == 50,
          f"{date}: source binding date/seed/epochs drift")
    _need(binding.get("target_recordings_opened") == 0 and binding.get("target_bytes_read") == 0
          and binding.get("warm_start_forbidden") is True, f"{date}: source binding target/warm-start drift")
    for key in ("source_manifest_sha256", "preflight_sha256", "batch_order_sha256", "calibration_schedule_sha256",
                "normalizer_sha256", "source_cache_sha256", "normalized_cache_sha256", "source_window_indices_sha256"):
        _need_sha(binding.get(key), f"{date}: source_binding.{key}")
    for path_key, sha_key, label in (
        ("source_manifest_path", "source_manifest_sha256", "source manifest"),
        ("preflight_path", "preflight_sha256", "phase-1 preflight"),
    ):
        path = map_path(str(binding.get(path_key, "")), path_maps)
        _need(path.is_file() and not path.is_symlink(), f"{date}: {label} missing")
        _need(_sha_file(path) == binding.get(sha_key), f"{date}: {label} SHA drift")


def _verify_preflight(path: Path, body: Mapping[str, Any], *, date: str, arm: str,
                      path_maps: Sequence[tuple[str, Path]]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    _need(body.get("schema") == PREFLIGHT_SCHEMA and body.get("status") == PREFLIGHT_STATUS and body.get("outer_date") == date,
          f"{date}/{arm}: CI preflight schema/status/date drift")
    scope, controls, fresh = body.get("scope"), body.get("source_controls"), body.get("fresh_models")
    _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
          and scope.get("target_bytes_read") == 0 and scope.get("cuda_constructed_or_launched") is False,
          f"{date}/{arm}: preflight target/GPU scope drift")
    _need(isinstance(controls, Mapping) and tuple(controls.get("all_arms", ())) == ARMS
          and controls.get("same_source_windows") is True and controls.get("same_source_schedule") is True
          and controls.get("same_source_normalizer") is True and controls.get("same_fresh_seed") == 42
          and controls.get("fixed_terminal_epoch_zero_based") == 49 and controls.get("epochs") == 50,
          f"{date}/{arm}: preflight source-control drift")
    _need(isinstance(fresh, Mapping) and set(fresh) == set(ARMS) and isinstance(fresh.get(arm), Mapping),
          f"{date}/{arm}: preflight fresh-model grid drift")
    row = fresh[arm]
    width, intervention, zero = _expected_arm(arm)
    _need(row.get("arm") == arm and row.get("fresh_seed") == 42 and row.get("interface_dim") == width
          and row.get("carrier_intervention") == intervention and row.get("zero_carrier_at_model_boundary") is zero,
          f"{date}/{arm}: preflight model identity drift")
    for key in ("initial_state_sha256", "shared_backbone_initial_state_sha256"):
        _need_sha(row.get(key), f"{date}/{arm}: preflight fresh_models.{key}")
    configurations = body.get("configuration")
    _need(isinstance(configurations, Mapping) and isinstance(configurations.get(arm), Mapping),
          f"{date}/{arm}: preflight config binding missing")
    static_evidence = _verify_static_experiment_config(configurations[arm], arm=arm, path_maps=path_maps)
    code = body.get("code_sha256")
    _need(isinstance(code, Mapping), f"{date}/{arm}: preflight code binding missing")
    for key, relative in PREFLIGHT_CODE_FILES.items():
        _need(code.get(key) == _sha_file(ROOT / relative), f"{date}/{arm}: preflight code SHA drift at {relative}")
    binding = body.get("source_binding")
    _need(isinstance(binding, Mapping), f"{date}/{arm}: preflight source binding missing")
    _verify_source_binding(binding, date=date, path_maps=path_maps)
    return row, static_evidence


def _verify_checkpoint(checker_row: Mapping[str, Any], *, arm: str, date: str,
                       checker_preflight: Mapping[str, Any], checker_aggregate: Mapping[str, Any],
                       path_maps: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    checkpoint_path = map_path(str(checker_row.get("checkpoint_path", "")), path_maps)
    config_path = map_path(str(checker_row.get("config_path", "")), path_maps)
    _need(checkpoint_path.is_file() and not checkpoint_path.is_symlink(), f"{date}/{arm}: checkpoint missing")
    _need(config_path.is_file() and not config_path.is_symlink(), f"{date}/{arm}: resolved config missing")
    _need(checkpoint_path.name == "epoch_049.ckpt"
          and checkpoint_path.parent.name == "fixed_epoch50"
          and checkpoint_path.parent.parent.name == "checkpoints",
          f"{date}/{arm}: checkpoint is not exact checkpoints/fixed_epoch50/epoch_049.ckpt")
    run_dir = checkpoint_path.parent.parent.parent
    _need(run_dir.name == arm and run_dir.parent.name == date,
          f"{date}/{arm}: checkpoint run-directory lineage drift")
    _need(config_path == run_dir / ".hydra" / "config.yaml",
          f"{date}/{arm}: resolved config is not checkpoint-adjacent .hydra/config.yaml")
    _need(checker_row.get("checkpoint_sha256") == _sha_file(checkpoint_path), f"{date}/{arm}: checkpoint SHA drift")
    _need(checker_row.get("config_sha256") == _sha_file(config_path), f"{date}/{arm}: config SHA drift")
    preflight_path, preflight = _mapped_reference(checker_preflight, path_key="path", sha_key="sha256",
                                                   path_maps=path_maps, label=f"{date}/{arm}: CI preflight", immutable=True)
    aggregate_path, _aggregate = _mapped_reference(checker_aggregate, path_key="path", sha_key="sha256",
                                                    path_maps=path_maps, label=f"{date}/{arm}: five-date aggregate",
                                                    immutable=True)
    resolved_evidence = _verify_resolved_config(config_path, arm=arm, date=date, preflight_path=preflight_path,
                                                 five_date_aggregate_path=aggregate_path, path_maps=path_maps)
    fresh, static_evidence = _verify_preflight(preflight_path, preflight, date=date, arm=arm, path_maps=path_maps)
    hydra_evidence = _verify_hydra_lineage(resolved_config_path=config_path, run_dir=run_dir, arm=arm, date=date,
                                            path_maps=path_maps)

    import torch  # real verification only; never imported by --preflight-schema-only

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping) and payload["state_dict"],
          f"{date}/{arm}: checkpoint payload/state_dict malformed")
    _need(int(payload.get("epoch", -1)) == 49 and isinstance(payload.get("global_step"), int)
          and int(payload["global_step"]) > 0, f"{date}/{arm}: checkpoint is not e49")
    for name, tensor in payload["state_dict"].items():
        _need(isinstance(tensor, torch.Tensor), f"{date}/{arm}: checkpoint non-tensor state {name}")
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            _need(bool(torch.isfinite(tensor).all().item()), f"{date}/{arm}: nonfinite checkpoint state {name}")
    meta = payload.get("h1_carrierid_date_lodo_ci")
    _need(isinstance(meta, Mapping), f"{date}/{arm}: checkpoint CI metadata missing")
    source = preflight.get("source_binding")
    _need(isinstance(source, Mapping), f"{date}/{arm}: preflight source binding missing")
    required = {
        "schema": CHECKPOINT_SCHEMA, "arm": arm, "outer_date": date, "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
        "component_initial_state_sha256": fresh.get("initial_state_sha256"),
        "shared_backbone_initial_state_sha256": fresh.get("shared_backbone_initial_state_sha256"),
        "phase2_base_source_binding_sha256": _canonical_sha(source),
        "phase1_source_manifest_sha256": source.get("source_manifest_sha256"),
        "phase1_preflight_sha256": source.get("preflight_sha256"),
        "five_date_aggregate_sha256": _sha_file(aggregate_path),
        "config_sha256": resolved_evidence["byte_sha256"], "target_optimizer_steps": 0, "target_backward_steps": 0,
        "checkpoint_warm_start": False,
    }
    for key, expected in required.items():
        _need(meta.get(key) == expected, f"{date}/{arm}: checkpoint metadata drift at {key}")
    for key in ("initial_state_sha256", "component_initial_state_sha256", "shared_backbone_initial_state_sha256",
                "ci_source_binding_sha256", "phase2_base_source_binding_sha256", "phase1_source_manifest_sha256",
                "phase1_preflight_sha256", "ci_preflight_sha256", "five_date_aggregate_sha256", "config_sha256"):
        _need_sha(meta.get(key), f"{date}/{arm}: checkpoint metadata.{key}")
    _need(meta.get("ci_preflight_sha256") == _sha_file(preflight_path), f"{date}/{arm}: checkpoint/preflight SHA drift")
    return {
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "metadata": dict(meta),
        "fresh": dict(fresh),
        "static_experiment_config": static_evidence,
        "resolved_config": resolved_evidence,
        "hydra_sidecars": hydra_evidence,
    }


def _verify_checker(path: Path, body: Mapping[str, Any], *, date: str,
                    path_maps: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    _need(body.get("schema") == CHECKER_SCHEMA and body.get("status") == CHECKER_STATUS and body.get("outer_date") == date,
          f"{date}: terminal checker schema/status/date drift")
    code = body.get("code_sha256")
    _need(isinstance(code, Mapping) and set(code) == set(CHECKER_CLOSURE_FILES), f"{date}: checker code closure drift")
    for relative in CHECKER_CLOSURE_FILES:
        _need(code.get(relative) == _sha_file(ROOT / relative), f"{date}: checker code SHA drift at {relative}")
    checker_preflight = body.get("ci_preflight")
    checker_aggregate = body.get("five_date_aggregate")
    rows = body.get("checkpoints")
    _need(isinstance(checker_preflight, Mapping) and isinstance(checker_aggregate, Mapping)
          and isinstance(rows, Mapping) and set(rows) == set(ARMS),
          f"{date}: checker preflight/five-arm grid drift")
    checked = {arm: _verify_checkpoint(rows[arm], arm=arm, date=date, checker_preflight=checker_preflight,
                                       checker_aggregate=checker_aggregate, path_maps=path_maps) for arm in ARMS}
    ci64_initial = {checked[arm]["metadata"]["initial_state_sha256"] for arm in ARMS if arm.startswith("CI64-")}
    shared = {checked[arm]["metadata"]["shared_backbone_initial_state_sha256"] for arm in ARMS}
    _need(len(ci64_initial) == 1 and len(shared) == 1, f"{date}: CI64/shared-backbone initialization mismatch")
    return checked


def _verify_evaluation(path: Path, *, date: str, path_maps: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    body = _read_json(path, immutable=True, label=f"{date}: terminal evaluation")
    _need(body.get("schema") == EVALUATION_SCHEMA and body.get("status") == f"PASS_H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_EVALUATED"
          and body.get("outer_date") == date, f"{date}: terminal evaluation schema/status/date drift")
    expected_path = map_path(str(body.get("one_shot", {}).get("canonical_output_path", "")), path_maps)
    _need(expected_path == path and body.get("one_shot", {}).get("same_date_prior_terminal_evaluation_receipts") == 0,
          f"{date}: terminal evaluation is not canonical one-shot evidence")
    _need(body.get("deployment_updates") == {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
          f"{date}: terminal evaluation records target BP/update")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping) and scope.get("formal_heldout_opened") is False
          and scope.get("minival_opened") is False and scope.get("evalai_opened") is False,
          f"{date}: terminal evaluation scope drift")
    target, metrics = body.get("target"), body.get("metrics")
    _need(isinstance(target, Mapping) and isinstance(metrics, Mapping) and set(metrics) == set(ARMS),
          f"{date}: target/five-arm metric grid drift")
    sessions = tuple(target.get("sessions", ()))
    files = target.get("files")
    shared_hash = _need_sha(target.get("shared_query_window_indices_sha256"), f"{date}: shared query hash")
    _need(sessions and len(set(sessions)) == len(sessions) and isinstance(files, Mapping) and tuple(files) == sessions,
          f"{date}: target session/file identity drift")
    for session in sessions:
        _need_sha(files[session], f"{date}: target file SHA {session}")
    checker_path, checker = _mapped_reference(body.get("terminal_checker", {}), path_key="path", sha_key="sha256",
                                              path_maps=path_maps, label=f"{date}: terminal checker", immutable=True)
    checked = _verify_checker(checker_path, checker, date=date, path_maps=path_maps)
    pooled: dict[str, float] = {}
    for arm in ARMS:
        row = metrics[arm]
        _need(isinstance(row, Mapping) and row.get("query_window_indices_sha256") == shared_hash
              and row.get("state_immutable") is True and tuple(row.get("per_session", {})) == sessions,
              f"{date}/{arm}: query or state identity drift")
        _need_sha(row.get("state_sha256_before"), f"{date}/{arm}: state before hash")
        _need(row.get("state_sha256_before") == row.get("state_sha256_after"), f"{date}/{arm}: model state mutated")
        pooled[arm] = _finite(row.get("pooled_r2"), f"{date}/{arm}: pooled R2")
        for session in sessions:
            per_session = row["per_session"].get(session)
            _need(isinstance(per_session, Mapping), f"{date}/{arm}/{session}: missing per-session metric")
            _finite(per_session.get("r2"), f"{date}/{arm}/{session}: R2")
    # The evaluator stores its checker rows too.  This cross-binding prevents a
    # syntactically valid terminal receipt from swapping one arm after checking.
    embedded = body.get("checkpoints")
    _need(isinstance(embedded, Mapping) and set(embedded) == set(ARMS), f"{date}: evaluation checkpoint grid missing")
    for arm in ARMS:
        row = embedded[arm]
        _need(isinstance(row, Mapping) and map_path(str(row.get("path", "")), path_maps) == checked[arm]["checkpoint_path"]
              and map_path(str(row.get("config_path", "")), path_maps) == checked[arm]["config_path"]
              and row.get("sha256") == _sha_file(checked[arm]["checkpoint_path"])
              and row.get("config_sha256") == _sha_file(checked[arm]["config_path"]),
              f"{date}/{arm}: evaluation/checker artifact binding drift")
    return {
        "receipt_path": str(path),
        "receipt_sha256": _sha_file(path),
        "pooled": pooled,
        "target": {"sessions": list(sessions), "files": dict(files), "query_window_indices_sha256": shared_hash},
        "configuration_evidence": {
            arm: {
                "static_experiment_config": checked[arm]["static_experiment_config"],
                "resolved_checkpoint_adjacent_config": checked[arm]["resolved_config"],
                "hydra_sidecars": checked[arm]["hydra_sidecars"],
            }
            for arm in ARMS
        },
    }


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "positive_date_count": int(np.count_nonzero(values > 0.0)),
            "negative_date_count": int(np.count_nonzero(values < 0.0)),
            "zero_date_count": int(np.count_nonzero(values == 0.0)),
            "per_date": {date: float(value) for date, value in zip(DATES, values, strict=True)}}


def _decision(historical: bool, practical: bool) -> str:
    if not historical:
        return "STOP_HISTORICAL_MECHANISM_GATE_FAILED"
    if not practical:
        return "STOP_PRACTICAL_WIDTH_THRESHOLD_FAILED"
    return "H1_H64_SLODO_ELIGIBLE_NOT_AUTHORIZED"


def _write_immutable(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    output = path.resolve()
    _need(not output.exists() and not output.is_symlink() and not os.path.lexists(str(output)),
          f"verifier refuses to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(stat.S_IMODE(output.stat().st_mode) == 0o444, f"verifier output lost immutable mode: {output}")
    return output, hashlib.sha256(encoded).hexdigest()


def verify(*, evaluation_dir: Path = DEFAULT_EVALUATION_DIR, output: Path = DEFAULT_OUTPUT,
           path_maps: Sequence[tuple[str, Path]] = ()) -> dict[str, Any]:
    """Verify the full 25-cell artifact chain and publish one immutable receipt."""

    rows: dict[str, Any] = {}
    values = {arm: [] for arm in ARMS}
    for date in DATES:
        receipt = Path(evaluation_dir).resolve() / f"H1_CARRIERID_DATE_LODO_CI_{date}_FIVE_ARM_TERMINAL_EVALUATION_v1.json"
        row = _verify_evaluation(receipt, date=date, path_maps=path_maps)
        rows[date] = row
        for arm in ARMS:
            values[arm].append(row["pooled"][arm])
    arrays = {arm: np.asarray(rows_for_arm, dtype=np.float64) for arm, rows_for_arm in values.items()}
    deltas = {
        "ci64_full_minus_ci32_full": arrays["CI64-FULL"] - arrays["CI32-FULL"],
        "ci64_full_minus_ci64_c0": arrays["CI64-FULL"] - arrays["CI64-C0"],
        "ci64_full_minus_ci64_ls": arrays["CI64-FULL"] - arrays["CI64-LS"],
        "ci64_full_minus_ci64_rs": arrays["CI64-FULL"] - arrays["CI64-RS"],
    }
    summaries = {name: _summary(delta) for name, delta in deltas.items()}
    historical_criteria = {
        name: summaries[name]["mean"] > 0.0 and summaries[name]["positive_date_count"] >= 4
        for name in ("ci64_full_minus_ci32_full", "ci64_full_minus_ci64_c0", "ci64_full_minus_ci64_ls")
    }
    historical = all(historical_criteria.values())
    practical = summaries["ci64_full_minus_ci32_full"]["mean"] >= PRACTICAL_WIDTH_THRESHOLD
    decision = _decision(historical, practical)
    body = {
        "schema": VERIFIER_SCHEMA, "status": VERIFIER_STATUS, "inference_unit": "outer_date",
        "required_outer_dates": list(DATES), "required_arms": list(ARMS), "exact_grid_cells": 25,
        "verifier_code_sha256": _sha_file(Path(__file__).resolve()),
        "per_date": rows,
        "paired_deltas": summaries,
        "historical_mechanism_gate": {
            "criteria": historical_criteria,
            "rules": [
                "mean_date(CI64-FULL - CI32-FULL) > 0 and at least 4/5 dates positive",
                "mean_date(CI64-FULL - CI64-C0) > 0 and at least 4/5 dates positive",
                "mean_date(CI64-FULL - CI64-LS) > 0 and at least 4/5 dates positive",
            ], "passed": historical,
        },
        "paper_practical_width_gate": {
            "contrast": "CI64-FULL - CI32-FULL", "threshold": PRACTICAL_WIDTH_THRESHOLD,
            "rule": "mean_date(CI64-FULL - CI32-FULL) >= +0.03", "passed": practical,
        },
        "h64_escalation_gate": {"requires_historical_mechanism_gate": True,
                                 "requires_paper_practical_width_gate": True,
                                 "passed": historical and practical, "decision": decision,
                                 "launch_authorized": False},
        "rs_control": {"contrast": "CI64-FULL - CI64-RS", "reported": True,
                       "hard_gate": False, "cannot_change_historical_or_practical_gate": True},
        "scope": {"read_only_verification": True, "nwb_opened": 0, "target_data_opened": 0,
                  "trainer_constructed": False, "cuda_constructed": False, "target_backpropagation": 0},
        "path_maps_applied": [{"remote_prefix": remote, "local_prefix": str(local)} for remote, local in path_maps],
    }
    written, digest = _write_immutable(Path(output), body)
    return {"status": VERIFIER_STATUS, "receipt_path": str(written), "receipt_sha256": digest,
            "decision": decision}


def preflight_schema_only() -> dict[str, Any]:
    """Return an in-memory, no-Torch/no-data synthetic 25-cell contract check."""

    synthetic = {
        date: {
            "outer_date": date,
            "shared_query_window_indices_sha256": hashlib.sha256(f"schema-only-query-{date}".encode()).hexdigest(),
            "metrics": {arm: {"pooled_r2": 0.50 + (0.03 if arm == "CI64-FULL" else 0.0),
                              "state_immutable": True} for arm in ARMS},
        }
        for date in DATES
    }
    _need(tuple(synthetic) == DATES and all(tuple(synthetic[date]["metrics"]) == ARMS for date in DATES),
          "schema-only synthetic grid is not exact 5x5")
    width = np.asarray([synthetic[date]["metrics"]["CI64-FULL"]["pooled_r2"]
                        - synthetic[date]["metrics"]["CI32-FULL"]["pooled_r2"] for date in DATES])
    _need(bool(np.isclose(float(width.mean()), PRACTICAL_WIDTH_THRESHOLD, rtol=0.0, atol=1e-15)),
          "schema-only practical boundary drift")
    return {"schema": "h1_carrierid_date_lodo_ci_fivedate_terminal_verifier_v2_schema_only_preflight",
            "status": "PASS_SCHEMA_ONLY_SYNTHETIC_5_DATES_X_5_ARMS_NO_TORCH_NWB_GPU_TARGET",
            "exact_grid_cells": 25, "dates": list(DATES), "arms": list(ARMS),
            "synthetic_terminal_bundle": synthetic,
            "practical_threshold_boundary": {"mean": float(width.mean()), "threshold": PRACTICAL_WIDTH_THRESHOLD,
                                             "passed": True},
            "scope": {"torch_imported": False, "nwb_opened": 0, "target_data_opened": 0,
                      "cuda_constructed": False, "writes": 0}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--path-map", action="append", default=[], metavar="REMOTE=LOCAL")
    parser.add_argument("--execute-verify", action="store_true", help="required before reading terminal artifacts")
    parser.add_argument("--preflight-schema-only", action="store_true")
    args = parser.parse_args()
    _need(not (args.execute_verify and args.preflight_schema_only), "choose one of --execute-verify or --preflight-schema-only")
    if args.preflight_schema_only:
        print(json.dumps(preflight_schema_only(), sort_keys=True))
        return
    if not args.execute_verify:
        raise SystemExit("refusing terminal artifact access: pass --execute-verify or --preflight-schema-only")
    print(json.dumps(verify(evaluation_dir=args.evaluation_dir, output=args.output,
                            path_maps=parse_path_maps(args.path_map)), sort_keys=True))


if __name__ == "__main__":
    main()
