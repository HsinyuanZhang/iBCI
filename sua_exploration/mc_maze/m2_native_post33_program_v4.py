"""Reachable Phase-C program/source closure and EOF-supersession validation.

The program receipt is deliberately a small, source-only closure.  It binds
the immutable Phase-A/B receipts through the one-byte EOF proof, the explicit
deep raw-source audit receipt, and every production-reachable Phase-C source
needed to train, evaluate, finalize, or open the matrix.  It never imports an
endpoint scorer or opens raw NWB bytes.
"""
from __future__ import annotations

import ast
from collections import deque
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from sua_exploration.mc_maze.m2_native_post33_cost_v4 import (
    validate_deep_source_audit_receipt,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    PHASE_ID,
    PROTOCOL_ID,
    file_metadata,
    require_canonical_regular_file,
    sha256_file,
    sha256_json,
)


ROOT = Path(__file__).resolve().parents[2]
PHASE_A_RECEIPT = (
    ROOT
    / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_draft_prelaunch_v2_hash_hardened_20260804/draft_scorefree_receipt.json"
)
PHASE_B_RECEIPT = (
    ROOT
    / "sua_exploration/results/m2_native_post33_phase_b_v3_scorefree_20260804/phase_b_scorefree_receipt.json"
)
EOF_SCHEMA = "m2_post33_phase_c_upstream_eof_canonicalization_v4"
PROGRAM_SCHEMA = "m2_post33_phase_c_program_receipt_v4"
EOF_EXCEPTION_SOURCE = "SPINT-main/src/data/falcon_post33_confirm_v1_datamodule.py"


# The executable roots are explicit because each is an independently
# invokable production CLI or worker. Everything below these roots is
# discovered from the source/configuration graph; it is not hand-maintained.
# A new decision-signing CLI or other production runtime root must be added
# here, followed by a reviewed regeneration of PROGRAM_EXPECTED_CLOSURE.
# The synthetic capacity-benchmark CLI is intentionally not a runtime root:
# the signed cost supplement binds each benchmark artifact, and
# validate_capacity_benchmark re-hashes its source_bindings (including the
# benchmark writer) on every supplement validation. That artifact provenance
# is therefore closed through authorization -> cost supplement, not this map.
PROGRAM_RUNTIME_ROOTS: Mapping[str, tuple[str, ...]] = {
    "matrix_and_cell_launch_clis": (
        "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py",
        "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_cell_pipeline.py",
    ),
    "open_and_finalize_clis": (
        "sua_exploration/scripts/open_m2_native_post33_phase_c_v4_stage_a.py",
        "sua_exploration/scripts/open_m2_native_post33_phase_c_v4_full.py",
        "sua_exploration/scripts/finalize_m2_native_post33_phase_c_v4_cell.py",
        "sua_exploration/scripts/finalize_m2_native_post33_phase_c_v4_matrix.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_assert_live.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_live_signer.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_supervisor.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_continue_gate.py",
        "sua_exploration/scripts/verify_m2_native_post33_phase_c_v4_artifacts.py",
    ),
    "receipt_and_audit_clis": (
        "sua_exploration/scripts/verify_m2_native_post33_upstream_eof_canonicalization_v4.py",
        "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_program_receipt.py",
        "sua_exploration/scripts/verify_m2_native_post33_phase_c_v4_program_receipt.py",
        "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_deep_source_audit.py",
        "sua_exploration/scripts/verify_m2_native_post33_phase_c_v4_deep_source_audit.py",
        "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_cost_receipt.py",
        "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_cost_supplement.py",
        "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_portable_manifest.py",
        "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_shard_manifest.py",
        "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_source_batch_audit.py",
    ),
    "evaluator_frontdoor": (
        "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v4.py",
    ),
    "spint_runtime_workers": (
        "SPINT-main/src/train_post33_phase_c_v4.py",
        "SPINT-main/src/evaluate_post33_phase_c_v4.py",
        "SPINT-main/src/audit_post33_source_batches_phase_c_v4.py",
    ),
    "t4_runtime_workers": (
        "streaming_calibration_exp/src/train_post33_phase_c_v4.py",
        "streaming_calibration_exp/src/evaluate_post33_phase_c_v4.py",
        "streaming_calibration_exp/src/audit_post33_source_batches_phase_c_v4.py",
    ),
}

# Hydra receives experiment=m2_native_post33_confirm_v4_* as a composition
# override. These plans select the final data/model/callback/hydra choices,
# rather than walking every obsolete choice in each config directory.
PROGRAM_HYDRA_PLANS: tuple[Mapping[str, str], ...] = (
    {
        "name": "spint_phase_c_v4",
        "project_root": "SPINT-main",
        "entry_config": "configs/train.yaml",
        "experiment_config": "configs/experiment/m2_native_post33_confirm_v4_spint.yaml",
    },
    {
        "name": "t4_phase_c_v4",
        "project_root": "streaming_calibration_exp",
        "entry_config": "configs/train.yaml",
        "experiment_config": "configs/experiment/m2_native_post33_confirm_v4_t4.yaml",
    },
)

# This source-only receipt writer puts the streaming project on sys.path before
# importing src. It is an explicit runtime context, not a discovery exception.
LOCAL_IMPORT_CONTEXT_OVERRIDES = {
    "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_cost_receipt.py":
        "streaming_calibration_exp",
}

# Hydra packages these defaults itself; they are outside this workspace and
# therefore cannot be part of a local signed source map.
EXTERNAL_HYDRA_DEFAULT_GROUPS = frozenset({"hydra_logging", "job_logging"})

# Frozen after reviewing the discovered graph. The audit rejects either a
# newly reachable local dependency or an obsolete listed dependency.
PROGRAM_EXPECTED_CLOSURE = frozenset((
    "SPINT-main/configs/callbacks/model_summary.yaml",
    "SPINT-main/configs/callbacks/override_epoch_step.yaml",
    "SPINT-main/configs/callbacks/post33_source_only_v4.yaml",
    "SPINT-main/configs/data/falcon_m2_post33_confirm_v4.yaml",
    "SPINT-main/configs/experiment/m2_native_post33_confirm_v4_spint.yaml",
    "SPINT-main/configs/extras/default.yaml",
    "SPINT-main/configs/hydra/post33_cell_v4.yaml",
    "SPINT-main/configs/logger/tensorboard.yaml",
    "SPINT-main/configs/model/falcon_m2_post33_confirm_v4.yaml",
    "SPINT-main/configs/paths/default.yaml",
    "SPINT-main/configs/train.yaml",
    "SPINT-main/configs/trainer/default.yaml",
    "SPINT-main/configs/trainer/gpu.yaml",
    "SPINT-main/src/__init__.py",
    "SPINT-main/src/audit_post33_source_batches_phase_c_v4.py",
    "SPINT-main/src/callbacks/__init__.py",
    "SPINT-main/src/callbacks/override_epoch_step.py",
    "SPINT-main/src/callbacks/post33_source_cost_runtime_v4.py",
    "SPINT-main/src/callbacks/post33_source_selector_v4.py",
    "SPINT-main/src/data/__init__.py",
    "SPINT-main/src/data/falcon_datamodule.py",
    "SPINT-main/src/data/falcon_post33_confirm_v1_datamodule.py",
    "SPINT-main/src/data/falcon_post33_confirm_v4_datamodule.py",
    "SPINT-main/src/evaluate_post33_phase_c_v4.py",
    "SPINT-main/src/models/__init__.py",
    "SPINT-main/src/models/components/__init__.py",
    "SPINT-main/src/models/components/spint.py",
    "SPINT-main/src/models/components/spint_cached_deployment_v4.py",
    "SPINT-main/src/models/falcon_module.py",
    "SPINT-main/src/models/falcon_post33_confirm_v3_module.py",
    "SPINT-main/src/models/falcon_post33_confirm_v4_module.py",
    "SPINT-main/src/train.py",
    "SPINT-main/src/train_post33_phase_c_v4.py",
    "SPINT-main/src/utils/__init__.py",
    "SPINT-main/src/utils/instantiators.py",
    "SPINT-main/src/utils/logging_utils.py",
    "SPINT-main/src/utils/pylogger.py",
    "SPINT-main/src/utils/rich_utils.py",
    "SPINT-main/src/utils/utils.py",
    "SPINT-main/third_party/__init__.py",
    "SPINT-main/third_party/catalyst/__init__.py",
    "SPINT-main/third_party/catalyst/distributed_sampler.py",
    "SPINT-main/third_party/falcon_challenge/__init__.py",
    "SPINT-main/third_party/falcon_challenge/filtering.py",
    "streaming_calibration_exp/configs/callbacks/post33_t4_source_only_v4.yaml",
    "streaming_calibration_exp/configs/data/falcon_m2_post33_confirm_v4.yaml",
    "streaming_calibration_exp/configs/experiment/m2_native_post33_confirm_v4_t4.yaml",
    "streaming_calibration_exp/configs/extras/default.yaml",
    "streaming_calibration_exp/configs/hydra/post33_cell_v4.yaml",
    "streaming_calibration_exp/configs/logger/tensorboard.yaml",
    "streaming_calibration_exp/configs/model/streaming_b3s_t4_post33_exact_v4.yaml",
    "streaming_calibration_exp/configs/paths/default.yaml",
    "streaming_calibration_exp/configs/train.yaml",
    "streaming_calibration_exp/configs/trainer/default.yaml",
    "streaming_calibration_exp/configs/trainer/gpu.yaml",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/audit_post33_source_batches_phase_c_v4.py",
    "streaming_calibration_exp/src/callbacks/__init__.py",
    "streaming_calibration_exp/src/callbacks/decoder_lifecycle_phase_c_v4.py",
    "streaming_calibration_exp/src/callbacks/post33_source_cost_runtime_v4.py",
    "streaming_calibration_exp/src/callbacks/post33_t4_source_selector_v4.py",
    "streaming_calibration_exp/src/data/__init__.py",
    "streaming_calibration_exp/src/data/falcon_d4_features.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "streaming_calibration_exp/src/data/falcon_k4_features.py",
    "streaming_calibration_exp/src/data/falcon_post33_confirm_v3_datamodule.py",
    "streaming_calibration_exp/src/data/falcon_post33_confirm_v4_datamodule.py",
    "streaming_calibration_exp/src/data/falcon_t4_features.py",
    "streaming_calibration_exp/src/data/validation_protocol.py",
    "streaming_calibration_exp/src/evaluate_post33_phase_c_v4.py",
    "streaming_calibration_exp/src/metrics/__init__.py",
    "streaming_calibration_exp/src/metrics/baseline.py",
    "streaming_calibration_exp/src/metrics/gate2_matrix.py",
    "streaming_calibration_exp/src/metrics/run_artifacts.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/neuron_dropout.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_cached_deployment_v4.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/falcon_module.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/streaming_post33_exact_t4_v4_module.py",
    "streaming_calibration_exp/src/train.py",
    "streaming_calibration_exp/src/train_post33_phase_c_v4.py",
    "streaming_calibration_exp/src/utils/__init__.py",
    "streaming_calibration_exp/src/utils/clean_teacher_validation.py",
    "streaming_calibration_exp/src/utils/decoder_lifecycle_phase_c_v4.py",
    "streaming_calibration_exp/src/utils/instantiators.py",
    "streaming_calibration_exp/src/utils/logging_utils.py",
    "streaming_calibration_exp/src/utils/post33_paired_teacher_phase_c_v4.py",
    "streaming_calibration_exp/src/utils/pylogger.py",
    "streaming_calibration_exp/src/utils/rich_utils.py",
    "streaming_calibration_exp/src/utils/t4_outer_runtime_phase_c_v4.py",
    "streaming_calibration_exp/src/utils/utils.py",
    "streaming_calibration_exp/third_party/__init__.py",
    "streaming_calibration_exp/third_party/catalyst/__init__.py",
    "streaming_calibration_exp/third_party/catalyst/distributed_sampler.py",
    "streaming_calibration_exp/third_party/falcon_challenge/__init__.py",
    "streaming_calibration_exp/third_party/falcon_challenge/filtering.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py",
    "sua_exploration/mc_maze/m2_native_post33_cost_v4.py",
    "sua_exploration/mc_maze/m2_native_post33_deployment_profiler_v4.py",
    "sua_exploration/mc_maze/m2_native_post33_evaluator_v4.py",
    "sua_exploration/mc_maze/m2_native_post33_openers_v4.py",
    "sua_exploration/mc_maze/m2_native_post33_phase_c_v4.py",
    "sua_exploration/mc_maze/m2_native_post33_program_v4.py",
    "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v4.py",
    "sua_exploration/scripts/finalize_m2_native_post33_phase_c_v4_cell.py",
    "sua_exploration/scripts/finalize_m2_native_post33_phase_c_v4_matrix.py",
    "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_assert_live.py",
    "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_live_signer.py",
    "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_supervisor.py",
    "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_continue_gate.py",
    "sua_exploration/scripts/open_m2_native_post33_phase_c_v4_full.py",
    "sua_exploration/scripts/open_m2_native_post33_phase_c_v4_stage_a.py",
    "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_cell_pipeline.py",
    "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py",
    "sua_exploration/scripts/verify_m2_native_post33_phase_c_v4_artifacts.py",
    "sua_exploration/scripts/verify_m2_native_post33_phase_c_v4_deep_source_audit.py",
    "sua_exploration/scripts/verify_m2_native_post33_phase_c_v4_program_receipt.py",
    "sua_exploration/scripts/verify_m2_native_post33_upstream_eof_canonicalization_v4.py",
    "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_cost_receipt.py",
    "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_cost_supplement.py",
    "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_deep_source_audit.py",
    "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_portable_manifest.py",
    "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_program_receipt.py",
    "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_shard_manifest.py",
    "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_source_batch_audit.py",
))
PROGRAM_SOURCE_PATHS = tuple(sorted(PROGRAM_EXPECTED_CLOSURE))


def _workspace_file(workspace_root: Path, relative_path: str, *, label: str) -> Path:
    """Return one non-symlink source/config file or fail closed."""
    candidate = workspace_root / relative_path
    try:
        return require_canonical_regular_file(candidate, within=workspace_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(
            f"required {label} is missing/non-canonical: {relative_path}"
        ) from exc


def _relative_path(workspace_root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            workspace_root.resolve(strict=True)
        ).as_posix()
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"program closure path escapes workspace: {path}") from exc


def _relative_candidate(workspace_root: Path, path: Path) -> str:
    """Lexical workspace-relative path for a not-yet-existing candidate."""
    try:
        return path.relative_to(workspace_root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ValueError(f"program closure candidate escapes workspace: {path}") from exc


def _project_context(workspace_root: Path, path: Path) -> str | None:
    relative = _relative_path(workspace_root, path)
    override = LOCAL_IMPORT_CONTEXT_OVERRIDES.get(relative)
    if override is not None:
        return override
    first = relative.split("/", 1)[0]
    if first in {"SPINT-main", "streaming_calibration_exp"}:
        return first
    return None


def _package_initializers(workspace_root: Path, module_path: Path) -> set[Path]:
    """Include every local package initializer executed before module_path."""
    output: set[Path] = set()
    root = workspace_root.resolve(strict=True)
    cursor = module_path.parent.resolve(strict=True)
    while cursor != root:
        initializer = cursor / "__init__.py"
        if initializer.exists():
            output.add(
                _workspace_file(
                    workspace_root,
                    _relative_path(workspace_root, initializer),
                    label="local package initializer",
                )
            )
        cursor = cursor.parent
    return output


def _module_file_from_stem(
    workspace_root: Path,
    stem: Path,
    *,
    label: str,
    require_exact: bool,
) -> set[Path]:
    """Resolve a module stem to its .py or package initializer."""
    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
        if candidate.exists():
            canonical = _workspace_file(
                workspace_root,
                _relative_path(workspace_root, candidate),
                label=label,
            )
            return {canonical, *_package_initializers(workspace_root, canonical)}
    if require_exact:
        raise ValueError(
            f"local import/config target is missing: "
            f"{_relative_candidate(workspace_root, stem)}"
        )
    return set()


def _absolute_module_paths(
    workspace_root: Path,
    module_name: str,
    *,
    source_path: Path,
    allow_target_attribute_tail: bool,
) -> set[Path]:
    """Resolve known workspace module namespaces without importing them."""
    parts = tuple(module_name.split("."))
    if not parts or not all(parts):
        raise ValueError(f"invalid module reference: {module_name!r}")
    project = _project_context(workspace_root, source_path)
    prefix = parts[0]
    if prefix == "sua_exploration":
        bases = (workspace_root,)
    elif prefix in {"src", "third_party"}:
        if project is not None:
            bases = (workspace_root / project,)
        elif (workspace_root / prefix).exists():
            bases = (workspace_root,)
        else:
            raise ValueError(
                f"ambiguous local {prefix} import without project context: {module_name}"
            )
    else:
        # Third-party/stdlib imports have no local source-map authority.
        return set()

    for base in bases:
        # A Hydra target ends with an attribute, so permit stripping that
        # attribute (and nested attributes), but never accept only the top
        # package such as src when the intended local module disappeared.
        ends = range(len(parts), 1, -1) if allow_target_attribute_tail else (len(parts),)
        for end in ends:
            resolved = _module_file_from_stem(
                workspace_root,
                base.joinpath(*parts[:end]),
                label=f"local module {module_name}",
                require_exact=False,
            )
            if resolved:
                return resolved
    context = f" in project {project}" if project is not None else ""
    raise ValueError(f"local import/config target is missing{context}: {module_name}")


def _optional_alias_module_paths(
    workspace_root: Path,
    module_name: str,
    *,
    source_path: Path,
) -> set[Path]:
    """Import aliases may name symbols; include them only when they are modules."""
    try:
        return _absolute_module_paths(
            workspace_root,
            module_name,
            source_path=source_path,
            allow_target_attribute_tail=False,
        )
    except ValueError as exc:
        if "local import/config target is missing" not in str(exc):
            raise
        return set()


def _relative_module_paths(
    workspace_root: Path,
    *,
    source_path: Path,
    level: int,
    module: str | None,
    aliases: Sequence[str] = (),
) -> set[Path]:
    """Resolve a Python from-dot import directly from its package path."""
    if level < 1:
        raise ValueError("relative import has invalid level")
    base = source_path.parent
    for _ in range(level - 1):
        base = base.parent
    output: set[Path] = set()
    if module:
        stem = base.joinpath(*module.split("."))
        output |= _module_file_from_stem(
            workspace_root, stem, label=f"relative import {module}", require_exact=True
        )
    else:
        stem = base
        output |= _module_file_from_stem(
            workspace_root, stem, label="relative package import", require_exact=True
        )
    for alias in aliases:
        output |= _module_file_from_stem(
            workspace_root,
            stem / alias,
            label=f"relative import alias {alias}",
            require_exact=False,
        )
    return output


def _python_import_paths(workspace_root: Path, source_path: Path) -> set[Path]:
    """AST-only local import discovery; no runtime source is executed."""
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError(f"cannot parse local program source: {source_path}") from exc
    output: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                output |= _absolute_module_paths(
                    workspace_root,
                    alias.name,
                    source_path=source_path,
                    allow_target_attribute_tail=False,
                )
        elif isinstance(node, ast.ImportFrom):
            aliases = tuple(alias.name for alias in node.names if alias.name != "*")
            if node.level:
                output |= _relative_module_paths(
                    workspace_root,
                    source_path=source_path,
                    level=node.level,
                    module=node.module,
                    aliases=aliases,
                )
            elif node.module:
                output |= _absolute_module_paths(
                    workspace_root,
                    node.module,
                    source_path=source_path,
                    allow_target_attribute_tail=False,
                )
                for alias in aliases:
                    output |= _optional_alias_module_paths(
                        workspace_root,
                        f"{node.module}.{alias}",
                        source_path=source_path,
                    )
    return output


def _yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot parse Hydra YAML: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Hydra config must be a mapping: {path}")
    return payload


def _default_reference(
    entry: Any,
    *,
    config_path: Path,
    config_root: Path,
) -> tuple[str, Path] | None:
    """Resolve one local Hydra default without composing or importing Hydra."""
    if entry == "_self_":
        return None
    if isinstance(entry, str):
        token = entry.strip()
        if not token or token == "_self_":
            return None
        # A bare default is relative to the including config group.
        return token, config_path.parent / f"{token}.yaml"
    if not isinstance(entry, Mapping) or len(entry) != 1:
        raise ValueError(f"unsupported Hydra defaults entry in {config_path}: {entry!r}")
    raw_group, option = next(iter(entry.items()))
    if not isinstance(raw_group, str):
        raise ValueError(f"Hydra default group is not a string in {config_path}")
    if option is None:
        return None
    if not isinstance(option, str):
        raise ValueError(f"Hydra default option is not a string in {config_path}")
    group = raw_group.strip()
    for modifier in ("override ", "optional "):
        if group.startswith(modifier):
            group = group[len(modifier):].strip()
    group = group.split("@", 1)[0]
    normalized_group = group.lstrip("/")
    if normalized_group in EXTERNAL_HYDRA_DEFAULT_GROUPS:
        return None
    return normalized_group, config_root / normalized_group / f"{option}.yaml"


def _config_targets(
    workspace_root: Path,
    config_paths: Iterable[Path],
) -> set[Path]:
    """Find all local Hydra _target_ modules recursively."""
    output: set[Path] = set()

    def visit(value: Any, config_path: Path) -> None:
        if isinstance(value, Mapping):
            target = value.get("_target_")
            if target is not None:
                if not isinstance(target, str):
                    raise ValueError(f"Hydra _target_ is not a string: {config_path}")
                output.update(
                    _absolute_module_paths(
                        workspace_root,
                        target,
                        source_path=config_path,
                        allow_target_attribute_tail=True,
                    )
                )
            for nested in value.values():
                visit(nested, config_path)
        elif isinstance(value, list):
            for nested in value:
                visit(nested, config_path)

    for config_path in config_paths:
        visit(_yaml_mapping(config_path), config_path)
    return output


def _hydra_plan_paths(
    workspace_root: Path,
    plan: Mapping[str, str],
) -> tuple[set[Path], set[Path]]:
    """Resolve final selected local YAMLs and their local target modules."""
    name = plan["name"]
    project_root = plan["project_root"]
    config_root = workspace_root / project_root / "configs"
    entry = _workspace_file(
        workspace_root,
        f"{project_root}/{plan['entry_config']}",
        label=f"Hydra entry config {name}",
    )
    experiment = _workspace_file(
        workspace_root,
        f"{project_root}/{plan['experiment_config']}",
        label=f"Hydra experiment config {name}",
    )
    selected: dict[str, Path] = {}
    config_paths: set[Path] = {entry, experiment}

    # Apply the common train defaults, then the exact experiment override
    # passed by each production wrapper. Superseded default selections are not
    # source-map entries because they are not in the final composed program.
    for config_path in (entry, experiment):
        payload = _yaml_mapping(config_path)
        defaults = payload.get("defaults")
        if defaults is None:
            continue
        if not isinstance(defaults, list):
            raise ValueError(f"Hydra defaults must be a list: {config_path}")
        for default in defaults:
            reference = _default_reference(
                default, config_path=config_path, config_root=config_root
            )
            if reference is None:
                continue
            group, candidate = reference
            selected[group] = _workspace_file(
                workspace_root,
                _relative_candidate(workspace_root, candidate),
                label=f"Hydra default {group}",
            )

    # Selected configs may have their own sibling defaults, most notably the
    # callbacks and Hydra logging configuration selected by Phase-C.
    pending: deque[Path] = deque(selected.values())
    while pending:
        config_path = pending.popleft()
        if config_path in config_paths:
            continue
        config_paths.add(config_path)
        payload = _yaml_mapping(config_path)
        defaults = payload.get("defaults")
        if defaults is None:
            continue
        if not isinstance(defaults, list):
            raise ValueError(f"Hydra defaults must be a list: {config_path}")
        for default in defaults:
            reference = _default_reference(
                default, config_path=config_path, config_root=config_root
            )
            if reference is None:
                continue
            group, candidate = reference
            del group
            pending.append(
                _workspace_file(
                    workspace_root,
                    _relative_candidate(workspace_root, candidate),
                    label="nested Hydra default",
                )
            )
    return config_paths, _config_targets(workspace_root, config_paths)


def discover_phase_c_program_closure(
    *,
    workspace_root: str | Path = ROOT,
    runtime_roots: Mapping[str, Sequence[str]] = PROGRAM_RUNTIME_ROOTS,
    hydra_plans: Sequence[Mapping[str, str]] = PROGRAM_HYDRA_PLANS,
) -> tuple[str, ...]:
    """Discover all local production source/config dependencies without import."""
    workspace = Path(workspace_root).resolve(strict=True)
    discovered: set[Path] = set()
    pending: deque[Path] = deque()
    for label, roots in runtime_roots.items():
        for relative in roots:
            source = _workspace_file(workspace, relative, label=f"runtime root {label}")
            if source.suffix != ".py":
                raise ValueError(f"runtime root must be Python source: {relative}")
            pending.append(source)
    for plan in hydra_plans:
        configs, targets = _hydra_plan_paths(workspace, plan)
        discovered.update(configs)
        pending.extend(targets)
    while pending:
        source = pending.popleft()
        if source in discovered:
            continue
        discovered.add(source)
        pending.extend(_python_import_paths(workspace, source))
    return tuple(sorted(_relative_path(workspace, path) for path in discovered))


def audit_phase_c_program_closure(
    *,
    workspace_root: str | Path = ROOT,
    runtime_roots: Mapping[str, Sequence[str]] = PROGRAM_RUNTIME_ROOTS,
    hydra_plans: Sequence[Mapping[str, str]] = PROGRAM_HYDRA_PLANS,
    expected_paths: Iterable[str] = PROGRAM_EXPECTED_CLOSURE,
) -> tuple[str, ...]:
    """Fail closed on reachable-unlisted and listed-unreachable drift."""
    observed = frozenset(
        discover_phase_c_program_closure(
            workspace_root=workspace_root,
            runtime_roots=runtime_roots,
            hydra_plans=hydra_plans,
        )
    )
    expected = frozenset(expected_paths)
    reachable_unlisted = sorted(observed - expected)
    listed_unreachable = sorted(expected - observed)
    if reachable_unlisted or listed_unreachable:
        raise ValueError(
            "Phase-C program closure allowlist mismatch; "
            f"reachable-unlisted={reachable_unlisted}; "
            f"listed-unreachable={listed_unreachable}"
        )
    return tuple(sorted(observed))


def phase_c_program_source_paths() -> tuple[str, ...]:
    """Return the audited closure used by the signed program receipt."""
    return audit_phase_c_program_closure()


def _metadata(path: str | Path) -> dict[str, Any]:
    return file_metadata(require_canonical_regular_file(path))


def _source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in phase_c_program_source_paths():
        path = require_canonical_regular_file(ROOT / relative, within=ROOT)
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _parse_receipt(path: str | Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    receipt = require_canonical_regular_file(path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} is not a mapping")
    return receipt, payload


def validate_upstream_eof_canonicalization_receipt(
    payload: Mapping[str, Any],
) -> None:
    """Revalidate the append-only Phase-A EOF exception and all frozen maps."""
    required = {
        "schema", "protocol_id", "phase_id", "append_only_supersession",
        "phase_a_receipt", "phase_b_receipt", "phase_a_non_exception_entries_exact",
        "phase_a_source_map_entries", "phase_b_entries_exact", "phase_b_source_map_entries",
        "exception", "all_other_phase_a_sources_zero_drift", "all_phase_b_sources_zero_drift",
    }
    if set(payload) != required or payload.get("schema") != EOF_SCHEMA:
        raise ValueError("EOF canonicalization receipt schema/exact set mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("phase_id") != PHASE_ID:
        raise ValueError("EOF canonicalization receipt identity mismatch")
    if (
        payload.get("append_only_supersession") is not True
        or payload.get("all_other_phase_a_sources_zero_drift") is not True
        or payload.get("all_phase_b_sources_zero_drift") is not True
    ):
        raise ValueError("EOF canonicalization receipt supersession flags failed")
    phase_a_path, phase_a = _parse_receipt(PHASE_A_RECEIPT, "Phase-A receipt")
    phase_b_path, phase_b = _parse_receipt(PHASE_B_RECEIPT, "Phase-B receipt")
    if payload.get("phase_a_receipt") != _metadata(phase_a_path):
        raise ValueError("EOF canonicalization Phase-A receipt substitution")
    if payload.get("phase_b_receipt") != _metadata(phase_b_path):
        raise ValueError("EOF canonicalization Phase-B receipt substitution")
    phase_a_map = phase_a.get("source_map")
    phase_b_map = phase_b.get("source_map")
    if not isinstance(phase_a_map, Mapping) or not isinstance(phase_b_map, Mapping):
        raise ValueError("EOF canonicalization frozen source map missing")
    if (
        payload.get("phase_a_source_map_entries") != len(phase_a_map)
        or payload.get("phase_b_source_map_entries") != len(phase_b_map)
    ):
        raise ValueError("EOF canonicalization source-map cardinality mismatch")
    phase_a_exact = 0
    for relative, expected in phase_a_map.items():
        if not isinstance(expected, Mapping):
            raise ValueError("Phase-A frozen source metadata invalid")
        source = require_canonical_regular_file(ROOT / str(relative), within=ROOT)
        if relative == EOF_EXCEPTION_SOURCE:
            continue
        if source.stat().st_size != expected.get("size_bytes") or sha256_file(source) != expected.get("sha256"):
            raise ValueError(f"Phase-A non-exception source drift: {relative}")
        phase_a_exact += 1
    phase_b_exact = 0
    for relative, expected in phase_b_map.items():
        if not isinstance(expected, Mapping):
            raise ValueError("Phase-B frozen source metadata invalid")
        source = require_canonical_regular_file(ROOT / str(relative), within=ROOT)
        if source.stat().st_size != expected.get("size_bytes") or sha256_file(source) != expected.get("sha256"):
            raise ValueError(f"Phase-B source drift: {relative}")
        phase_b_exact += 1
    if (
        payload.get("phase_a_non_exception_entries_exact") != phase_a_exact
        or payload.get("phase_b_entries_exact") != phase_b_exact
    ):
        raise ValueError("EOF canonicalization exact-entry count mismatch")
    expected = phase_a_map.get(EOF_EXCEPTION_SOURCE)
    if not isinstance(expected, Mapping):
        raise ValueError("EOF canonicalization exception source is absent")
    current = require_canonical_regular_file(ROOT / EOF_EXCEPTION_SOURCE, within=ROOT).read_bytes()
    canonical = current + b"\n"
    exception = payload.get("exception")
    if not isinstance(exception, Mapping) or set(exception) != {
        "relative_path", "current_size_bytes", "current_sha256", "canonicalization",
        "canonicalized_size_bytes", "canonicalized_sha256", "sealed_expected_size_bytes",
        "sealed_expected_sha256", "python_ast_equal", "both_variants_compile",
        "runtime_bytes_modified",
    }:
        raise ValueError("EOF canonicalization exception schema mismatch")
    current_ast = ast.dump(ast.parse(current.decode("utf-8")), include_attributes=False)
    canonical_ast = ast.dump(ast.parse(canonical.decode("utf-8")), include_attributes=False)
    compile(current, EOF_EXCEPTION_SOURCE, "exec")
    compile(canonical, EOF_EXCEPTION_SOURCE, "exec")
    expected_exception = {
        "relative_path": EOF_EXCEPTION_SOURCE,
        "current_size_bytes": len(current),
        "current_sha256": hashlib.sha256(current).hexdigest(),
        "canonicalization": "append exactly one LF byte at EOF for receipt comparison only",
        "canonicalized_size_bytes": len(canonical),
        "canonicalized_sha256": hashlib.sha256(canonical).hexdigest(),
        "sealed_expected_size_bytes": expected.get("size_bytes"),
        "sealed_expected_sha256": expected.get("sha256"),
        "python_ast_equal": current_ast == canonical_ast,
        "both_variants_compile": True,
        "runtime_bytes_modified": False,
    }
    if exception != expected_exception:
        raise ValueError("EOF canonicalization exception proof substitution")
    if (
        len(current) != int(expected["size_bytes"]) - 1
        or len(canonical) != int(expected["size_bytes"])
        or hashlib.sha256(canonical).hexdigest() != expected["sha256"]
    ):
        raise ValueError("EOF canonicalization append-only proof failed")


def build_phase_c_program_receipt(
    *,
    eof_canonicalization_receipt_path: str | Path,
    deep_source_audit_receipt_path: str | Path,
) -> dict[str, Any]:
    eof_path, eof = _parse_receipt(eof_canonicalization_receipt_path, "EOF receipt")
    validate_upstream_eof_canonicalization_receipt(eof)
    deep_path, deep = _parse_receipt(deep_source_audit_receipt_path, "deep source audit receipt")
    audit_metadata = deep.get("source_batch_audit")
    if not isinstance(audit_metadata, Mapping):
        raise ValueError("deep source audit receipt lacks its source-batch audit")
    validate_deep_source_audit_receipt(
        deep,
        source_batch_audit_path=audit_metadata.get("canonical_path", ""),
        deep_verify=False,
    )
    source_map = _source_rows()
    return {
        "schema": PROGRAM_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "absolute_workspace_root": str(ROOT.resolve()),
        "score_data_accessed": False,
        "formal_data_accessed": False,
        "gpu_used": False,
        "phase_a_b_eof_canonicalization": _metadata(eof_path),
        "deep_source_audit_receipt": _metadata(deep_path),
        "source_map": source_map,
        "source_map_sha256": sha256_json(source_map),
        "source_map_entry_count": len(source_map),
    }


def validate_phase_c_program_receipt(path: str | Path) -> Mapping[str, Any]:
    """Revalidate an exact program receipt at every production capability edge."""
    receipt_path, payload = _parse_receipt(path, "Phase-C program receipt")
    required = {
        "schema", "protocol_id", "phase_id", "absolute_workspace_root",
        "score_data_accessed", "formal_data_accessed", "gpu_used",
        "phase_a_b_eof_canonicalization", "deep_source_audit_receipt", "source_map",
        "source_map_sha256", "source_map_entry_count",
    }
    if set(payload) != required or payload.get("schema") != PROGRAM_SCHEMA:
        raise ValueError("Phase-C program receipt schema/exact set mismatch")
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("phase_id") != PHASE_ID
        or payload.get("absolute_workspace_root") != str(ROOT.resolve())
    ):
        raise ValueError("Phase-C program receipt identity/workspace mismatch")
    if any(payload.get(field) is not False for field in (
        "score_data_accessed", "formal_data_accessed", "gpu_used"
    )):
        raise ValueError("Phase-C program receipt execution-scope mismatch")
    expected_map = _source_rows()
    source_map = payload.get("source_map")
    if source_map != expected_map:
        raise ValueError("Phase-C program receipt source-map drift/omission/substitution")
    if (
        payload.get("source_map_entry_count") != len(expected_map)
        or payload.get("source_map_sha256") != sha256_json(expected_map)
    ):
        raise ValueError("Phase-C program receipt source-map digest/cardinality mismatch")
    eof_metadata = payload.get("phase_a_b_eof_canonicalization")
    if not isinstance(eof_metadata, Mapping):
        raise ValueError("Phase-C program receipt EOF proof metadata missing")
    eof_path = require_canonical_regular_file(eof_metadata.get("canonical_path", ""))
    if eof_metadata != _metadata(eof_path):
        raise ValueError("Phase-C program receipt EOF proof substitution")
    _, eof = _parse_receipt(eof_path, "EOF receipt")
    validate_upstream_eof_canonicalization_receipt(eof)
    deep_metadata = payload.get("deep_source_audit_receipt")
    if not isinstance(deep_metadata, Mapping):
        raise ValueError("Phase-C program receipt deep-audit metadata missing")
    deep_path = require_canonical_regular_file(deep_metadata.get("canonical_path", ""))
    if deep_metadata != _metadata(deep_path):
        raise ValueError("Phase-C program receipt deep-audit substitution")
    _, deep = _parse_receipt(deep_path, "deep source audit receipt")
    audit_metadata = deep.get("source_batch_audit")
    if not isinstance(audit_metadata, Mapping):
        raise ValueError("Phase-C program receipt deep audit lacks source-audit binding")
    validate_deep_source_audit_receipt(
        deep,
        source_batch_audit_path=audit_metadata.get("canonical_path", ""),
        deep_verify=False,
    )
    # Return the parsed mapping only after every closure edge has been checked.
    del receipt_path
    return payload
