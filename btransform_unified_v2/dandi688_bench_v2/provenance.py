"""Stage-specific source bindings and the explicit metadata-only B3S migration."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

from . import protocol
from .common import PACKAGE, sha256, source_hashes

_LOCAL = "btransform_unified_v2/dandi688_bench_v2/"
_SHARED_NEURAL = (
    "btransform_unified_v1/src/btransform_unified_v1/",
    "btransform_unified_v2/src/btransform_unified_v2/",
    "btransform_unified_v2/learnable_recency_v1/src/learnable_recency_v1/",
)
_BASELINE_SHARED = "btransform_unified_v2/external_baselines_v1/fair_v2/numerics.py"
_COMMON_NAME = _LOCAL + "common.py"
_LEGACY_COMMON_SHA = "f15ee90e6268cff57c6fbea1e8b01201d4156bc2aa569b78ccef205e72529cb6"
_LEGACY_COMMON = PACKAGE / "results/b3s_migration_v1/common_before_b3s.py"


def _non_encoder_metadata_ast(path: Path) -> str:
    """Only SCHEMA/RECIPE changed for the B3S model, neither drives CPU fitting."""
    module = ast.parse(Path(path).read_text())
    retained = []
    for node in module.body:
        names = [target.id for target in node.targets if isinstance(target, ast.Name)] if isinstance(node, ast.Assign) else []
        if names in (["SCHEMA"], ["RECIPE"]):
            continue
        retained.append(node)
    module.body = retained
    return ast.dump(module, include_attributes=False)


def baseline_common_compatibility(recorded_sha: str, *, current_path: Path | None = None,
                                  snapshot_path: Path | None = None) -> dict[str, Any] | None:
    """Accept the one recorded B3S metadata migration, never changed helper code."""
    current_path = PACKAGE / "common.py" if current_path is None else Path(current_path)
    snapshot_path = _LEGACY_COMMON if snapshot_path is None else Path(snapshot_path)
    if recorded_sha != _LEGACY_COMMON_SHA or not snapshot_path.is_file() or sha256(snapshot_path) != recorded_sha:
        return None
    if _non_encoder_metadata_ast(current_path) != _non_encoder_metadata_ast(snapshot_path):
        return None
    return {"kind": "B3S_SCHEMA_RECIPE_ONLY", "recorded_sha256": recorded_sha,
            "current_sha256": sha256(current_path), "before_source": str(snapshot_path.resolve()),
            "excluded_top_level_metadata": ["SCHEMA", "RECIPE"],
            "all_other_python_ast_unchanged": True}


def execution_dependencies(kind: str, current: Mapping[str, str]) -> dict[str, str]:
    baseline = kind == "baseline"
    local_names = {"protocol.py", "data.py", "carrier.py", "common.py"}
    if baseline:
        local_names |= {"baselines.py", "baseline_runner.py"}
    else:
        local_names |= {"model.py", "training.py"}
        if kind == "static":
            local_names |= {"baselines.py", "baseline_runner.py"}
    required = {_LOCAL + name for name in local_names}
    if baseline or kind == "static":
        required.add(_BASELINE_SHARED)
    if not baseline:
        required.update(name for name in current if name.startswith(_SHARED_NEURAL))
    missing = required - set(current)
    if missing:
        raise ValueError(f"execution source files are missing: {sorted(missing)}")
    return {name: current[name] for name in sorted(required)}


def verify_execution_hashes(receipt: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    recorded = receipt.get("code_hashes", receipt.get("code"))
    if not isinstance(recorded, dict):
        raise ValueError(f"{kind} receipt lacks code hashes")
    required = execution_dependencies(kind, source_hashes())
    migrations = {}
    for name, expected in required.items():
        actual = recorded.get(name)
        if actual == expected:
            continue
        migration = baseline_common_compatibility(actual) if kind == "baseline" and name == _COMMON_NAME else None
        if migration is None:
            raise ValueError(f"{kind} execution source hash drift: {name}")
        migrations[name] = migration
    return {"stage": kind, "dependencies": required, "explicit_metadata_migrations": migrations}


def bind_execution_sources(artifacts: dict[str, str]) -> dict[str, Any]:
    """Seal current scorer, model, feature and shared implementation bytes."""
    current = source_hashes()
    for name, value in current.items():
        path = str((protocol.WORKSPACE_ROOT / name).resolve())
        previous = artifacts.setdefault(path, value)
        if previous != value:
            raise ValueError(f"execution source changed while sealing: {name}")
    if _LEGACY_COMMON.is_file():
        artifacts[str(_LEGACY_COMMON.resolve())] = sha256(_LEGACY_COMMON)
    return {"code_hashes": current, "legacy_baseline_common_snapshot": str(_LEGACY_COMMON.resolve())}
