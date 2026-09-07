#!/usr/bin/env python3
"""Append-only, fail-closed RT L-D device handoff (v4).

This is intentionally independent of the v1--v3 handoff scripts.  Merely
running it is inert: ``--prepare`` creates review artifacts; GPU work requires
the explicit, paired ``--watch --execute --run-root`` invocation.  We do not
signal or wait for H1.  A static H1 partition is released only when its exact
parent has exited, no replacement static runner owns that device, and two
exact-device compute-app samples are empty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
STREAMING = ROOT / "streaming_calibration_exp"
V2_RECEIPT = ROOT / "sua_exploration/results/rt_ld_fold0_preflight_v2/RT_LD_CPU_PREFLIGHT_RECEIPT_v2.json"
V2_PLAN = ROOT / "sua_exploration/results/rt_ld_fold0_preflight_v2/RT_LD_FOLD0_LAUNCH_PLAN_v2.json"
RESULT_DIR = ROOT / "sua_exploration/results/rt_ld_device_handoff_v4"
DRIFT = RESULT_DIR / "RT_LD_V2_CODE_DRIFT_SUPPLEMENT_v4.json"
PLAN = RESULT_DIR / "RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v4.json"
V2_RECEIPT_SHA = "0e3bdb78038f975dce693f5c85d82b48fe8e97e697e645549144990152b09dfd"
V2_PLAN_SHA = "1e5bbb2e20f4d30290a20e715ebac46751dbf4871e6bddf8212679bc89b26888"
SELECTION_CONTRACT_TEST = ROOT / "streaming_calibration_exp/tests/test_rt_ld_selection_receipt_contract.py"

# The experiment name, final Hydra identity, and source of only the gain
# branch are all independently asserted before launch and after fit.
ARM_SPECS: dict[str, dict[str, str]] = {
    "rt_ld_a0_full_m24_fold0_seed42": {
        "directory": "01_a0",
        "run_id": "rt_ld_a0_full_m24_fold0_seed42",
        "model_rt_ld_arm": "a0",
        "data_rt_ld_gain_source": "full",
    },
    "rt_ld_g_full_m24_fold0_seed42": {
        "directory": "02_g_full",
        "run_id": "rt_ld_g_full_m24_fold0_seed42",
        "model_rt_ld_arm": "g_full",
        "data_rt_ld_gain_source": "full",
    },
    "rt_ld_g_xls_m24_fold0_seed42": {
        "directory": "03_g_xls",
        "run_id": "rt_ld_g_xls_m24_fold0_seed42",
        "model_rt_ld_arm": "g_xls",
        "data_rt_ld_gain_source": "xls_v2",
    },
}
ARM_ORDER = list(ARM_SPECS)
DRIFT_PATHS = {
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py": "4-to-50 live activity gain with zero Parameter construction that preserves CPU RNG",
    "streaming_calibration_exp/src/models/components/streaming_spint.py": "decoder-side carrier-times-live-activity gain injection and cache state",
    "streaming_calibration_exp/tests/test_rt_ld_gain.py": "exact null, CPU RNG, and CUDA-noninitialization regression coverage",
    "streaming_calibration_exp/configs/experiment/rt_ld_a0_full_m24_fold0_seed42.yaml": "fit-end clean nested selection receipt callback binding",
}

# These process identities are the allocated partition owners, not transient
# train/source-executor children.  A different matching static parent blocks
# the device just as firmly as the expected parent being alive.
PARTITIONS: dict[str, dict[str, Any]] = {
    "gpu0_h1_static_ci64": {
        "device_index": 0,
        "runner_pid": 2814205,
        "runner_starttime": "160906485",
        "runner_cmdline": "scripts/h1_carrierid_date_lodo_ci_static_partition_runner.sh 0",
    },
    "gpu1_h1_static_ci64": {
        "device_index": 1,
        "runner_pid": 2814209,
        "runner_starttime": "160906485",
        "runner_cmdline": "scripts/h1_carrierid_date_lodo_ci_static_partition_runner.sh 1",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _yaml(path: Path) -> dict[str, Any]:
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected YAML object: {path}")
    return result


def _absolute_run_root(value: Path) -> Path:
    """Resolve before any existence check, directory creation, or spawning."""
    return value.expanduser().resolve()


def _validate_v2_anchor() -> dict[str, Any]:
    if _sha(V2_RECEIPT) != V2_RECEIPT_SHA or _sha(V2_PLAN) != V2_PLAN_SHA:
        raise ValueError("sealed v2 preflight SHA drift")
    receipt, plan = _json(V2_RECEIPT), _json(V2_PLAN)
    if receipt.get("status") != "CPU_PASS_QUEUE_READY":
        raise ValueError("v2 CPU preflight did not pass")
    if plan.get("receipt_sha256") != V2_RECEIPT_SHA or len(receipt.get("artifacts", {})) != 14:
        raise ValueError("v2 receipt/plan artifact binding invalid")
    expected = [spec["run_id"] for spec in ARM_SPECS.values()]
    actual = [receipt["arms"][label]["experiment"] for label in ("A0", "G-Full", "G-XLS")]
    if actual != expected or receipt["frozen_score_gates"].get("fold") != 0 or receipt["frozen_score_gates"].get("seed") != 42:
        raise ValueError("v2 arm/fold/seed anchor drift")
    return receipt


def _validate_drift(receipt: Mapping[str, Any], drift: Mapping[str, Any]) -> None:
    rows = drift.get("artifact_drift")
    if not isinstance(rows, Mapping) or set(rows) != set(DRIFT_PATHS):
        raise ValueError("v4 drift supplement has an unapproved path set")
    for relative, v2_sha in receipt["artifacts"].items():
        actual = _sha(ROOT / relative)
        if relative in DRIFT_PATHS:
            row = rows[relative]
            if not isinstance(row, Mapping) or row.get("v2_sha256") != v2_sha or row.get("replacement_sha256") != actual:
                raise ValueError(f"v4 drift binding mismatch: {relative}")
        elif actual != v2_sha:
            raise ValueError(f"unexpected relative-to-v2 drift: {relative}")


def _plan_self_check(plan: Mapping[str, Any]) -> None:
    own = Path(__file__).resolve()
    required = {
        "schema": "rt_ld_device_handoff_supplemental_plan_v4",
        "status": "REVIEW_REQUIRED_NOT_ARMED",
        "runner_path": str(own.relative_to(ROOT)),
        "runner_sha256": _sha(own),
        "test_path": "sua_exploration/tests/test_rt_ld_device_handoff_v4.py",
        "test_sha256": _sha(ROOT / "sua_exploration/tests/test_rt_ld_device_handoff_v4.py"),
        "selection_contract_test_path": str(SELECTION_CONTRACT_TEST.relative_to(ROOT)),
        "selection_contract_test_sha256": _sha(SELECTION_CONTRACT_TEST),
    }
    for key, expected in required.items():
        if plan.get(key) != expected:
            raise ValueError(f"v4 plan self-check failed: {key}")
    if plan.get("gpu_launched") is not False or plan.get("formal_heldout_opened") is not False:
        raise ValueError("v4 plan is not inert")


def prepare() -> None:
    """Create sealed v4 review artifacts once; never overwrite a result set."""
    if RESULT_DIR.exists():
        raise FileExistsError("v4 handoff review artifacts are append-only")
    receipt = _validate_v2_anchor()
    drift = {
        "schema": "rt_ld_v2_code_drift_supplement_v4",
        "status": "APPEND_ONLY_REPLACEMENT_BINDING",
        "v2_receipt_sha256": V2_RECEIPT_SHA,
        "v2_plan_sha256": V2_PLAN_SHA,
        "v2_receipt_remains_immutable": True,
        "artifact_drift": {
            relative: {
                "reason": reason,
                "v2_sha256": receipt["artifacts"][relative],
                "replacement_sha256": _sha(ROOT / relative),
            }
            for relative, reason in DRIFT_PATHS.items()
        },
    }
    _validate_drift(receipt, drift)
    RESULT_DIR.mkdir(parents=True)
    DRIFT.write_text(json.dumps(drift, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan = {
        "schema": "rt_ld_device_handoff_supplemental_plan_v4",
        "status": "REVIEW_REQUIRED_NOT_ARMED",
        "v2_receipt_sha256": V2_RECEIPT_SHA,
        "v2_plan_sha256": V2_PLAN_SHA,
        "code_drift_supplement_path": str(DRIFT.relative_to(ROOT)),
        "code_drift_supplement_sha256": _sha(DRIFT),
        "runner_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": _sha(Path(__file__).resolve()),
        "test_path": "sua_exploration/tests/test_rt_ld_device_handoff_v4.py",
        "test_sha256": _sha(ROOT / "sua_exploration/tests/test_rt_ld_device_handoff_v4.py"),
        "selection_contract_test_path": str(SELECTION_CONTRACT_TEST.relative_to(ROOT)),
        "selection_contract_test_sha256": _sha(SELECTION_CONTRACT_TEST),
        "partitions": PARTITIONS,
        "release_rule": "expected static parent exited, no replacement static runner for exact device, then two consecutive empty exact-device compute-app samples",
        "fixed_arm_mapping": ARM_SPECS,
        "fixed_training_order": ARM_ORDER,
        "seed": 42,
        "fold": 0,
        "frozen_score_gates": receipt["frozen_score_gates"],
        "fail_closed": True,
        "never_signal_h1": True,
        "never_wait_for_other_partition": True,
        "selection_contract": "exact absolute best_model_path must exist beneath that arm_dir/checkpoints; cross-arm paths rejected",
        "post_fit_contract": "resolved Hydra config, selection receipt, and split manifest must agree on run_id/arm/gain/fold/seed/callback/rt_ld; outer JSON repeats run_id and rt_ld gain",
        "gpu_launched": False,
        "formal_heldout_opened": False,
    }
    PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(DRIFT, 0o444)
    os.chmod(PLAN, 0o444)


def _runner_state(partition: Mapping[str, Any]) -> str:
    pid = int(partition["runner_pid"])
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return "exited"
    try:
        stat = (proc / "stat").read_text(encoding="utf-8").split()
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return "exited"
    if len(stat) < 22 or stat[21] != str(partition["runner_starttime"]) or str(partition["runner_cmdline"]) not in cmdline:
        return "identity_drift"
    return "active"


def _replacement_static_runner_present(device: int, expected_pid: int) -> bool:
    needle = f"h1_carrierid_date_lodo_ci_static_partition_runner.sh {device}"
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
            if pid != expected_pid and needle in (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"):
                return True
        except (OSError, ValueError):
            continue
    return False


def _compute_apps(device: int) -> list[str] | None:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(device), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip() and line.strip() != "No running processes found"]


def _eligible_from_probes(first: tuple[str, bool, list[str] | None], second: tuple[str, bool, list[str] | None]) -> bool:
    return first[0] == second[0] == "exited" and first[1] is second[1] is False and first[2] == second[2] == []


def eligible(name: str, stability_seconds: float = 1.0) -> bool:
    plan, receipt = _json(PLAN), _validate_v2_anchor()
    _plan_self_check(plan)
    if plan.get("code_drift_supplement_sha256") != _sha(DRIFT):
        raise ValueError("v4 code-drift supplement SHA mismatch")
    _validate_drift(receipt, _json(DRIFT))
    partition = plan.get("partitions", {}).get(name)
    if not isinstance(partition, Mapping):
        raise ValueError(f"unknown partition: {name}")
    def probe() -> tuple[str, bool, list[str] | None]:
        device = int(partition["device_index"])
        return (_runner_state(partition), _replacement_static_runner_present(device, int(partition["runner_pid"])), _compute_apps(device))
    first = probe()
    time.sleep(stability_seconds)
    second = probe()
    return _eligible_from_probes(first, second)


def _callback_contract(cfg: Mapping[str, Any]) -> None:
    callback = cfg.get("callbacks", {}).get("rt_nested_selection_receipt")
    if not isinstance(callback, Mapping):
        raise ValueError("missing RT nested selection receipt callback")
    expected = {
        "_target_": "src.callbacks.rt_nested_selection_receipt.RtNestedSelectionReceipt",
        "monitor": "val_heldin/r2_mean",
    }
    for key, value in expected.items():
        if callback.get(key) != value:
            raise ValueError(f"selection callback contract drift: {key}")
    output = str(callback.get("output_path", ""))
    split = str(callback.get("split_manifest_path", ""))
    config = str(callback.get("config_path", ""))
    # Compose resolves paths (to ``None/...`` without a Hydra runtime); actual
    # fit config resolves them to the absolute arm directory.  In both cases
    # the callback must retain the exact three artifact basenames.
    if not output.endswith("/rt_nested_selection_receipt.json") or not split.endswith("/split_manifest.json") or not config.endswith("/.hydra/config.yaml"):
        raise ValueError("selection callback artifact path contract drift")


def _validate_composed_arm(experiment: str) -> None:
    """Perform a real Hydra compose, then parse and bind all fixed identifiers."""
    spec = ARM_SPECS[experiment]
    command = [sys.executable, "src/train.py", f"experiment={experiment}", "--cfg", "job", "--resolve"]
    result = subprocess.run(command, cwd=STREAMING, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"Hydra compose failed for {experiment}: {result.stderr[-1000:]}")
    cfg = yaml.safe_load(result.stdout)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Hydra compose did not return a mapping: {experiment}")
    if str(cfg.get("run_id")) != spec["run_id"]:
        raise ValueError(f"composed run_id mismatch: {experiment}")
    if str(cfg.get("model", {}).get("rt_ld_arm")) != spec["model_rt_ld_arm"]:
        raise ValueError(f"composed model arm mismatch: {experiment}")
    if str(cfg.get("data", {}).get("rt_ld_gain_source")) != spec["data_rt_ld_gain_source"]:
        raise ValueError(f"composed gain source mismatch: {experiment}")
    if int(cfg.get("data", {}).get("outer_loso_fold", -1)) != 0 or int(cfg.get("seed", -1)) != 42:
        raise ValueError(f"composed fold/seed mismatch: {experiment}")
    _callback_contract(cfg)


def validate_all_composed() -> None:
    for experiment in ARM_ORDER:
        _validate_composed_arm(experiment)


def _under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_fit_artifacts(arm_dir: Path, experiment: str) -> Path:
    """Reject any mismatched post-fit provenance before outer evaluation."""
    spec = ARM_SPECS[experiment]
    arm_dir = arm_dir.resolve()
    selection_path = arm_dir / "rt_nested_selection_receipt.json"
    split_path = arm_dir / "split_manifest.json"
    config_path = arm_dir / ".hydra/config.yaml"
    if not all(path.is_file() for path in (selection_path, split_path, config_path)):
        raise FileNotFoundError("required exact post-fit artifact is absent")
    selection, split, cfg = _json(selection_path), _json(split_path), _yaml(config_path)
    if str(cfg.get("run_id")) != spec["run_id"] or str(cfg.get("model", {}).get("rt_ld_arm")) != spec["model_rt_ld_arm"] or str(cfg.get("data", {}).get("rt_ld_gain_source")) != spec["data_rt_ld_gain_source"]:
        raise ValueError("resolved Hydra RT-LD mapping mismatch")
    if int(cfg.get("data", {}).get("outer_loso_fold", -1)) != 0 or int(cfg.get("seed", -1)) != 42:
        raise ValueError("resolved Hydra fold/seed mismatch")
    _callback_contract(cfg)
    required_selection = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1",
        "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "run_id": spec["run_id"],
        "arm": "afc4_vel",
        "outer_loso_fold": 0,
        "seed": 42,
        "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only",
    }
    for key, expected in required_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"selection receipt mismatch: {key}")
    checkpoint_text = selection.get("best_model_path")
    checkpoint = Path(str(checkpoint_text))
    if not checkpoint.is_absolute() or not checkpoint.is_file() or not _under(checkpoint, arm_dir / "checkpoints"):
        raise ValueError("best_model_path must be an existing absolute checkpoint beneath this arm_dir/checkpoints")
    if Path(str(selection.get("run_dir", ""))).resolve() != arm_dir:
        raise ValueError("selection run directory mismatches arm directory")
    rt_ld = split.get("rt_ld")
    expected_gain_carrier = "aligned_full_afc4" if spec["data_rt_ld_gain_source"] == "full" else "strong_xls_v2"
    if not isinstance(rt_ld, Mapping) or split.get("validation_protocol") != "nested_loso" or split.get("requested_side_feature_group") != "afc4_vel":
        raise ValueError("split manifest does not prove clean RT nested selection")
    if rt_ld.get("identity_carrier") != "aligned_full_afc4" or rt_ld.get("gain_carrier") != expected_gain_carrier or rt_ld.get("identity_never_receives_xls_v2") is not True:
        raise ValueError("split manifest RT-LD identity/gain contract mismatch")
    nested = split.get("nested_selection", {})
    if nested.get("checkpoint_metric") != "val_heldin/r2_mean" or nested.get("inner_validation_only_for_checkpoint_selection") is not True:
        raise ValueError("split manifest nested selection contract mismatch")
    return checkpoint


def _validate_outer_artifact(outer_path: Path, experiment: str) -> None:
    spec = ARM_SPECS[experiment]
    outer = _json(outer_path)
    expected_gain_carrier = "aligned_full_afc4" if spec["data_rt_ld_gain_source"] == "full" else "strong_xls_v2"
    if outer.get("run_id") != spec["run_id"] or outer.get("arm") != "afc4_vel":
        raise ValueError("outer JSON run/arm mismatch")
    rt_ld = outer.get("rt_ld")
    if not isinstance(rt_ld, Mapping) or rt_ld.get("identity_carrier") != "aligned_full_afc4" or rt_ld.get("gain_carrier") != expected_gain_carrier:
        raise ValueError("outer JSON RT-LD gain mismatch")
    if outer.get("formal_heldout_opened") is not False:
        raise ValueError("outer JSON unexpectedly claims a formal endpoint")


def execute(name: str, run_root: Path) -> None:
    """Explicit launch path.  This function is never reached by default."""
    absolute_root = _absolute_run_root(run_root)
    if absolute_root.exists() and any(absolute_root.iterdir()):
        raise FileExistsError("run root must be empty; exact artifacts only, no globbing")
    # Compose every arm before *any* mkdir or spawn, preventing a partial set.
    validate_all_composed()
    absolute_root.mkdir(parents=True, exist_ok=True)
    partition = _json(PLAN)["partitions"][name]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(partition["device_index"]))
    for experiment, spec in ARM_SPECS.items():
        arm_dir = (absolute_root / spec["directory"]).resolve()
        if arm_dir.parent != absolute_root:
            raise ValueError("arm directory escapes resolved run root")
        if not eligible(name):
            raise RuntimeError("TOCTOU revalidation refused arm spawn")
        train = subprocess.run(
            [sys.executable, "src/train.py", f"experiment={experiment}", "seed=42", f"hydra.run.dir={arm_dir}"],
            cwd=STREAMING, env=env, check=False,
        )
        if train.returncode:
            raise RuntimeError(f"training failed: {experiment}")
        checkpoint = _validate_fit_artifacts(arm_dir, experiment)
        outer = arm_dir / "rt_ld_outer_eval.json"
        evaluated = subprocess.run(
            [sys.executable, "src/rt_clean_nested_loso_eval.py", "--config", str(arm_dir / ".hydra/config.yaml"), "--checkpoint", str(checkpoint), "--split-manifest", str(arm_dir / "split_manifest.json"), "--selection-receipt", str(arm_dir / "rt_nested_selection_receipt.json"), "--output", str(outer), "--device", "cuda:0"],
            cwd=STREAMING, env=env, check=False,
        )
        if evaluated.returncode or not outer.is_file():
            raise RuntimeError(f"outer evaluation failed: {experiment}")
        _validate_outer_artifact(outer, experiment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="create immutable review artifacts; no GPU work")
    parser.add_argument("--partition", choices=sorted(PARTITIONS))
    parser.add_argument("--validate", action="store_true", help="perform read-only release eligibility check")
    parser.add_argument("--watch", action="store_true", help="watch only when paired with --execute")
    parser.add_argument("--execute", action="store_true", help="requires --watch and explicit --run-root")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if not args.partition or not (args.validate or args.watch):
        parser.error("use --prepare or --partition with --validate/--watch")
    if args.execute and not args.watch:
        parser.error("--execute is inert unless paired with --watch")
    if args.watch and (not args.execute or args.run_root is None):
        parser.error("--watch requires explicit --execute and --run-root")
    if args.watch:
        while not eligible(args.partition):
            time.sleep(args.poll_seconds)
        execute(args.partition, args.run_root)
        return
    print(json.dumps({"partition": args.partition, "eligible": eligible(args.partition)}))


if __name__ == "__main__":
    main()
