#!/usr/bin/env python3
"""CPU-only CTXV2 Stage A experiment-config parity verifier.

Loads the three Stage A configs and their sealed fold-0 counterparts via Hydra
compose (train.yaml + experiment=...) and fails closed on any disallowed diff.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.h1_m4_eb_normalized_v2_contract import sha256_file, write_immutable_json

RECEIPT_SCHEMA = "ctxv2_stage_a_config_parity_v1"
DEFAULT_RECEIPT = (
    ROOT / "pilot_artifacts/h1_ctxv2_19250108/CTXV2_STAGE_A_CONFIG_PARITY_v1.json"
)

STAGE_A_ARMS: dict[str, dict[str, str]] = {
    "context_full": {
        "new": "h1_ctxv2_context_full_19250108",
        "sealed": "h1_context_event_carrier_full",
    },
    "hse5_full": {
        "new": "h1_ctxv2_hse5_full_19250108",
        "sealed": "h1_sparse_event_endpoint_full",
    },
    "zero5": {
        "new": "h1_ctxv2_zero5_19250108",
        "sealed": "h1_context_event_carrier_zero",
    },
}

ARM_TO_ARM_PERMITTED_PATHS = frozenset(
    {
        "defaults",
        "task_name",
        "protocol_id",
        "tags",
        "pilot.arm",
        "pilot.zero_carrier",
        "pilot.shared_cache_dir",
    }
)

DATE_TO_DATE_PERMITTED_PATHS = frozenset(
    {
        "task_name",
        "protocol_id",
        "tags",
        "pilot.fold_date",
        "pilot.shared_cache_dir",
    }
)

ARM_TO_ARM_SCALAR_PATHS = (
    "seed",
    "trainer.max_epochs",
    "trainer.min_epochs",
    "trainer.precision",
    "pilot.batch_size",
    "pilot.calibration_n_trials",
    "pilot.fixed_terminal_epochs",
    "pilot.no_checkpoint_selection",
)

ARM_TO_ARM_OPTIONAL_SCALAR_PATHS = ("pilot.side_dim",)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            child = value[key]
            if isinstance(child, dict):
                out.update(_flatten(child, path))
            else:
                out[path] = child
        return out
    return {prefix: value} if prefix else {}


def _experiment_yaml_path(name: str) -> Path:
    return ROOT / "configs/experiment" / f"{name}.yaml"


def compose_experiment(name: str) -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="train.yaml", overrides=[f"experiment={name}"])


def load_experiment_yaml(name: str) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(_experiment_yaml_path(name)), resolve=False)


def diff_paths(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    permitted: frozenset[str],
) -> dict[str, dict[str, Any]]:
    all_keys = sorted(set(left) | set(right))
    illegal: dict[str, dict[str, Any]] = {}
    permitted_diffs: dict[str, dict[str, Any]] = {}
    for key in all_keys:
        lv = left.get(key, "<missing>")
        rv = right.get(key, "<missing>")
        if lv == rv:
            continue
        entry = {"left": lv, "right": rv}
        if key in permitted:
            permitted_diffs[key] = entry
        else:
            illegal[key] = entry
    return {"illegal": illegal, "permitted": permitted_diffs}


def _arm_to_arm_view(cfg: DictConfig, *, experiment_name: str) -> dict[str, Any]:
    """Extract only the subtrees that must match arm-to-arm."""
    experiment = load_experiment_yaml(experiment_name)
    net = OmegaConf.to_container(cfg.model.net, resolve=False)
    if isinstance(net, dict):
        net = dict(net)
        net.pop("zero_carrier", None)
    return _flatten(
        {
            "seed": experiment["seed"],
            "trainer": experiment["trainer"],
            "pilot": experiment["pilot"],
            "model.net": net,
            "defaults": experiment["defaults"],
            "task_name": experiment["task_name"],
            "protocol_id": experiment["protocol_id"],
            "tags": experiment["tags"],
        }
    )


def compare_arm_to_arm(
    cfgs: dict[str, DictConfig],
    *,
    experiment_names: dict[str, str],
) -> dict[str, Any]:
    names = list(cfgs)
    reference = names[0]
    ref_flat = _arm_to_arm_view(cfgs[reference], experiment_name=experiment_names[reference])
    pairwise: dict[str, Any] = {}
    illegal_union: dict[str, dict[str, Any]] = {}
    permitted_union: dict[str, dict[str, Any]] = {}

    for other in names[1:]:
        other_flat = _arm_to_arm_view(
            cfgs[other], experiment_name=experiment_names[other]
        )
        left = {k: v for k, v in ref_flat.items() if k != "pilot.side_dim"}
        right = {k: v for k, v in other_flat.items() if k != "pilot.side_dim"}
        result = diff_paths(left, right, permitted=ARM_TO_ARM_PERMITTED_PATHS)
        pairwise[f"{reference}_vs_{other}"] = result
        illegal_union.update(result["illegal"])
        permitted_union.update(result["permitted"])

    scalar_checks: dict[str, Any] = {}
    for path in ARM_TO_ARM_SCALAR_PATHS:
        values = {
            name: _arm_to_arm_view(cfgs[name], experiment_name=experiment_names[name]).get(
                path, "<missing>"
            )
            for name in names
        }
        scalar_checks[path] = {"values": values, "pass": len(set(values.values())) == 1}

    side_dim_values = {
        name: _arm_to_arm_view(cfgs[name], experiment_name=experiment_names[name]).get(
            "pilot.side_dim", "<absent>"
        )
        for name in names
        if _arm_to_arm_view(cfgs[name], experiment_name=experiment_names[name]).get(
            "pilot.side_dim", "<absent>"
        )
        != "<absent>"
    }
    side_dim_pass = len(set(side_dim_values.values())) <= 1 and (
        not side_dim_values or all(v == 5 for v in side_dim_values.values())
    )
    scalar_checks["pilot.side_dim"] = {"values": side_dim_values, "pass": side_dim_pass}

    trainer_blocks = {
        name: load_experiment_yaml(experiment_names[name])["trainer"] for name in names
    }
    net_blocks = {
        name: _arm_to_arm_view(cfgs[name], experiment_name=experiment_names[name])
        for name in names
    }
    net_pass = all(
        {
            key: value
            for key, value in net_blocks[name].items()
            if key.startswith("model.net.")
        }
        == {
            key: value
            for key, value in net_blocks[reference].items()
            if key.startswith("model.net.")
        }
        for name in names[1:]
    )

    trainer_pass = len({json.dumps(v, sort_keys=True) for v in trainer_blocks.values()}) == 1

    scalar_pass = all(item["pass"] for item in scalar_checks.values())
    pass_ = (
        not illegal_union
        and scalar_pass
        and trainer_pass
        and net_pass
    )
    return {
        "pass": pass_,
        "permitted_difference_paths": sorted(ARM_TO_ARM_PERMITTED_PATHS),
        "compared_scalar_paths": list(ARM_TO_ARM_SCALAR_PATHS) + ["pilot.side_dim"],
        "scalar_checks": scalar_checks,
        "trainer_blocks_identical": trainer_pass,
        "model_net_blocks_identical": net_pass,
        "pairwise_diffs": pairwise,
        "illegal_diffs": illegal_union,
        "permitted_diffs": permitted_union,
    }


def compare_date_to_date(
    new_names: dict[str, str],
) -> dict[str, Any]:
    per_arm: dict[str, Any] = {}
    all_pass = True
    for arm, names in new_names.items():
        new_flat = _flatten(load_experiment_yaml(names["new"]))
        sealed_flat = _flatten(load_experiment_yaml(names["sealed"]))
        result = diff_paths(new_flat, sealed_flat, permitted=DATE_TO_DATE_PERMITTED_PATHS)
        arm_pass = not result["illegal"]
        per_arm[arm] = {
            "pass": arm_pass,
            "new_config": names["new"],
            "sealed_config": names["sealed"],
            "illegal_diffs": result["illegal"],
            "permitted_diffs": result["permitted"],
        }
        all_pass = all_pass and arm_pass
    return {
        "pass": all_pass,
        "permitted_difference_paths": sorted(DATE_TO_DATE_PERMITTED_PATHS),
        "per_arm": per_arm,
    }


def check_zero_carrier_semantics(cfgs: dict[str, DictConfig]) -> dict[str, Any]:
    observed = {
        name: {
            "pilot.arm": str(cfgs[name].pilot.arm),
            "pilot.zero_carrier": bool(cfgs[name].pilot.zero_carrier),
        }
        for name in cfgs
    }
    context_ok = (
        observed["context_full"]["pilot.arm"] == "full"
        and observed["context_full"]["pilot.zero_carrier"] is False
    )
    hse5_ok = (
        observed["hse5_full"]["pilot.arm"] == "full"
        and observed["hse5_full"]["pilot.zero_carrier"] is False
    )
    zero_ok = (
        observed["zero5"]["pilot.arm"] == "zero"
        and observed["zero5"]["pilot.zero_carrier"] is True
    )
    return {
        "pass": context_ok and hse5_ok and zero_ok,
        "observed": observed,
        "required": {
            "context_full": {"arm": "full", "zero_carrier": False},
            "hse5_full": {"arm": "full", "zero_carrier": False},
            "zero5": {"arm": "zero", "zero_carrier": True},
        },
    }


def build_parity_report(*, outer_date: str = "19250108") -> dict[str, Any]:
    composed = {
        arm: compose_experiment(spec["new"]) for arm, spec in STAGE_A_ARMS.items()
    }
    experiment_names = {arm: spec["new"] for arm, spec in STAGE_A_ARMS.items()}
    arm_to_arm = compare_arm_to_arm(composed, experiment_names=experiment_names)
    date_to_date = compare_date_to_date(STAGE_A_ARMS)
    zero_carrier = check_zero_carrier_semantics(composed)

    config_shas = {
        arm: {
            "new": {
                "path": str(_experiment_yaml_path(spec["new"])),
                "sha256": sha256_file(_experiment_yaml_path(spec["new"])),
            },
            "sealed": {
                "path": str(_experiment_yaml_path(spec["sealed"])),
                "sha256": sha256_file(_experiment_yaml_path(spec["sealed"])),
            },
        }
        for arm, spec in STAGE_A_ARMS.items()
    }

    all_pass = arm_to_arm["pass"] and date_to_date["pass"] and zero_carrier["pass"]
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_CTXV2_STAGE_A_CONFIG_PARITY" if all_pass else "STOP_CTXV2_CONFIG_PARITY_FAILED",
        "outer_date": outer_date,
        "compose_method": "hydra_compose_train_yaml_plus_experiment_override",
        "clauses": {
            "arm_to_arm_parity": arm_to_arm,
            "date_to_date_parity": date_to_date,
            "zero_carrier_semantics": zero_carrier,
        },
        "config_file_shas": config_shas,
        "scope": {
            "cuda_used": False,
            "training_launched": False,
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"CTXV2 Stage A config parity: {report['status']}")
    for clause, payload in report["clauses"].items():
        mark = "PASS" if payload["pass"] else "FAIL"
        print(f"  [{mark}] {clause}")
        if clause == "arm_to_arm_parity" and not payload["pass"]:
            if payload["illegal_diffs"]:
                print("    illegal diffs:")
                for path, entry in sorted(payload["illegal_diffs"].items()):
                    print(f"      {path}: {entry}")
        if clause == "date_to_date_parity" and not payload["pass"]:
            for arm, arm_payload in payload["per_arm"].items():
                if arm_payload["illegal_diffs"]:
                    print(f"    {arm} illegal diffs:")
                    for path, entry in sorted(arm_payload["illegal_diffs"].items()):
                        print(f"      {path}: {entry}")
    print("  permitted arm-to-arm difference paths:")
    for path in report["clauses"]["arm_to_arm_parity"]["permitted_difference_paths"]:
        print(f"    - {path}")
    print("  permitted date-to-date difference paths:")
    for path in report["clauses"]["date_to_date_parity"]["permitted_difference_paths"]:
        print(f"    - {path}")
    arm_permitted = report["clauses"]["arm_to_arm_parity"]["permitted_diffs"]
    if arm_permitted:
        print("  arm-to-arm permitted diffs (sample):")
        for path in sorted(arm_permitted):
            print(f"    {path}: {arm_permitted[path]}")
    for arm, arm_payload in report["clauses"]["date_to_date_parity"]["per_arm"].items():
        if arm_payload["permitted_diffs"]:
            print(f"  date-to-date permitted diffs ({arm}):")
            for path in sorted(arm_payload["permitted_diffs"]):
                print(f"    {path}: {arm_payload['permitted_diffs'][path]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--outer-date", default="19250108")
    args = parser.parse_args()

    report = build_parity_report(outer_date=args.outer_date)
    _print_summary(report)

    out, digest = write_immutable_json(args.output, report)
    print(json.dumps({"receipt": str(out), "sha256": digest}, indent=2))
    return 0 if report["status"] == "PASS_CTXV2_STAGE_A_CONFIG_PARITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
