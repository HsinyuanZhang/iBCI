#!/usr/bin/env python3
"""Fail-closed aggregate for the fresh T4 decoupled K/V five-arm screen."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sua_confidence_film_t4_budget import (  # noqa: E402
    _per_session_epoch_mean,
    summarize,
)


ARMS = (
    "coupled_t4",
    "kv_e_t4",
    "kv_e_ts4",
    "kv_e_only",
    "kv_x_only",
)
KEY_MODES = {
    "coupled_t4": None,
    "kv_e_t4": "e_t4",
    "kv_e_ts4": "e_ts4",
    "kv_e_only": "e_only",
    "kv_x_only": "x_only",
}
CONTRASTS = {
    "kv_e_t4_vs_coupled_t4": ("kv_e_t4", "coupled_t4"),
    "kv_e_t4_vs_kv_e_ts4": ("kv_e_t4", "kv_e_ts4"),
    "kv_e_t4_vs_kv_e_only": ("kv_e_t4", "kv_e_only"),
    "kv_e_t4_vs_kv_x_only": ("kv_e_t4", "kv_x_only"),
    "kv_e_only_vs_coupled_t4": ("kv_e_only", "coupled_t4"),
    "kv_e_only_vs_kv_x_only": ("kv_e_only", "kv_x_only"),
}
EPOCHS = list(range(5, 13))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def validate_arm(
    path: Path, arm: str, seed: int
) -> tuple[list[str], np.ndarray, dict]:
    payload = _load(path)
    protocol = payload.get("protocol") or {}
    _require(f"{path}: calibration_n", protocol.get("calibration_n"), 30)
    _require(f"{path}: pool_size", protocol.get("pool_size"), 50)
    _require(f"{path}: epoch window", protocol.get("epoch_window"), EPOCHS)
    metadata_path = Path(payload.get("run_metadata_path", ""))
    if not metadata_path.is_file():
        raise ValueError(f"{path}: run metadata is missing: {metadata_path}")
    metadata = _load(metadata_path)
    _require(f"{path}: variant", metadata.get("variant"), "B3S")
    _require(f"{path}: seed", metadata.get("seed"), seed)
    _require(
        f"{path}: side group",
        (metadata.get("side_features") or {}).get("group"),
        "t4",
    )
    _require(
        f"{path}: T4 feature version",
        (metadata.get("side_features") or {}).get("feature_version"),
        1,
    )
    _require(
        f"{path}: T4 pool",
        (metadata.get("side_features") or {}).get("pool_size"),
        50,
    )
    training = metadata.get("training") or {}
    _require(f"{path}: activity budget", training.get("calibration_n_trials"), 30)
    _require(f"{path}: epochs", training.get("max_epochs"), 12)
    _require(f"{path}: no early stopping", training.get("no_early_stopping"), True)
    _require(f"{path}: status", metadata.get("status"), "completed")
    _require(f"{path}: no formal test", metadata.get("held_out_test_evaluated"), False)
    _require(f"{path}: no warmstart", metadata.get("encoder_warmstart_path"), None)

    decoder = metadata.get("decoder_architecture") or {}
    expected_mode = "coupled" if arm == "coupled_t4" else "decoupled"
    _require(f"{path}: decoder mode", decoder.get("mode"), expected_mode)
    _require(f"{path}: key mode", decoder.get("key_mode"), KEY_MODES[arm])
    _require(f"{path}: fixed slots", decoder.get("fixed_slot_count"), 0)
    cost_comparison = decoder.get(
        "decoder_cost_comparison_receipt_reference_n64"
    ) or {}
    _require(
        f"{path}: coupled cost reference mode",
        (cost_comparison.get("coupled") or {}).get("persistent_state_width"),
        50,
    )
    if arm != "coupled_t4":
        _require(f"{path}: key width", decoder.get("key_width"), 32)
        _require(f"{path}: value width", decoder.get("value_width"), 32)
        _require(f"{path}: layers", decoder.get("num_layers"), 1)
        _require(f"{path}: heads", decoder.get("num_heads"), 2)
        _require(
            f"{path}: encoder input",
            decoder.get("encoder_side_input"),
            "aligned_real_t4",
        )
        cost = decoder.get("online_cost_receipt_reference_n64") or {}
        _require(
            f"{path}: no N squared",
            (cost.get("online_macs_per_frame") or {}).get(
                "no_unit_quadratic_term"
            ),
            True,
        )
        decoupled_cost = cost_comparison.get("decoupled") or {}
        if float(
            decoupled_cost.get(
                "online_mac_reduction_fraction_vs_coupled", -1.0
            )
        ) < 0.25:
            raise ValueError(f"{path}: decoupled online MAC reduction is below 25%")
        _require(
            f"{path}: cost state nonincreasing",
            decoupled_cost.get(
                "persistent_state_nonincreasing_vs_coupled"
            ),
            True,
        )
        cache = decoder.get("persistent_cache_receipt_reference_n64") or {}
        if arm == "kv_x_only":
            _require(f"{path}: x-only cache bytes", cache.get("cache_bytes"), 0)
        else:
            _require(
                f"{path}: state nonincreasing",
                cache.get("state_nonincreasing_vs_identity"),
                True,
            )
            _require(f"{path}: n64 cache bytes", cache.get("cache_bytes"), 8192)
    sessions, values = _per_session_epoch_mean(payload, path)
    return sessions, values, metadata


def aggregate(result_dir: Path, seeds: tuple[int, ...]) -> dict:
    matrices: dict[str, np.ndarray] = {}
    artifacts: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    session_names: list[str] | None = None
    common_receipts: dict[str, dict[str, str]] = {}

    for arm in ARMS:
        rows = []
        for seed in seeds:
            path = result_dir / f"{arm}_m50_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            sessions, values, metadata = validate_arm(path, arm, seed)
            if session_names is None:
                session_names = sessions
            elif sessions != session_names:
                raise ValueError(f"{path}: validation session matrix differs")
            teacher_sha = metadata.get("teacher_sha256")
            normalization_sha = (metadata.get("side_features") or {}).get(
                "normalization_sha256"
            )
            receipt = common_receipts.setdefault(
                str(seed),
                {
                    "teacher_sha256": teacher_sha,
                    "normalization_sha256": normalization_sha,
                },
            )
            _require(f"{path}: teacher SHA", teacher_sha, receipt["teacher_sha256"])
            _require(
                f"{path}: normalization SHA",
                normalization_sha,
                receipt["normalization_sha256"],
            )
            rows.append(values)
            artifacts[arm][str(seed)] = str(path.resolve())
        matrices[arm] = np.asarray(rows, dtype=np.float64)

    assert session_names is not None
    # New-module initialization must be bitwise identical across the four
    # decoupled arms of each paired seed.
    for seed in seeds:
        hashes = []
        shared_hashes = []
        for arm in ARMS:
            result = _load(result_dir / f"{arm}_m50_s{seed}.json")
            metadata = _load(Path(result["run_metadata_path"]))
            decoder = metadata["decoder_architecture"]
            shared_hashes.append(decoder.get("shared_decoder_base_sha256"))
            if arm != "coupled_t4":
                hashes.append(decoder.get("new_projection_init_sha256"))
        if len(set(hashes)) != 1 or not hashes[0]:
            raise ValueError(f"seed {seed}: decoupled projection initialization drift")
        if len(set(shared_hashes)) != 1 or not shared_hashes[0]:
            raise ValueError(f"seed {seed}: shared decoder-base initialization drift")

    contrasts = {
        name: summarize(
            matrices[treatment],
            matrices[control],
            seeds=seeds,
            sessions=session_names,
        )
        for name, (treatment, control) in CONTRASTS.items()
    }
    e_t4_stage0 = all(
        contrasts[name]["passes_stage0_descriptive_gates"]
        for name in (
            "kv_e_t4_vs_coupled_t4",
            "kv_e_t4_vs_kv_x_only",
        )
    )
    e_only_stage0 = all(
        contrasts[name]["passes_stage0_descriptive_gates"]
        for name in (
            "kv_e_only_vs_coupled_t4",
            "kv_e_only_vs_kv_x_only",
        )
    )
    e_t4_formal = all(
        contrasts[name]["passes_formal_effectiveness_gates"]
        for name in (
            "kv_e_t4_vs_coupled_t4",
            "kv_e_t4_vs_kv_x_only",
        )
    )
    e_only_formal = all(
        contrasts[name]["passes_formal_effectiveness_gates"]
        for name in (
            "kv_e_only_vs_coupled_t4",
            "kv_e_only_vs_kv_x_only",
        )
    )
    def content_stage0(name: str) -> bool:
        summary = contrasts[name]
        gates = summary["descriptive_stage0_gates"]
        return all(
            gates[key]
            for key in (
                "all_observed_seed_means_positive",
                "all_six_session_means_positive",
                "session_paired_exact_wilcoxon_two_sided_le_0p05",
            )
        )

    def content_formal(name: str) -> bool:
        summary = contrasts[name]
        gates = summary["formal_effectiveness_gates"]
        return all(
            gates[key]
            for key in (
                "all_observed_seed_means_positive",
                "all_six_session_means_positive",
                "session_paired_exact_wilcoxon_two_sided_le_0p05",
                "at_least_three_predeclared_seeds",
                "hierarchical_bootstrap_95ci_lower_positive",
            )
        )

    def lower_two_seed_se(treatment: str, control: str) -> float | None:
        delta = matrices[treatment] - matrices[control]
        seed_means = delta.mean(axis=1)
        if seed_means.size < 2:
            return None
        standard_error = float(
            np.std(seed_means, ddof=1) / np.sqrt(seed_means.size)
        )
        return float(delta.mean() - 2.0 * standard_error)

    e_t4_lower = lower_two_seed_se("kv_e_t4", "coupled_t4")
    e_only_lower = lower_two_seed_se("kv_e_only", "coupled_t4")
    e_t4_deployment_stage0 = (
        contrasts["kv_e_t4_vs_coupled_t4"]["mean_paired_delta_r2"] >= -0.03
        and content_stage0("kv_e_t4_vs_kv_e_ts4")
    )
    e_only_deployment_stage0 = (
        contrasts["kv_e_only_vs_coupled_t4"]["mean_paired_delta_r2"] >= -0.03
        and content_stage0("kv_e_only_vs_kv_x_only")
    )
    e_t4_deployment_formal = (
        e_t4_lower is not None
        and e_t4_lower >= -0.03
        and content_formal("kv_e_t4_vs_kv_e_ts4")
    )
    e_only_deployment_formal = (
        e_only_lower is not None
        and e_only_lower >= -0.03
        and content_formal("kv_e_only_vs_kv_x_only")
    )
    stage0_any = (
        e_t4_stage0
        or e_only_stage0
        or e_t4_deployment_stage0
        or e_only_deployment_stage0
    )
    formal_any = (
        e_t4_formal
        or e_only_formal
        or e_t4_deployment_formal
        or e_only_deployment_formal
    )
    selected = (
        "kv_e_only"
        if e_only_formal or e_only_deployment_formal
        else "kv_e_t4"
        if e_t4_formal or e_t4_deployment_formal
        else None
    )
    return {
        "schema_version": 1,
        "purpose": "fresh_t4_decoupled_key_value_screen",
        "generated_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "M_activity": 30,
            "M_T4": 50,
            "common_evaluation_start": 50,
            "epochs": 12,
            "scored_epoch_window": EPOCHS,
            "seeds": list(seeds),
            "sessions": session_names,
            "formal_test_evaluated": False,
        },
        "artifacts": artifacts,
        "common_receipts_by_seed": common_receipts,
        "arm_mean_r2": {
            arm: float(matrix.mean()) for arm, matrix in matrices.items()
        },
        "contrasts": contrasts,
        "candidate_rules": {
            "kv_e_t4_requires": [
                "kv_e_t4_vs_coupled_t4",
                "kv_e_t4_vs_kv_x_only",
            ],
            "kv_e_only_requires": [
                "kv_e_only_vs_coupled_t4",
                "kv_e_only_vs_kv_x_only",
            ],
            "simplicity_preference_if_both_pass": "kv_e_only",
            "direct_t4_diagnostics": [
                "kv_e_t4_vs_kv_e_ts4",
                "kv_e_t4_vs_kv_e_only",
            ],
            "deployment_noninferiority": (
                "paired seed-mean lower 2SE >= -0.03, >=25% online decoder "
                "MAC reduction, nonincreasing persistent state, and strict "
                "positive content contrast"
            ),
        },
        "deployment_effectiveness": {
            "kv_e_t4": {
                "paired_lower_2se_vs_coupled": e_t4_lower,
                "stage0_pass": e_t4_deployment_stage0,
                "formal_pass": e_t4_deployment_formal,
                "content_contrast": "kv_e_t4_vs_kv_e_ts4",
                "online_decoder_mac_reduction_at_least_25pct": True,
                "persistent_state_nonincreasing": True,
            },
            "kv_e_only": {
                "paired_lower_2se_vs_coupled": e_only_lower,
                "stage0_pass": e_only_deployment_stage0,
                "formal_pass": e_only_deployment_formal,
                "content_contrast": "kv_e_only_vs_kv_x_only",
                "online_decoder_mac_reduction_at_least_25pct": True,
                "persistent_state_nonincreasing": True,
            },
        },
        "stage0_descriptive_mechanism_pass": stage0_any,
        "stage0_candidate_pass": {
            "kv_e_t4": e_t4_stage0 or e_t4_deployment_stage0,
            "kv_e_only": e_only_stage0 or e_only_deployment_stage0,
        },
        "formal_effectiveness_eligible": len(seeds) >= 3,
        "formal_effectiveness_pass": formal_any,
        "formal_candidate_pass": {
            "kv_e_t4": e_t4_formal or e_t4_deployment_formal,
            "kv_e_only": e_only_formal or e_only_deployment_formal,
        },
        "selected_effective_candidate": selected,
    }


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    if not set(seeds).issubset({42, 43, 44}):
        raise argparse.ArgumentTypeError("seeds must be a subset of 42,43,44")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=(42, 43, 44))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.result_dir.expanduser().resolve(), args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "arm_mean_r2": result["arm_mean_r2"],
        "stage0_descriptive_mechanism_pass": result[
            "stage0_descriptive_mechanism_pass"
        ],
        "formal_effectiveness_pass": result["formal_effectiveness_pass"],
        "selected_effective_candidate": result["selected_effective_candidate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
