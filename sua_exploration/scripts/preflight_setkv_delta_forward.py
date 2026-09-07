#!/usr/bin/env python3
"""CPU-only preflight for the sealed-A2 SetKV-delta forward diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for value in (REPO_ROOT, SUA_ROOT, SUA_ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mc_maze import a2_matched_subject_shift_v2_core as a2  # noqa: E402
from mc_maze import setkv_delta_forward_core as core  # noqa: E402
from mc_maze.setkv_delta import decode_setkv_delta  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise core.SetKVForwardContractError(message)


def _require_canonical_output(path: Path) -> None:
    expected = core.OFFICIAL_PREFLIGHT.resolve()
    require(path.resolve() == expected, f"official SetKV preflight output must use canonical path: {expected}")


def _verify_a2_authority() -> tuple[dict, str, dict, str, dict[str, dict]]:
    official, official_sha = core.load_immutable(core.A2_PREFLIGHT, "sealed A2 official preflight")
    terminal, terminal_sha = core.load_immutable(core.A2_TERMINAL, "sealed A2 terminal aggregate")
    require(official_sha == core.EXPECTED_A2_PREFLIGHT_SHA256, "A2 preflight SHA drift")
    require(terminal_sha == core.EXPECTED_A2_TERMINAL_SHA256, "A2 terminal SHA drift")
    # A2's own verifier proves the current production loader still exactly
    # matches the implementation snapshot used by the sealed checkpoints.
    a2.verify_implementation_bindings(official.get("implementation_bindings"))
    require(terminal.get("official_preflight_sha256") == official_sha, "A2 terminal/preflight linkage drift")
    require(terminal.get("implementation_bindings") == official.get("implementation_bindings"),
            "A2 terminal implementation binding drift")
    pairing = terminal.get("receipt_pairing_validation") or {}
    require(pairing.get("passed") is True and pairing.get("all_required_receipts_present") is True,
            "A2 terminal receipt-pairing authority drift")
    receipts: dict[str, dict] = {}
    for domain in core.DOMAINS:
        for source_arm in ("source_t4", "source_z4"):
            value, _digest = core.load_immutable(
                core.a2_baseline_receipt_path(source_arm, domain), "sealed A2 domain receipt"
            )
            require(value.get("seed") == core.SEED and value.get("source_arm") == source_arm,
                    "A2 domain source-cell drift")
            require(value.get("domain") == domain, "A2 domain drift")
            require(value.get("official_preflight_sha256") == official_sha,
                    "A2 domain receipt/preflight linkage drift")
            require(value.get("implementation_bindings") == official.get("implementation_bindings"),
                    "A2 domain receipt implementation binding drift")
            require(value.get("query_policy") == official.get("query_policy"),
                    "A2 domain receipt query-policy drift")
            require(value.get("formal_subc_test_nwb_opened") is False, "formal A2 data opened")
            require(value.get("target_session_carrier_fit_performed") is True, "A2 carrier-fit ledger drift")
            require(value.get("backward_gradients") is False and value.get("decoder_weight_updates") is False,
                    "A2 target update ledger drift")
            receipts[f"{domain}_{source_arm}"] = value
    for source_arm in ("source_t4", "source_z4"):
        left = receipts[f"within_subject_{source_arm}"]
        right = receipts[f"external_subject_M_{source_arm}"]
        require(left["source_checkpoint_sha256_bundle"] == right["source_checkpoint_sha256_bundle"],
                "A2 within/external checkpoint bundle drift")
        run_dir = Path(str(left["source_run"]["source_run_dir"]))
        checkpoints = a2.source_epoch_checkpoint_paths(run_dir)
        for epoch in a2.EPOCH_WINDOW:
            require(a2.sha256_file(checkpoints[epoch]) == left["source_checkpoint_sha256_bundle"][str(epoch)],
                    f"A2 {source_arm} epoch{epoch} checkpoint byte drift")
    normalizer_authorities = [value["normalizer_authority"] for value in receipts.values()]
    require(all(value == normalizer_authorities[0] for value in normalizer_authorities[1:]),
            "A2 seed42 normalizer authority differs across matched receipt lattice")
    return official, official_sha, terminal, terminal_sha, receipts


def _production_cpu_smoke(receipts: dict[str, dict]) -> dict[str, object]:
    import torch
    from scripts.select_gradient_free_protocol_dandi688 import load_frozen_model

    torch.manual_seed(20260814)
    evidence: dict[str, object] = {}
    for source_arm in ("source_t4", "source_z4"):
        receipt = receipts[f"within_subject_{source_arm}"]
        run_dir = Path(str(receipt["source_run"]["source_run_dir"]))
        checkpoint = a2.source_epoch_checkpoint_paths(run_dir)[5]
        model = load_frozen_model(
            checkpoint, a2.TEACHER_PATH, "B3S", torch.device("cpu"), identity_mode="calibrated"
        )
        decoder = model.student.decoder
        require((decoder.model_dim, decoder.num_covariates, decoder.num_layers) == (512, 2, 1),
                "SetKV analytic cost topology drift")
        before = tuple((name, parameter.numel()) for name, parameter in model.named_parameters())
        neural = torch.randn(1, 50, 7)
        calib = torch.randn(1, 30, 100, 7)
        side = torch.randn(1, 7, 4) if source_arm == "source_t4" else torch.zeros(1, 7, 4)
        with torch.no_grad():
            expected, expected_identity = model.student(
                neural, calib_trials=calib, side_features=side
            )
            baseline = decode_setkv_delta(
                model.student, neural, calib, side, decode_mode="baseline"
            )
            carrier = decode_setkv_delta(
                model.student,
                neural,
                calib,
                side,
                decode_mode="carrier",
                carrier_mode="aligned" if source_arm == "source_t4" else "zero",
            )
        after = tuple((name, parameter.numel()) for name, parameter in model.named_parameters())
        require(torch.equal(expected, baseline.prediction), "production baseline prediction parity failed")
        require(torch.equal(expected_identity, baseline.identity), "production identity parity failed")
        require(before == after, "SetKV registered parameters during production smoke")
        require(carrier.appended_tokens is not None and torch.isfinite(carrier.prediction).all(),
                "SetKV production smoke is non-finite")
        if source_arm == "source_z4":
            require(torch.equal(carrier.appended_tokens, torch.zeros_like(carrier.appended_tokens)),
                    "Z4 carrier token is not bitwise zero")
        evidence[source_arm] = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": a2.sha256_file(checkpoint),
            "baseline_prediction_bitwise_equal": True,
            "baseline_identity_bitwise_equal": True,
            "parameter_registry_unchanged": True,
            "carrier_prediction_finite": True,
            "z4_carrier_token_bitwise_zero": source_arm == "source_z4",
            "decoder_cost_topology": {
                "model_dim": decoder.model_dim,
                "num_queries": decoder.num_covariates,
                "num_layers": decoder.num_layers,
                "num_heads": decoder.num_heads,
            },
        }
    return evidence


def build_payload() -> dict:
    official, official_sha, terminal, terminal_sha, receipts = _verify_a2_authority()
    bindings = core.current_implementation_bindings()
    smoke = _production_cpu_smoke(receipts)
    return {
        "schema_version": 1,
        "receipt_kind": "setkv_delta_forward_official_cpu_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": core.SCREEN_ID,
        "status": core.PREFLIGHT_STATUS,
        "contract_path": str(core.CONTRACT_PATH.resolve()),
        "contract_sha256": core.sha256_file(core.CONTRACT_PATH),
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": core.canonical_sha256(bindings),
        "a2_official_preflight_path": str(core.A2_PREFLIGHT.resolve()),
        "a2_official_preflight_sha256": official_sha,
        "a2_terminal_aggregate_path": str(core.A2_TERMINAL.resolve()),
        "a2_terminal_aggregate_sha256": terminal_sha,
        "a2_implementation_bindings_verified_live": True,
        "a2_terminal_preflight_link_verified": terminal.get("official_preflight_sha256") == official_sha,
        "a2_seed42_checkpoint_bundles": {
            source_arm: receipts[f"within_subject_{source_arm}"]["source_checkpoint_sha256_bundle_sha256"]
            for source_arm in ("source_t4", "source_z4")
        },
        "production_cpu_smoke": smoke,
        "interventions": {name: dict(value) for name, value in core.INTERVENTIONS.items()},
        "domains": list(core.DOMAINS),
        "epoch_window": list(a2.EPOCH_WINDOW),
        "formal_subc_test_nwb_opened": False,
        "source_nwb_opened": False,
        "target_nwb_opened": False,
        "cuda_used": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=core.OFFICIAL_PREFLIGHT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        bindings = core.current_implementation_bindings()
        print(json.dumps({
            "status": "DRY_RUN__NO_MODEL_NO_DATA_NO_GPU",
            "output": str(args.output),
            "implementation_bindings_sha256": core.canonical_sha256(bindings),
            "interventions": list(core.INTERVENTIONS),
            "domains": list(core.DOMAINS),
        }, indent=2, sort_keys=True))
        return 0
    _require_canonical_output(args.output)
    payload = build_payload()
    _body, _sidecar, digest = core.write_immutable(args.output, payload)
    print(json.dumps({"status": payload["status"], "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
