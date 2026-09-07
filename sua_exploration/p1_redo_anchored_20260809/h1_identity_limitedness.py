#!/usr/bin/env python3
"""H1 anchor gate + identity-limitedness diagnostic (P1-REDO-ANCHORED, 2026-08-09).

Read-only, CPU-only, forward-pass-only diagnostic. Writes nothing outside this
directory. Reuses the sealed epoch-49 H-C checkpoint and the exact strict
fold-0 query windows already opened by
``SPINT-main/scripts/h1_carrierid_same_checkpoint_dose_response.py`` and
``SPINT-main/scripts/h1_carrierid_evaluate.py``. This script does not call any
function in those modules that writes a file; it only calls the read/forward
helpers (`_require_preflight`, `_prepare_opened_development`,
`_gate_and_full_metric`, `_evaluate`).

Anchor: sealed pooled_r2 for the H-C "full" arm on strict fold-0 target
windows, recorded in
``pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/H1_CARRIERID_H32_FOLD0_TERMINAL_GATE_FLOAT64_R2.json``
(``metrics.h_c_interventions.full.pooled_r2`` == 0.52551078006707), which is
the same anchor already reproduced by the dose-response v2 runtime P0 gate.

Identity-limitedness: the same checkpoint scored on the dataset-level "zero"
carrier intervention, whose source line is
``SPINT-main/src/data/h1_m4_eb_pilot.py:1040``
(``carriers["zero"] = np.zeros_like(full)``), reaffirmed for the normalized-V2
wrapper at ``SPINT-main/src/data/h1_m4_eb_normalized_v2.py`` (the
``np.zeros_like`` branch in ``H1M4EBNormalizedV2StrictTargetDataset.__init__``).
That zero carrier tensor is what ``H1CarrierIdSpint.forward`` combines into
the neural window at
``SPINT-main/src/models/components/h1_carrierid_spint.py:147``
(``src = src + identity``), where ``identity`` is produced at line
136-139 from the (now identically zero) ``carrier`` argument.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path("/home/xinyuan/Work_host/SPINT/SPINT-main").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.h1_carrierid_same_checkpoint_dose_response import (  # noqa: E402
    _gate_and_full_metric,
    _prepare_opened_development,
    _require_preflight,
    H_C_GATE,
)
from scripts.h1_carrierid_evaluate import _evaluate  # noqa: E402
from src.h1_m4_eb_normalized_v2_contract import sha256_file  # noqa: E402

ABSOLUTE_TOLERANCE = 1.0e-6  # frozen before looking at any runtime number; same device (cpu) as the dose-response P0 gate


def run() -> dict[str, Any]:
    torch.set_num_threads(8)

    # --- Section 2: anchor gate -------------------------------------------------
    _require_preflight()  # read-only sanity: immutable dose-response preflight is intact
    gate, sealed_full = _gate_and_full_metric()
    sealed_zero = gate["metrics"]["h_c_interventions"]["zero"]

    preflight, cfg, source, model, base, device, observed_files = _prepare_opened_development("cpu")
    state_before_full = model.state_dict()
    before_hash_manual = torch.cat([p.detach().flatten() for p in state_before_full.values()]).sum().item()

    runtime_full = _evaluate(model, base, device, "H-C/anchor-full")
    anchor_delta = float(runtime_full["pooled_r2"]) - float(sealed_full["pooled_r2"])
    anchor_pass = abs(anchor_delta) <= ABSOLUTE_TOLERANCE

    anchor_gate = {
        "task": "H1",
        "sealed_receipt_path": str(H_C_GATE),
        "sealed_receipt_sha256": sha256_file(H_C_GATE),
        "sealed_key": "metrics.h_c_interventions.full.pooled_r2",
        "sealed_value": float(sealed_full["pooled_r2"]),
        "runtime_value": float(runtime_full["pooled_r2"]),
        "absolute_delta": anchor_delta,
        "frozen_absolute_tolerance": ABSOLUTE_TOLERANCE,
        "tolerance_justification": "same device (cpu), same code path, same query windows as the sealed dose-response v2 P0 reproduction gate, which already reproduced this exact sealed value to 1.3e-08",
        "pass": anchor_pass,
        "state_sha256_before": runtime_full["state_sha256_before"],
        "state_sha256_after": runtime_full["state_sha256_after"],
        "state_immutable": runtime_full["state_immutable"],
        "windows_scored": int(runtime_full["samples"]),
        "windows_available_note": "all strict fold-0 post-support windows in the dataset; no subsampling",
    }

    result: dict[str, Any] = {
        "schema": "p1_redo_anchored_h1_identity_limitedness_v1",
        "created": "2026-08-09",
        "anchor_gate": anchor_gate,
    }

    if not anchor_pass:
        result["status"] = "ANCHOR_NOT_REPRODUCED"
        return result

    # --- Section 3: the measurement ---------------------------------------------
    zero_dataset = base.with_intervention("zero")
    runtime_zero = _evaluate(model, zero_dataset, device, "H-C/zeroed-identity-diagnostic")

    # cross-check against the sealed same-checkpoint "zero" intervention, which
    # is not itself the anchor but is an independent sealed number computed by
    # the exact same procedure (full checkpoint, dataset-level zero carrier).
    zero_delta = float(runtime_zero["pooled_r2"]) - float(sealed_zero["pooled_r2"])

    state_after_full = model.state_dict()
    after_hash_manual = torch.cat([p.detach().flatten() for p in state_after_full.values()]).sum().item()

    per_session = {}
    for name in runtime_full["per_session"]:
        full_r2 = float(runtime_full["per_session"][name]["r2"])
        zero_r2 = float(runtime_zero["per_session"][name]["r2"])
        per_session[name] = {
            "full_r2": full_r2,
            "zero_r2": zero_r2,
            "identity_limitedness": full_r2 - zero_r2,
        }

    result.update(
        {
            "status": "MEASURED",
            "zero_cross_check": {
                "sealed_receipt_path": str(H_C_GATE),
                "sealed_key": "metrics.h_c_interventions.zero.pooled_r2",
                "sealed_value": float(sealed_zero["pooled_r2"]),
                "runtime_value": float(runtime_zero["pooled_r2"]),
                "absolute_delta": zero_delta,
                "note": "not a gated anchor; independent sealed confirmation of the same same-checkpoint zero-carrier procedure this script runs fresh",
            },
            "zeroing_source_lines": {
                "data_zero_carrier_construction": "SPINT-main/src/data/h1_m4_eb_pilot.py:1040 (carriers[\"zero\"] = np.zeros_like(full))",
                "normalized_v2_zero_reaffirmation": "SPINT-main/src/data/h1_m4_eb_normalized_v2.py (np.zeros_like branch in H1M4EBNormalizedV2StrictTargetDataset.__init__, ~line 180)",
                "entry_into_neural_window": "SPINT-main/src/models/components/h1_carrierid_spint.py:147 (src = src + identity), identity computed lines 136-139 from the now-zero carrier",
            },
            "full": {
                "pooled_r2": float(runtime_full["pooled_r2"]),
                "samples": int(runtime_full["samples"]),
                "per_session": {k: v["r2"] for k, v in runtime_full["per_session"].items()},
            },
            "zeroed": {
                "pooled_r2": float(runtime_zero["pooled_r2"]),
                "samples": int(runtime_zero["samples"]),
                "per_session": {k: v["r2"] for k, v in runtime_zero["per_session"].items()},
            },
            "identity_limitedness_pooled": float(runtime_full["pooled_r2"]) - float(runtime_zero["pooled_r2"]),
            "identity_limitedness_per_session": per_session,
            "known_carrier_gain_five_date": 0.056287,
            "model_state_hash_manual_checksum_before": before_hash_manual,
            "model_state_hash_manual_checksum_after": after_hash_manual,
            "model_state_hash_manual_checksum_identical": before_hash_manual == after_hash_manual,
            "state_immutable_full": runtime_full["state_immutable"],
            "state_immutable_zeroed": runtime_zero["state_immutable"],
            "endpoint": {
                "description": "strict fold-0 held-in-calib target windows, sessions ses-19250101T111740 and ses-19250101T112404",
                "opened_by_receipt": str(H_C_GATE),
            },
            "claim_boundary": "same-checkpoint forward-only diagnostic, non-routing",
        }
    )
    return result


def main() -> None:
    started = time.monotonic()
    out = run()
    out["elapsed_seconds"] = time.monotonic() - started
    out_path = Path(__file__).resolve().parent / "h1_identity_limitedness_receipt.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "receipt": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
