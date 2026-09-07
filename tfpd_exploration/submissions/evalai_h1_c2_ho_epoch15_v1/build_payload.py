#!/usr/bin/env python3
"""Build the held-out-selected C2 epoch-15 H1 payload.

The public M3 activity/carrier rows are inherited byte-for-byte from the
already-audited 27-session payload used by submission 581900.  Only the H1
network state is replaced by the independently sealed C2/epoch-15 state.  A
zero-output FiLM shim preserves the existing deployment entrypoint while
making its identity operator bitwise equal to native early pooling.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
SPINT_MAIN = ROOT / "SPINT-main"
if str(SPINT_MAIN) not in sys.path:
    sys.path.insert(0, str(SPINT_MAIN))

from src.h1_m4_cce_contract import state_hash  # noqa: E402
from third_party.falcon_challenge.h1_epfilm_spint_decoder import (  # noqa: E402
    H1EPFiLMSpintDecoder,
)
from falcon_challenge.config import FalconConfig, FalconTask  # noqa: E402


SOURCE = ROOT / "tfpd_exploration/submissions/evalai_h1_epfilm_no_readout_v1/artifacts/decoder.pt"
C2 = Path(
    "/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/"
    "h1_series_20260830/artifacts/c2_references/c2_epoch_015.ckpt"
)
SELECTION = Path(
    "/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/"
    "h1_series_20260830/results/h1_cal_aug_m3_aware_dual_selection_v2_eval_a1/"
    "selection/c2_ho.json"
)
OUTPUT = Path(__file__).resolve().parent / "artifacts/decoder.pt"

SOURCE_SHA256 = "523d3d2e55a8fd4620a3f0a94dea6a99ae6ff53cefe807c3ff809a30b1d2479d"
C2_SHA256 = "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("source 581900 payload drift")
    if sha256(C2) != C2_SHA256:
        raise RuntimeError("C2 epoch-15 checkpoint drift")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    selected = selection["selected"]
    if int(selected["epoch_zero_based"]) != 15 or selected["checkpoint_sha256"] != C2_SHA256:
        raise RuntimeError("held-out epoch selection authority drift")

    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    checkpoint = torch.load(C2, map_location="cpu", weights_only=False)
    c2_state = checkpoint["state_dict"]
    if set(c2_state) != set(payload["state_dict"]):
        raise RuntimeError("C2 and deployment model state topology differ")
    payload["state_dict"] = {
        key: value.detach().cpu().clone() for key, value in c2_state.items()
    }
    payload["model_state_sha256"] = state_hash(payload["state_dict"])
    payload["checkpoint_sha256"] = C2_SHA256

    film_state = payload["film"]["state_dict"]
    film_state["2.weight"] = torch.zeros_like(film_state["2.weight"])
    film_state["2.bias"] = torch.zeros_like(film_state["2.bias"])
    payload["film_state_sha256"] = state_hash(film_state)
    payload["film_checkpoint_sha256"] = "C2_NATIVE_ZERO_FILM_SHIM_V1"
    payload["film_checkpoint_film_state_sha256"] = payload["film_state_sha256"]
    payload["readout_selection_sha256"] = "IDENTITY_READOUT_NO_FIT_V1"
    payload["derivation"] = {
        "candidate": "H1_C2_HO_SELECTED_EPOCH15_NATIVE_VIA_ZERO_FILM_SHIM",
        "source_public_m3_payload_sha256": SOURCE_SHA256,
        "checkpoint_sha256": C2_SHA256,
        "epoch_zero_based": 15,
        "selection_surface": "visible held-out H1 calibration/development recordings",
        "selection_metric": selection["selection_metric"],
        "selection_mean_r2": selected["val_ho_m3_grouped/r2_mean"],
        "selection_authority_sha256": sha256(SELECTION),
        "hidden_test_opened": False,
        "optimizer_steps": 0,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUTPUT)
    decoder = H1EPFiLMSpintDecoder(
        FalconConfig(task=FalconTask.h1), OUTPUT, batch_size=1, device="cpu"
    )
    first_key = sorted(payload["sessions"])[0]
    first_session = str(payload["sessions"][first_key]["session"])
    decoder.reset([Path(f"sub-HumanPitt-held-out-calib_{first_session}")])
    with torch.inference_mode():
        encoded = decoder.model.carrier_pre_pool(
            decoder.local_activity.permute(0, 1, 3, 2)
        )
        effective = decoder.local_carrier.to(encoded)
        native = decoder.model.carrier_post_pool(
            torch.cat((encoded.mean(dim=1), effective), dim=-1)
        )
        shim = decoder._film_identity()
    if not torch.equal(native, shim):
        raise RuntimeError("zero-FiLM shim is not bitwise native identity")
    prediction = decoder.predict(torch.zeros((1, 176), dtype=torch.float32).numpy())
    if not bool(torch.isfinite(torch.from_numpy(prediction)).all()):
        raise RuntimeError("nonfinite H1 smoke prediction")

    receipt = {
        "schema": "h1_c2_ho_epoch15_evalai_payload_v1",
        "payload_sha256": sha256(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "checkpoint_sha256": C2_SHA256,
        "model_state_sha256": payload["model_state_sha256"],
        "film_state_sha256": payload["film_state_sha256"],
        "epoch_zero_based": 15,
        "selection_surface": payload["derivation"]["selection_surface"],
        "selection_mean_r2": payload["derivation"]["selection_mean_r2"],
        "selection_authority_sha256": payload["derivation"]["selection_authority_sha256"],
        "session_count": len(payload["sessions"]),
        "native_identity_bitwise_equal": True,
        "hidden_test_opened": False,
    }
    (OUTPUT.parent / "payload.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
