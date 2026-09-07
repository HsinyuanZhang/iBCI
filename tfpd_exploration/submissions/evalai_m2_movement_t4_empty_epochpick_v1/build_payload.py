#!/usr/bin/env python3
"""Export the visible-held-out-selected MOVE-T4/EMPTY adapter for EvalAI."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SELECTION_ROOT = ROOT / "tfpd_exploration/results/m2_movement_t4_empty_epoch_pick_v1"
COMPAT_ROOT = HERE / "artifacts/compat_result"
OUTPUT = HERE / "artifacts/t4_m2_seed44_epoch08_movement_t4_empty_identity.pkl"
EXPECTED_LOCAL = 0.36044922649589894
EXPECTED_HEAD = "13551d3fc33d1cc296670c577c11519c04abdb42fa081f7d061191bb5610e6c5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    selection = json.loads((SELECTION_ROOT / "selection.json").read_text(encoding="utf-8"))
    selected = selection["selected"]
    if (
        selected["seed"] != 44
        or selected["epoch_one_based"] != 8
        or abs(selected["external_equal_session_mean"] - EXPECTED_LOCAL) > 1.0e-12
        or selected["head_state_sha256"] != EXPECTED_HEAD
    ):
        raise RuntimeError("M2 epoch selection authority drift")
    selected_payload = torch.load(
        SELECTION_ROOT / "selected_head.pt", map_location="cpu", weights_only=False
    )
    if selected_payload["selection"] != selected:
        raise RuntimeError("selected head metadata drift")

    compat = COMPAT_ROOT / "seed42/film_heads.pt"
    compat.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"EMPTY": selected_payload["state_dict"]}, compat)

    source = HERE.parent / "evalai_m2_movement_t4_empty_v1/build_payload.py"
    spec = importlib.util.spec_from_file_location("m2_empty_export_base", source)
    base = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(base)
    base.RESULT = COMPAT_ROOT
    base.OUTPUT = OUTPUT
    base.EXPECTED_LOCAL = EXPECTED_LOCAL
    base.main()

    with OUTPUT.open("rb") as handle:
        payload = pickle.load(handle)
    payload["metadata"].update(
        {
            "candidate": "movement_t4_m33_empty_adapter_seed44_epoch08_epochpick",
            "adapter_training": (
                "visible-held-out-selected seed 44 epoch 8 from a frozen 3-seed x 12-epoch grid; "
                "base identity and decoder frozen"
            ),
            "epoch_pick": {
                "seed": 44,
                "epoch_one_based": 8,
                "selection_surface": "six locally visible external M2 sessions",
                "selection_equal_session_mean": EXPECTED_LOCAL,
                "head_state_sha256": EXPECTED_HEAD,
                "selection_receipt_sha256": sha256(SELECTION_ROOT / "selection.json"),
                "evalai_opened_during_selection": False,
            },
        }
    )
    with OUTPUT.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    receipt_path = OUTPUT.parent / "payload.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "schema": "m2_movement_t4_empty_epochpick_evalai_payload_v1",
            "payload_sha256": sha256(OUTPUT),
            "bytes": OUTPUT.stat().st_size,
            "selected_seed": 44,
            "selected_epoch_one_based": 8,
            "selected_head_state_sha256": EXPECTED_HEAD,
            "selection_receipt_sha256": sha256(SELECTION_ROOT / "selection.json"),
            "selection_surface": "six locally visible external M2 sessions",
            "evalai_opened_during_selection": False,
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
