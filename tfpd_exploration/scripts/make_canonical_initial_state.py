#!/usr/bin/env python3
"""Mint the one canonical shared initial-state artifact for the Gate-2 arms.

HANDOFF_CARRIER_ADMISSION_CURRICULUM_20260816.md §6.1: "one immutable initial
state loaded strictly by all arms".  Builds the seed-42 teacher-free
spintshape model exactly once (src/tfpd/spintshape_module.py:
build_spintshape_model(seed=42)), verifies the §3 zero-column invariant,
saves the state dict immutably (0444 + sha256 sidecar) under
results/admission_arms_v1/, and proves a strict reload into a fresh model is
bitwise identical.  Every arm refuses to start without this artifact and
asserts its own loaded state SHA equals the SHA sealed here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


arm_common = _load_module("tfpd_lane_arm_common", "src/tfpd_lane/arm_common.py")
receipt_mod = _load_module("tfpd_lane_receipt", "src/tfpd_lane/receipt.py")


def _seal_file(path: Path) -> str:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    digest = arm_common.sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(digest + "  " + path.name + "\n")
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/admission_arms_v1"
    )
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3

    import torch

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "canonical_initial_state.pt"
    receipt_path = out_dir / "canonical_initial_state_receipt.json"
    if artifact.exists() or receipt_path.exists():
        print(
            f"canonical initial state already exists (immutable): {artifact}",
            file=sys.stderr,
        )
        return 2

    spintshape = _load_module("tfpd_spintshape_module", "src/tfpd/spintshape_module.py")

    import lightning.pytorch as pl

    pl.seed_everything(args.seed, workers=True)
    model = spintshape.build_spintshape_model(seed=args.seed)
    encoder = model.id_encoder
    if getattr(encoder, "variant", "") != "B3S":
        print("unexpected encoder variant", file=sys.stderr)
        return 1
    if tuple(encoder.post_pool[0].weight.shape) != (
        encoder.hidden_dim,
        encoder.hidden_dim + encoder.side_dim,
    ):
        print("post_pool[0] geometry drift (expected Linear(68,64))", file=sys.stderr)
        return 1
    w_side = arm_common.w_side_block(model)
    if int(torch.count_nonzero(w_side).item()) != 0 or bool(w_side.signbit().any().item()):
        print("W_side is not exactly positive zero at construction", file=sys.stderr)
        return 1

    state = model.state_dict()
    state_sha = arm_common.state_sha256(model)
    payload = {
        "kind": "tfpd_admission_canonical_initial_state_v1",
        "seed": args.seed,
        "builder": "src/tfpd/spintshape_module.py:build_spintshape_model",
        "state_dict": state,
        "state_sha256": state_sha,
        "w_side_exact_positive_zero": True,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
    }
    torch.save(payload, artifact)
    artifact_sha = _seal_file(artifact)

    # strict reload into a fresh model must be bitwise identical
    reloaded = torch.load(artifact, map_location="cpu", weights_only=False)
    fresh = spintshape.build_spintshape_model(seed=args.seed)
    fresh.load_state_dict(reloaded["state_dict"], strict=True)
    reload_sha = arm_common.state_sha256(fresh)
    strict_reload_equal = reload_sha == state_sha

    receipt = {
        "schema": "tfpd_admission_canonical_initial_state_v1",
        "status": "CANONICAL_INITIAL_STATE_SEALED" if strict_reload_equal else "RELOAD_MISMATCH",
        "seed": args.seed,
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_sha,
        "state_dict_sha256": state_sha,
        "strict_reload_state_sha256": reload_sha,
        "strict_reload_bitwise_equal": strict_reload_equal,
        "w_side_exact_positive_zero": True,
        "adam_moments_present": False,
        "builder_closure": {
            "spintshape_module": receipt_mod.sha256_file(
                ROOT / "src/tfpd/spintshape_module.py"
            ),
            "arm_common": receipt_mod.sha256_file(ROOT / "src/tfpd_lane/arm_common.py"),
        },
        "created_utc": payload["created_utc"],
        "torch": torch.__version__,
        "note": (
            "shared, immutable, loaded strictly by arms A/B/C; no arm may rebuild it"
        ),
    }
    receipt_mod.write_receipt_transactionally(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "state_dict_sha256", "artifact_sha256")}, indent=1))
    return 0 if strict_reload_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
