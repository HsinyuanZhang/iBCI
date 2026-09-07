#!/usr/bin/env python3
"""Runner for M2 AJPF-C: continual-law matched joint training + scoring.

Stages (default dry):
  --stage train   paired 12-epoch training of the four modules on GPU0
  --stage score   CPU replay of the trained modules under their own law
  --stage execute train -> score in one process sequence

Receipts: tfpd_exploration/results/m2_ajpf_c_v1/{attempt.json, training.json,
score.json, terminal.json} (0444 + sidecars, prefix-preserving on failure).

GPU0 only (CUDA_VISIBLE_DEVICES=0 enforced for train); score is CPU-only.
Luna's AJPF roots are consumed strictly read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TFPD = ROOT / "tfpd_exploration"
from tfpd_exploration.src.m2_ajpf_c_v1 import plan as _plan
RESULT_ROOT = ROOT / _plan.RESULT_ROOT_RELATIVE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_pair(path: Path, payload: dict) -> str:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    path.write_bytes(body)
    path.chmod(0o444)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar.chmod(0o444)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="dry", choices=("dry", "train", "score", "execute"))
    parser.add_argument("--training-json", type=Path, default=None,
                        help="score from a predecessor root's training.json (successor score-only run)")
    args = parser.parse_args()
    if args.stage == "dry":
        from tfpd_exploration.src.m2_ajpf_c_v1 import plan
        print(json.dumps({
            "schema": plan.SCHEMA + "_dry_v1",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE",
            "laws": list(plan.LAWS),
            "modules": list(plan.MODULES),
            "epochs": plan.EPOCHS,
            "coordinate_budget": plan.TARGET_COORDINATE_BUDGET,
            "group_budget": plan.GROUP_BUDGET_PER_EPOCH,
            "wall_cap_per_law_s": plan.WALL_CAP_SECONDS_PER_LAW,
            "workorder": plan.WORKORDER_RELATIVE,
        }, sort_keys=True))
        return

    from tfpd_exploration.src.m2_ajpf_c_v1 import plan, score as score_module, train as train_module

    def progress(message: str) -> None:
        print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    attempt_path = RESULT_ROOT / "attempt.json"
    if not attempt_path.exists():
        write_pair(attempt_path, {
            "schema": plan.SCHEMA + "_attempt_v1",
            "status": "ATTEMPT_RESERVED",
            "workorder_sha256": sha256_file(ROOT / plan.WORKORDER_RELATIVE),
            "workorder_v2_addendum_sha256": sha256_file(ROOT / plan.WORKORDER_V2_ADDENDUM_RELATIVE),
            "v1_failed_root": plan.V1_FAILED_ROOT_RELATIVE,
            "probe_receipt_sha256": sha256_file(ROOT / plan.PROBE_RECEIPT_RELATIVE),
            "predecessor_training_json": (str(args.training_json) if args.training_json else None),
            "stage": args.stage,
            "cuda_for_training_only": True,
            "target_updates": 0,
            "checkpoint_updates": 0,
        })

    failure_path = RESULT_ROOT / "failure.json"
    if failure_path.exists():
        raise SystemExit("previous failure graph present; this attempt is closed")
    for leaf in ("training.json", "score.json", "terminal.json"):
        if (RESULT_ROOT / leaf).exists():
            raise SystemExit(f"{leaf} already present; one-shot cell")

    try:
        if args.stage in ("train", "execute"):
            import torch
            require_env = os.environ.get("CUDA_VISIBLE_DEVICES")
            if require_env != "0":
                raise RuntimeError("train stage requires CUDA_VISIBLE_DEVICES=0 (GPU0 only)")
            base, data_module, ajpf_runner = train_module.prepare_base(ROOT)
            trainings = {}
            for law in plan.LAWS:
                trainings[law] = train_module.train_one_law(
                    repo_root=ROOT, law=law, base_module=base, data_module=data_module,
                    ajpf_runner=ajpf_runner, device="cuda:0", result_root=RESULT_ROOT,
                    progress=progress)
                del base
                # rebuild a CPU base for the next law from the sealed loader
                base, data_module, ajpf_runner = train_module.prepare_base(ROOT)
            write_pair(RESULT_ROOT / "training.json", {
                "schema": plan.SCHEMA + "_training_v1",
                "status": "TRAINING_COMPLETE",
                "laws": trainings,
            })
        if args.stage == "score":
            import torch
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise RuntimeError("score stage must run CPU-only (CUDA_VISIBLE_DEVICES empty)")
            training_path = (args.training_json or (RESULT_ROOT / "training.json"))
            training = json.loads(training_path.read_text(encoding="utf-8"))
            base, data_module, ajpf_runner = train_module.prepare_base(ROOT)
            import copy
            import io
            modules = {"pooled": base}
            for arm in plan.MODULES:
                entry = training["laws"][plan.LAW_OF_MODULE[arm]]["checkpoints"][arm]
                body = (ROOT / entry["path"]).read_bytes()
                digest = hashlib.sha256(body).hexdigest()
                if digest != entry["sha256"]:
                    raise RuntimeError(f"{arm} checkpoint body drift")
                module = copy.deepcopy(base)
                if arm.endswith("R1"):
                    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import adapter
                    adapter.install_after_strict_load(module.student, "J-R1")
                module.load_state_dict(
                    torch.load(io.BytesIO(body), map_location="cpu", weights_only=True), strict=True)
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                state_sha = ajpf_runner._student_state_sha256(module.student)
                if state_sha != entry["student_state_sha256"]:
                    raise RuntimeError(f"{arm} reload state drift")
                modules[arm] = module
            summary = score_module.score_all(
                torch=torch, repo_root=ROOT, data_module=data_module, modules=modules,
                progress=progress)
            write_pair(RESULT_ROOT / "score.json", summary)
        write_pair(RESULT_ROOT / "terminal.json", {
            "schema": plan.SCHEMA + "_terminal_v1",
            "status": "TERMINAL",
            "stage": args.stage,
            "attempt_sha256": sha256_file(attempt_path),
            "training_sha256": (sha256_file(RESULT_ROOT / "training.json")
                                if (RESULT_ROOT / "training.json").exists() else None),
            "score_sha256": (sha256_file(RESULT_ROOT / "score.json")
                             if (RESULT_ROOT / "score.json").exists() else None),
            "terminal_xor_failure": True,
            "target_updates": 0,
            "parameter_updates_during_scoring": 0,
        })
        print(json.dumps({"status": "TERMINAL", "root": str(RESULT_ROOT)}, sort_keys=True))
    except BaseException as error:  # fail-closed, prefix preserved
        published = [leaf for leaf in ("attempt.json", "training.json", "score.json")
                     if (RESULT_ROOT / leaf).exists()]
        write_pair(RESULT_ROOT / "failure.json", {
            "schema": plan.SCHEMA + "_failure_v1",
            "status": "FAIL_CLOSED",
            "published_prefix": published,
            "exception_class": type(error).__name__,
            "exception_message": str(error)[:600],
            "traceback_tail": traceback.format_exc()[-1200:],
            "terminal_xor_failure": False,
        })
        raise


if __name__ == "__main__":
    main()
