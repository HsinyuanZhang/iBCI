#!/usr/bin/env python3
"""Stage CLI for PIT-M2 (prefix-invariance training on M2), V1.

Stages:

* ``attempt`` (default, inert): print the frozen pair contract; import no
  torch, open nothing, create nothing;
* ``smoke --execute``: the CPU matched-pair smoke (40 steps per arm, no CUDA
  at all; requires ``CUDA_VISIBLE_DEVICES`` empty or -1);
* ``train --execute --gpu-authorized --gpu-index {0,1} [--arm {t0m,c1m}]``:
  REFUSES without ``--gpu-authorized`` AND a live TARGET-GPU-idle check
  (nvidia-smi: the selected card must have < 100 MiB used, 0% utilization,
  and no compute apps; other cards may be busy -- amendment 2026-09-02).
  ``--arm`` trains one arm in this process so the pair may run as two
  parallel processes on one card; without it t0m then c1m run serially;
* ``phase3 --execute --gpu-authorized ...``: the 2x2 scoring under the same
  GPU authorization law (needs both arms COMPLETE).

The envelope (the route standard):

    cd /home/xinyuan/Work_host/SPINT
    PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SPINT \
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 \
    /home/xinyuan/miniconda3/envs/spint/bin/python \
        tfpd_exploration/scripts/run_pit_m2_v1.py --stage <stage> [--execute ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = REPO_ROOT / "tfpd_exploration"
for item in (str(REPO_ROOT), str(TFPD_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

STAGES = ("attempt", "smoke", "train", "phase3")


def _pair_spec_payload() -> dict[str, object]:
    from tfpd_exploration.src.pit_m2_v1 import plan

    return plan.dry_plan(REPO_ROOT)


def _require_cpu_envelope() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in ("", "-1"):
        raise SystemExit(
            "pit m2 smoke requires CUDA_VISIBLE_DEVICES empty or -1 (CPU only)")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("pit m2 stages require PYTHONNOUSERSITE=1 (route envelope)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="attempt")
    parser.add_argument("--execute", action="store_true",
                        help="execute the stage (attempt is always inert)")
    parser.add_argument("--gpu-authorized", action="store_true",
                        help="operator authorization for the train/phase3 stages")
    parser.add_argument("--gpu-index", type=int, default=0, choices=(0, 1),
                        help="the single visible card for the train/phase3 stages")
    parser.add_argument("--arm", choices=("t0m", "c1m"), default=None,
                        help="train ONE arm in this process (parallel-pair mode, "
                             "amendment 2026-09-02); default runs t0m then c1m "
                             "serially in this one process")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    if arguments.stage == "attempt" or not arguments.execute:
        print(json.dumps(_pair_spec_payload(), sort_keys=True, separators=(",", ":")))
        return 0

    if arguments.stage == "smoke":
        _require_cpu_envelope()
        from tfpd_exploration.src.pit_m2_v1 import driver

        terminal_sha, failure_sha = driver.execute_smoke(REPO_ROOT)
        print(json.dumps({"stage": "smoke", "terminal_sha256": terminal_sha,
                          "failure_sha256": failure_sha}, sort_keys=True))
        return 0 if terminal_sha else 1

    if arguments.stage in ("train", "phase3"):
        if not arguments.gpu_authorized:
            print(json.dumps({
                "stage": arguments.stage,
                "status": "REFUSED",
                "reason": "pass --gpu-authorized (operator decision required; "
                          "the target card must be idle)",
            }, sort_keys=True))
            return 2
        from tfpd_exploration.src.pit_m2_v1 import trainer

        # The refusal law runs BEFORE torch is imported by the driver: only the
        # TARGET card must be idle (amendment 2026-09-02); other cards may be
        # busy with other routes.
        trainer.require_train_authorization(gpu_authorized=True,
                                            gpu_index=arguments.gpu_index)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu_index)
        from tfpd_exploration.src.pit_m2_v1 import driver

        if arguments.stage == "train":
            arms = (arguments.arm,) if arguments.arm else ("t0m", "c1m")
            terminals = []
            for arm in arms:
                terminal_sha, failure_sha = driver.execute_arm(
                    REPO_ROOT, arm=arm, gpu_index=arguments.gpu_index,
                    gpu_authorized=True)
                terminals.append({"stage": f"train:{arm}", "terminal_sha256": terminal_sha,
                                  "failure_sha256": failure_sha})
            print(json.dumps(terminals, sort_keys=True))
            return 0 if all(item["terminal_sha256"] for item in terminals) else 1
        if arguments.arm:
            raise SystemExit("--arm applies to the train stage only")
        terminal_sha, failure_sha = driver.execute_phase3(
            REPO_ROOT, gpu_index=arguments.gpu_index, gpu_authorized=True)
        print(json.dumps({"stage": "phase3", "terminal_sha256": terminal_sha,
                          "failure_sha256": failure_sha}, sort_keys=True))
        return 0 if terminal_sha else 1

    raise SystemExit(f"unknown stage {arguments.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
