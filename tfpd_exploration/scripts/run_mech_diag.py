"""Deliverable B receipt runner: forward-only mechanism diagnostics over a
frozen Stage-1 checkpoint (HANDOFF_POST_BILINEAR_CONSUMER_EXTENSION_20260815.md
§5.7).

Four same-checkpoint, no-retraining diagnostics per (model, session):
aligned T4 / exact zero (torch.zeros_like) / wrong-pair (row permutation via
the model's own frozen ``wrong_pair_carrier``) / activity-destroyed (random
per-unit circular time shift, frozen seed).  R^2 uses the house metric
family (`src.tfpd.synth.r2_score`).  T4 minus Z4 is the necessary
carrier-content check; the wrong-pair contrast is the distinctive one; a
wrong-pair penalty larger than the zero penalty is strong evidence but NOT
required (parent §8.6).

Modes:

- no checkpoint available and no --synthetic: DRY RUN — prints the plan
  matrix and exits 0;
- ``--checkpoint PATH``: rebuilds the Stage-1 decoder from the Lightning
  checkpoint state_dict (no LightningModule instantiation) and scores the
  diagnostic matrix on SYNTHETIC sessions (real neural data is never
  loaded);
- ``--synthetic N``: same matrix on a randomly initialized model (smoke);
- ``--datamodule-config PATH``: real-datamodule interface ONLY — the
  implementation is a deliberate TODO and refuses before instantiating
  anything (exit 4).

Receipt discipline replicates `scripts/run_stage0_synth.py` §4 (env hard
gate, launch/final closure equality over this deliverable's new files,
O_EXCL transactional write, fail-closed failure receipts).

Usage (from the package root):
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_mech_diag.py --dry-run
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_mech_diag.py --synthetic 4 \
        --output-root /tmp/mech_diag_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # package root, so `src.tfpd_lane` resolves

BOUND_PATTERNS = ("src/tfpd_lane/*.py", "scripts/run_mech_diag.py", "tests/test_mech_diag.py")

HANDOFF_DOC = "docs/HANDOFF_POST_BILINEAR_CONSUMER_EXTENSION_20260815.md"
STAGE1_ROOT = ROOT / "results/stage1_source_cells_v1"


def discover_checkpoint(arm: str) -> Path | None:
    """Newest epoch checkpoint under the Stage-1 arm directory (read-only)."""
    ckpt_dir = STAGE1_ROOT / arm / "epoch_ckpts"
    if not ckpt_dir.is_dir():
        return None
    candidates = sorted(ckpt_dir.glob("*.ckpt"))
    return candidates[-1] if candidates else None


def print_plan(checkpoints: dict[str, Path | None], num_sessions: int) -> None:
    plan = {
        "mode": "dry-run (no receipt written)",
        "diagnostics": ["aligned", "zero", "wrong_pair", "activity_destroyed"],
        "scoring": "src.tfpd.synth.r2_score (pooled variance-weighted)",
        "frozen_seeds": {"wrong_pair": 0, "destroy": 0},
        "synthetic_sessions": num_sessions,
        "checkpoints": {arm: str(p) if p else None for arm, p in checkpoints.items()},
        "real_data": "--datamodule-config is interface-only (TODO); never instantiated",
    }
    print(json.dumps(plan, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=None, help="Lightning .ckpt to rebuild the model from")
    parser.add_argument("--arm", default="bl_t4", choices=["bl_t4", "pv_t4"], help="Stage-1 arm for auto-discovery")
    parser.add_argument("--synthetic", type=int, default=None, metavar="NUM_SESSIONS", help="smoke mode: random-init model over a synthetic cohort")
    parser.add_argument("--num-sessions", type=int, default=8, help="synthetic cohort size when scoring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wrong-pair-seed", type=int, default=0)
    parser.add_argument("--destroy-seed", type=int, default=0)
    parser.add_argument("--datamodule-config", type=Path, default=None, help="interface only; NOT implemented (TODO)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan matrix and exit 0 (no receipt)")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/mech_diag_v1")
    args = parser.parse_args()

    if args.datamodule_config is not None:
        print(
            "--datamodule-config is an interface placeholder only: real-datamodule scoring "
            "is a deliberate TODO and no datamodule is instantiated here.",
            file=sys.stderr,
        )
        return 4

    checkpoints = {arm: (args.checkpoint if (args.checkpoint and arm == args.arm) else discover_checkpoint(arm)) for arm in ("bl_t4", "pv_t4")}
    if args.dry_run or (args.checkpoint is None and args.synthetic is None):
        print_plan(checkpoints, args.num_sessions)
        return 0

    from src.tfpd_lane.receipt import (
        enforce_environment,
        sha256_file,
        source_closure,
        write_receipt_transactionally,
    )

    environment = enforce_environment()

    import torch

    from src.tfpd_lane.mech_diag import (
        load_stage1_model,
        run_mech_diag_cohort,
        synthetic_cohort,
    )

    if torch.cuda.is_available():
        print("environment hard gate failed: CUDA is available to torch", file=sys.stderr)
        return 3

    receipt_path = args.output_root / "receipt.json"
    if receipt_path.exists():
        print(f"receipt root already exists: {args.output_root}", file=sys.stderr)
        return 2

    handoff = ROOT / HANDOFF_DOC
    launch_closure = source_closure(ROOT, BOUND_PATTERNS)

    failure_payload_base = {
        "schema": "tfpd_mech_diag_v1",
        "handoff": {
            "path": HANDOFF_DOC,
            "section": "5.7",
            "sha256": sha256_file(handoff),
        },
        "seed": args.seed,
        "environment": environment,
        "launch_closure": launch_closure,
    }

    try:
        if args.synthetic is not None:
            # random-init smoke model; hyperparameters mirror the Stage-1
            # bilinear cell (src/tfpd/stage1_module.py build_stage1_model)
            # without importing the LightningModule.
            from src.tfpd.bilinear_readin import BilinearTaskFrameDecoder

            torch.manual_seed(args.seed)
            model = BilinearTaskFrameDecoder(
                window_size=20, carrier_dim=4, feature_dim=16, embed_dim=16,
                hidden_dim=64, latent_dim=32, gru_hidden=64, num_covariates=2,
            )
            model.eval()
            model_source = {"mode": "synthetic_random_init", "model_name": "bilinear", "checkpoint": None}
            sessions = synthetic_cohort(seed=args.seed, num_sessions=args.synthetic, num_units=64)
        else:
            ckpt_path = args.checkpoint if args.checkpoint is not None else checkpoints[args.arm]
            if ckpt_path is None:
                print(f"no checkpoint found for arm {args.arm!r}; use --dry-run or --synthetic", file=sys.stderr)
                return 2
            model = load_stage1_model(ckpt_path)
            model_source = {"mode": "stage1_checkpoint", "checkpoint": str(ckpt_path)}
            sessions = synthetic_cohort(seed=args.seed, num_sessions=args.num_sessions, num_units=64)

        results = run_mech_diag_cohort(
            model, sessions,
            wrong_pair_seed=args.wrong_pair_seed,
            destroy_seed=args.destroy_seed,
        )
    except Exception as error:  # noqa: BLE001 — any crash is a fail-closed event
        failure_payload_base.update(
            {"status": "FAIL_EXECUTION_ERROR", "error": f"{type(error).__name__}: {error}"}
        )
        write_receipt_transactionally(receipt_path, failure_payload_base)
        print(f"execution error; failure receipt: {receipt_path}", file=sys.stderr)
        return 1

    final_closure = source_closure(ROOT, BOUND_PATTERNS)
    if final_closure["closure_sha256"] != launch_closure["closure_sha256"]:
        failure_payload_base.update(
            {"status": "FAIL_SOURCE_CLOSURE_DRIFT", "final_closure": final_closure}
        )
        write_receipt_transactionally(receipt_path, failure_payload_base)
        print("source closure drifted between launch and final", file=sys.stderr)
        return 1

    receipt = {
        **failure_payload_base,
        "final_closure": final_closure,
        "launch_final_closure_equal": True,
        "model_source": model_source,
        "seeds": {"wrong_pair": args.wrong_pair_seed, "destroy": args.destroy_seed, "cohort": args.seed},
        "mech_diag": results,
        "status": "COMPLETED_FORWARD_ONLY",
        "authorizes": (
            "descriptive mechanism evidence only; forward-only diagnostics authorize no "
            "training and no promotion decision"
        ),
    }
    write_receipt_transactionally(receipt_path, receipt)

    summary_path = args.output_root / "mech_diag.json"
    if not summary_path.exists():
        import os

        body = (json.dumps(results, indent=1, sort_keys=True) + "\n").encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(dir=str(args.output_root), prefix=".summary-", suffix=".tmp")
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o444)
        try:
            os.link(tmp_path, summary_path)  # O_EXCL semantics: never overwrite
            os.unlink(tmp_name)
        except FileExistsError:
            tmp_path.unlink(missing_ok=True)

    print(json.dumps({
        "status": receipt["status"],
        "receipt": str(receipt_path),
        "per_diagnostic_r2_mean": results["per_diagnostic_r2_mean"],
        "model_source": model_source,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
