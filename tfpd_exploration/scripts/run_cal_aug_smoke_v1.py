#!/usr/bin/env python3
"""CAL-AUG V1 smoke: BOTH arms' first N steps on the training GPU (§3).

Feasibility + no-harm gate ONLY — it can never select the method.  One process
runs arm T0 then arm C1 for N steps (default 40) each, on the selected GPU,
producing every equality evidence item of work order §3 / guidance §6.5:

* exact initial-state equality (canonical artifact + strict-load ``state_sha256``);
* optimizer and schedule parameter equality (constructor + schedule receipts);
* first-N batch indices + session order digest equality across arms;
* first-N dropout-p sequence digest equality across arms (each arm reseeds, so
  the streams must be bit-identical);
* the C1 prefix length sequence (exact M list) + visible-slice digests, and the
  unchanged full-block digests (the operator slices, it never mutates input);
* T0 vs C1 first-N query-neural pre-dropout input digest equality (read-only
  forward-pre-hook hashing the neural bytes entering the model);
* B3S consumes the declared variable prefix: per-step ``push_trial`` count ==
  M via the encoder-state probe (T0: 30 every step);
* T4 bytes unchanged (``t4_authority_fingerprint``);
* finite loss / gradients / Adam state;
* no target path resolved (val/test rosters empty; external root never opened).

Writes ``results/cal_aug_v1/smoke/smoke.json`` (0444 + sidecar) and exits
nonzero on any failed equality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
# ROOT last: tfpd_exploration/src must win `import src` over the
# streaming tree; the streaming components are either file-path loaded
# by the sealed arm runner or merged via src.__path__ inside tfpd_lane.
sys.path.insert(0, str(ROOT))

from src.cal_aug_v1 import plan, receipts, schedule  # noqa: E402
from src.cal_aug_v1.hook import (  # noqa: E402
    CalPrefixOperator,
    EncoderTrialProbe,
    make_read_only_neural_digest_hook,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CAL-AUG V1 matched smoke (both arms)")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/cal_aug_v1")
    parser.add_argument("--b3s-prefix-cycle", default=plan.DEFAULT_CYCLE_TEXT)
    parser.add_argument("--seed", type=int, default=plan.SEED)
    parser.add_argument("--train-batch-size", type=int, default=plan.TRAIN_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--initial-state", type=Path,
        default=ROOT / "results/admission_arms_v1/canonical_initial_state.pt",
    )
    parser.add_argument("--require-gpu-uuid", default=plan.BOUND_GPU_UUID)
    parser.add_argument("--epochs", type=int, default=plan.EPOCHS,
                        help="the schedule the full arms use (warmup/cosine shape)")
    return parser.parse_args(argv)


class _CountingLoader:
    """Yield batches; after each processed step record the encoder probe count.

    ``arm_runner.train_epoch`` runs its whole per-step body between two ``next``
    calls on this generator, so the recorded cumulative ``push_trial`` count is
    exactly the count after each step's forward.
    """

    def __init__(self, loader, probe: EncoderTrialProbe, store: list):
        self.loader = loader
        self.probe = probe
        self.store = store

    def __iter__(self):
        for batch in self.loader:
            yield batch
            self.store.append(self.probe.count)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    import torch
    from torch.nn.parameter import UninitializedParameter

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3

    smoke_dir = Path(args.out_root) / plan.RESULT_SUBROOTS["smoke"]
    if smoke_dir.exists():
        print(f"fresh smoke output directory required: {smoke_dir}", file=sys.stderr)
        return 2
    smoke_dir.mkdir(parents=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    context = {"stage": "initialization"}
    try:
        return _run(args, smoke_dir, device, started, context)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        stack = getattr(main, "_receipt_stack", None)
        if stack is not None:
            receipts.publish_failure(
                smoke_dir, stack["receipt"],
                {
                    "schema": plan.SCHEMA + "_smoke",
                    "started_utc": started,
                    "stage": context["stage"],
                    "failure": {
                        "kind": type(exc).__name__, "detail": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def _run(args, smoke_dir: Path, device, started: str, context: dict) -> int:
    import lightning.pytorch as pl
    import torch
    from torch.nn.parameter import UninitializedParameter
    from mc_maze.multisession_datamodule import SessionBatchSampler
    from torch.utils.data import DataLoader

    context["stage"] = "pinned_verification"
    pinned = receipts.verify_pinned_files(REPO)
    sealed_predecessors = receipts.verify_sealed_predecessors(REPO)
    stack = receipts.load_sealed_runner_stack(REPO, ROOT)
    arm_runner = stack["arm_runner"]
    arm_common = stack["arm_common"]
    receipt_mod = stack["receipt"]
    pop_robust = stack["pop_robust"]
    main._receipt_stack = stack  # type: ignore[attr-defined]

    cycle = schedule.parse_cycle(args.b3s_prefix_cycle)
    closure_launch = receipt_mod.source_closure(ROOT, plan.BOUND_PATTERNS)
    gpu = receipts.gpu_binding(args.device, args.require_gpu_uuid or None)

    # ---- attempt receipt BEFORE any source/model/data access -----------------
    context["stage"] = "attempt_publication"
    attempt_sha = receipts.publish_attempt(
        smoke_dir, receipt_mod,
        {
            "schema": plan.SCHEMA + "_smoke_attempt",
            "status": "ATTEMPT_PUBLISHED",
            "cell": plan.CELL,
            "kind": "smoke_feasibility_no_harm_gate_only",
            "can_select_method": False,
            "started_utc": started,
            "steps": args.steps,
            "b3s_prefix_cycle": {
                "literal": args.b3s_prefix_cycle, "cycle": list(cycle),
                "cycle_sha256": schedule.cycle_digest(cycle),
            },
            "pinned_sha256": pinned,
            "sealed_predecessors": sealed_predecessors,
            "work_order": {
                "relative": plan.WORK_ORDER_RELATIVE,
                "sha256": pinned[plan.WORK_ORDER_RELATIVE],
            },
            "source_closure": closure_launch,
            "environment": receipts.environment_payload(receipt_mod, REPO, device, gpu),
        },
    )
    receipts.require_attempt(smoke_dir, "data_materialization")

    # ---- data (train-only; development rosters never opened) ------------------
    context["stage"] = "data_materialization"
    dm, a2 = arm_runner.build_datamodule(args)
    train_dataset = dm.train_dataset
    if len(dm.session_splits["train"]) != plan.SOURCE_SESSIONS_REQUIRED:
        raise SystemExit("strict-27 roster drift")
    if receipt_mod.sha256_file(a2.MANIFEST_PATH) != plan.EXPECTED_MANIFEST_SHA256:
        raise SystemExit("manifest SHA drift")
    behavior_semantic = a2.normalizer_value_sha256(*dm._behavior_stats)
    if not behavior_semantic.startswith(plan.BEHAVIOR_NORMALIZER_SEMANTIC_PREFIX):
        raise SystemExit("source behavior normalizer semantic SHA drift")
    t4_authority = arm_common.t4_authority_fingerprint(train_dataset.sessions)
    no_target_path = {
        "within_dev_sessions_opened": False,
        "external_sub_m_opened": False,
        "formal_or_organizer_held_data_opened": False,
        "val_paths_resolved": list(dm.session_files["val"]),
        "test_paths_resolved": list(dm.session_files["test"]),
        "val_and_test_empty": dm.session_files["val"] == [] and dm.session_files["test"] == [],
    }
    if not no_target_path["val_and_test_empty"]:
        raise SystemExit("target paths resolved by the datamodule (val/test not empty)")

    steps_per_epoch = None
    arms: dict[str, dict] = {}
    n = args.steps
    for arm in plan.ARMS:
        context["stage"] = f"arm_{arm}"
        pl.seed_everything(args.seed, workers=True)  # identical p-stream start per arm
        model = pop_robust.build_population_robustness_model(seed=args.seed, cell="D")
        initial_payload = torch.load(
            args.initial_state, map_location="cpu", weights_only=False
        )
        model.load_state_dict(initial_payload["state_dict"], strict=True)
        state_sha_loaded = arm_common.state_sha256(model)
        if state_sha_loaded != initial_payload["state_sha256"]:
            raise SystemExit("loaded initial state SHA != artifact state SHA")
        model.to(device)

        operator = CalPrefixOperator(arm, cycle, record_steps=n)
        operator.attach(model)
        neural_store: list[dict] = []
        neural_handle = model.register_forward_pre_hook(
            make_read_only_neural_digest_hook(neural_store, limit=None), with_kwargs=True
        )

        sampler = SessionBatchSampler(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, seed=args.seed
        )
        steps_per_epoch = len(sampler)
        loader = DataLoader(
            train_dataset, batch_sampler=sampler, num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=arm_common.ADAM_CONSTRUCTOR["lr"],
            betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
            eps=arm_common.ADAM_CONSTRUCTOR["eps"],
            weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
            amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
        )
        schedule_params = arm_common.schedule_params(args.epochs, steps_per_epoch)
        lr_fn = lambda step: arm_common.lr_at_step(step, args.epochs, steps_per_epoch)  # noqa: E731

        with EncoderTrialProbe(model.id_encoder) as probe:
            with pop_robust.dynamic_dropout_recorder() as record:
                stats = arm_runner.train_epoch(
                    model, optimizer, loader, "t4", lr_fn, device, 0,
                    max_steps=n,
                )
        neural_handle.remove()
        operator.detach()

        if stats["optimizer_steps"] != n:
            raise SystemExit(f"smoke step count drift: {stats['optimizer_steps']} != {n}")
        if len(record["sampled_p"]) != n:
            raise SystemExit("sampled-p count != smoke steps")
        if len(probe.calls_per_forward) != n:
            raise SystemExit("encoder forward-batch probe count != smoke steps")
        if len(operator.records) < n:
            raise SystemExit("operator records shorter than smoke steps")
        per_step_trials = list(probe.calls_per_forward)
        expected_trials = [operator.records[i]["effective_m"] for i in range(n)]
        batches = sampler.batched_indices[:n]
        session_names = [
            train_dataset.window_indices[i][0] for batch in batches for i in batch
        ]
        arms[arm] = {
            "initial_state": {
                "artifact_sha256": receipts.sha256_file(args.initial_state),
                "state_dict_sha256": initial_payload["state_sha256"],
                "loaded_state_sha256": state_sha_loaded,
                "strict_load": True,
            },
            "optimizer_constructor": dict(arm_common.ADAM_CONSTRUCTOR),
            "schedule_params": dict(schedule_params),
            "batch_order_digest": schedule.batch_order_digest(batches, session_names),
            "sampled_p_digest": schedule.float_stream_digest(record["sampled_p"][:n]),
            "sampled_p_count": len(record["sampled_p"]),
            "sampled_p_first": [float(v) for v in record["sampled_p"][: min(n, 8)]],
            "prefix_sequence": [operator.records[i]["scheduled_m"] for i in range(n)],
            "prefix_sequence_sha256": schedule.sequence_digest(
                [operator.records[i]["scheduled_m"] for i in range(n)]
            ),
            "effective_prefix_sequence": [operator.records[i]["effective_m"] for i in range(n)],
            "visible_slice_digests": [
                operator.records[i]["visible_slice_sha256"] for i in range(n)
            ],
            "full_block_digests": [
                operator.records[i]["full_block_sha256"] for i in range(n)
            ],
            "neural_digests": [entry["neural_sha256"] for entry in neural_store],
            "neural_digest_combined_sha256": schedule.string_list_digest(
                [entry["neural_sha256"] for entry in neural_store]
            ),
            "neural_hook_calls": len(neural_store),
            "neural_all_training": bool(neural_store)
            and all(entry["training"] for entry in neural_store),
            "encoder_trials_per_step": per_step_trials,
            "encoder_trials_expected": expected_trials,
            "encoder_trials_match": per_step_trials == expected_trials,
            "training_invocations": operator.training_invocations,
            "eval_invocations": operator.eval_invocations,
            "finiteness": {
                "nonfinite_loss_steps": stats["nonfinite_loss_steps"],
                "nonfinite_grad_steps": stats["nonfinite_grad_steps"],
                "nonfinite_side_violations": stats["visible_side_violation_count"],
                "parameters_finite": all(
                    bool(torch.isfinite(p.detach()).all().item())
                    for p in model.parameters()
                    if p.requires_grad
                    and not isinstance(p, UninitializedParameter)
                    and p.numel()
                ),
                "optimizer_state_finite": arm_runner.optimizer_state_finite(optimizer),
                "train_loss": stats["train_loss_mean_per_step"],
                "train_loss_finite": stats["train_loss_mean_per_step"]
                == stats["train_loss_mean_per_step"],
            },
            "operator_snapshot": operator.snapshot(),
        }

    t0_row, c1_row = arms["t0"], arms["c1"]

    def finite(row: dict) -> bool:
        f = row["finiteness"]
        return bool(
            f["nonfinite_loss_steps"] == 0
            and f["nonfinite_grad_steps"] == 0
            and f["nonfinite_side_violations"] == 0
            and f["parameters_finite"]
            and f["optimizer_state_finite"]
            and f["train_loss_finite"]
        )

    checks = {
        "initial_state_sha256": bool(
            t0_row["initial_state"]["loaded_state_sha256"]
            == c1_row["initial_state"]["loaded_state_sha256"]
            == t0_row["initial_state"]["state_dict_sha256"]
        ),
        "optimizer_and_schedule_params": bool(
            t0_row["optimizer_constructor"] == c1_row["optimizer_constructor"]
            and t0_row["schedule_params"] == c1_row["schedule_params"]
        ),
        "batch_order_digest": bool(
            t0_row["batch_order_digest"]["combined_sha256"]
            == c1_row["batch_order_digest"]["combined_sha256"]
        ),
        "sampled_p_digest": bool(t0_row["sampled_p_digest"] == c1_row["sampled_p_digest"]),
        "query_neural_pre_dropout_digest": bool(
            t0_row["neural_digest_combined_sha256"]
            == c1_row["neural_digest_combined_sha256"]
        ),
        "neural_one_forward_per_step": bool(
            t0_row["neural_hook_calls"] == n
            and c1_row["neural_hook_calls"] == n
            and t0_row["neural_all_training"]
            and c1_row["neural_all_training"]
        ),
        "calib_full_block_digests_unchanged": bool(
            t0_row["full_block_digests"] == c1_row["full_block_digests"]
        ),
        "encoder_trial_counts_match_declared_prefix": bool(
            t0_row["encoder_trials_match"] and c1_row["encoder_trials_match"]
        ),
        "c1_prefix_cycle_realized": bool(
            c1_row["effective_prefix_sequence"] == schedule.prefix_sequence(n, cycle)
        ),
        "t0_prefix_operator_disabled": bool(
            t0_row["effective_prefix_sequence"] == [plan.MAX_CALIBRATION_TRIALS] * n
        ),
        "t4_authority_fingerprint_bound": bool(len(t4_authority) == plan.SOURCE_SESSIONS_REQUIRED),
        "finite_t0": finite(t0_row),
        "finite_c1": finite(c1_row),
        "no_target_path": bool(no_target_path["val_and_test_empty"]),
    }
    equality = {
        "initial_state_sha256": {
            "t0": t0_row["initial_state"]["loaded_state_sha256"],
            "c1": c1_row["initial_state"]["loaded_state_sha256"],
            "canonical_state_sha256": t0_row["initial_state"]["state_dict_sha256"],
            "match": checks["initial_state_sha256"],
        },
        "optimizer_and_schedule_params": {
            "match": checks["optimizer_and_schedule_params"],
        },
        "batch_order_digest": {
            "t0": t0_row["batch_order_digest"],
            "c1": c1_row["batch_order_digest"],
            "match": checks["batch_order_digest"],
        },
        "sampled_p_digest": {
            "t0": t0_row["sampled_p_digest"],
            "c1": c1_row["sampled_p_digest"],
            "match": checks["sampled_p_digest"],
        },
        "query_neural_pre_dropout_digest": {
            "t0": t0_row["neural_digest_combined_sha256"],
            "c1": c1_row["neural_digest_combined_sha256"],
            "t0_hook_calls": t0_row["neural_hook_calls"],
            "c1_hook_calls": c1_row["neural_hook_calls"],
            "match": checks["query_neural_pre_dropout_digest"],
        },
        "calib_full_block_digests_unchanged": {
            "match": checks["calib_full_block_digests_unchanged"],
        },
        "encoder_trial_counts_match_declared_prefix": {
            "t0": t0_row["encoder_trials_match"],
            "c1": c1_row["encoder_trials_match"],
            "match": checks["encoder_trial_counts_match_declared_prefix"],
        },
        "c1_prefix_cycle_realized": {
            "match": checks["c1_prefix_cycle_realized"],
        },
        "t0_prefix_operator_disabled": {
            "match": checks["t0_prefix_operator_disabled"],
        },
        "t4_authority_fingerprint": {
            "mapping_sha256": schedule.mapping_digest(t4_authority),
            "n_sessions": len(t4_authority),
            "source": "one train-only datamodule shared by both arms in this process",
            "match": checks["t4_authority_fingerprint_bound"],
        },
        "finite_everywhere": {
            "t0": finite(t0_row), "c1": finite(c1_row),
            "match": checks["finite_t0"] and checks["finite_c1"],
        },
        "no_target_path": {"match": checks["no_target_path"], **no_target_path},
        "failed_checks": [name for name, ok in checks.items() if not ok],
    }
    equality["all_match"] = not equality["failed_checks"]

    closure_final = receipt_mod.source_closure(ROOT, plan.BOUND_PATTERNS)
    status = "SMOKE_PASSED" if equality["all_match"] else "SMOKE_FAILED__EQUALITY"
    receipt_mod.write_receipt_transactionally(
        smoke_dir / "smoke.json",
        {
            "schema": plan.SCHEMA + "_smoke",
            "status": status,
            "cell": plan.CELL,
            "can_select_method": False,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attempt_sha256": attempt_sha,
            "steps": args.steps,
            "steps_per_epoch": steps_per_epoch,
            "b3s_prefix_cycle": {
                "literal": args.b3s_prefix_cycle, "cycle": list(cycle),
                "cycle_sha256": schedule.cycle_digest(cycle),
            },
            "equality": equality,
            "arms": arms,
            "no_target_path": no_target_path,
            "data_contract": {
                "roster_n": len(dm.session_splits["train"]),
                "n_train_windows": len(train_dataset.window_indices),
                "behavior_normalizer_semantic_sha256": behavior_semantic,
            },
            "source_closure": {
                "launch": closure_launch, "final": closure_final,
                "launch_final_closure_equal":
                    closure_final["closure_sha256"] == closure_launch["closure_sha256"],
            },
            "environment": receipts.environment_payload(receipt_mod, REPO, device, gpu),
            "pinned_sha256": pinned,
        },
    )
    print(json.dumps({
        "status": status,
        "failed_checks": equality["failed_checks"],
        "sampled_p_digest": equality["sampled_p_digest"],
    }, indent=1))
    return 0 if status == "SMOKE_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
