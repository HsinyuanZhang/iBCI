#!/usr/bin/env python3
"""CAL-AUG V1 successor trainer: one arm (T0 or C1) of the matched pair.

Work order: ``docs/WORKORDER_CAL_AUG_V1_20260829.md`` (binding).  This runner
is the successor of the sealed ``run_pop_robust_cell.py`` Cell-D producer and
mirrors its ``_run`` flow exactly (canonical initial state strict load +
bitwise proof, the sealed arm-runner datamodule builder, the seed-42 SessionBatchSampler,
``arm_runner.train_epoch`` under the passive dropout recorder, per-epoch
diagnostics, final-four SWA + ``seal_file``) with ONLY these additions:

* BOTH sealed trainers are importlib-loaded after SHA-256 verification against
  the pinned literals (``plan.PINNED_SHA256``), fail-closed BEFORE any data or
  model access;
* ``attempt.json`` (environment, GPU UUID binding, pinned SHAs, work-order SHA,
  arm, cycle, closure) is published BEFORE the datamodule is built;
* the CAL-AUG operator: a route-owned ``forward_pre_hook(with_kwargs=True)`` on
  the ``StreamingSpintModel``.  Arm ``c1`` rewrites ``kwargs["calib_trials"]``
  to the deterministic chronological prefix ``cycle[counter % len(cycle)]``;
  arm ``t0`` registers the SAME hook disabled (kwargs untouched).  Gated on
  ``module.training``; the counter advances only on training-mode forwards, so
  every eval/diagnostic forward is untouched and the sealed
  ``n_forwards_with_sampled_p == optimizer_steps`` invariant is preserved as a
  hard check;
* the in-run 100-step throughput probe (steps 20..120 after warmup) publishing
  ``probe.json`` with per-arm projections and pair GPU-hours;
* a hard wall-clock guard (default 8 h) with per-batch granularity that raises
  and publishes an atomic ``CELL_FAILED`` receipt (stage, steps completed,
  ``terminal_published=false``).

Fresh roots only: ``<out-root>/<arm>/`` must not pre-exist.  Receipts are
transactional (0444 + sidecar, O_EXCL).  Sealed files are never edited.
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
from src.cal_aug_v1.hook import CalPrefixOperator  # noqa: E402

ARM_NAMES = {"t0": "t0_operator_disabled", "c1": "c1_prefix_cycle"}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CAL-AUG V1 matched arm trainer")
    parser.add_argument("--arm", required=True, choices=list(plan.ARMS))
    parser.add_argument("--epochs", type=int, default=plan.EPOCHS)
    parser.add_argument("--steps-limit", type=int, default=None,
                        help="smoke/probe budget: cap optimizer steps per epoch (None = full)")
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/cal_aug_v1")
    parser.add_argument("--b3s-prefix-cycle", default=plan.DEFAULT_CYCLE_TEXT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-seconds", type=int, default=plan.HARD_TIMEOUT_SECONDS)
    parser.add_argument("--publish-probe-at-step", type=int, default=plan.PROBE_DEFAULT_STEPS,
                        help="probe window length in optimizer steps, measured from step 20")
    parser.add_argument("--record-steps", type=int, default=plan.RECORD_STEPS_DEFAULT)
    parser.add_argument("--seed", type=int, default=plan.SEED)
    parser.add_argument("--train-batch-size", type=int, default=plan.TRAIN_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--initial-state", type=Path,
        default=ROOT / "results/admission_arms_v1/canonical_initial_state.pt",
    )
    parser.add_argument(
        "--smoke-receipt", type=Path,
        default=ROOT / "results/cal_aug_v1/smoke/smoke.json",
        help="the smoke receipt whose first-epoch digests epoch 0 must reproduce",
    )
    parser.add_argument("--allow-missing-smoke-receipt", action="store_true",
                        help="deviation from launch discipline; never for an authoritative arm")
    parser.add_argument("--require-gpu-uuid", default=plan.BOUND_GPU_UUID)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3

    out_dir = Path(args.out_root) / ARM_NAMES[args.arm]
    if out_dir.exists():
        print(f"fresh arm output directory required: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    guard = receipts.DeadlineGuard(args.timeout_seconds)
    context = {
        "arm": args.arm, "stage": "initialization", "steps_completed": 0,
        "terminal_published": False, "started_utc": started,
    }
    try:
        return _run(args, out_dir, device, started, guard, context)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - failure receipts are mandatory
        context["failure"] = {
            "kind": type(exc).__name__,
            "detail": str(exc),
            "timeout": isinstance(exc, receipts.CalAugTimeout),
            "traceback": traceback.format_exc(),
        }
        _stack = getattr(main, "_receipt_stack", None)
        if _stack is not None:
            receipts.publish_failure(
                out_dir, _stack["receipt"],
                {
                    "schema": plan.SCHEMA + "_cell",
                    "arm": args.arm,
                    "started_utc": started,
                    "stage": context["stage"],
                    "steps_completed": context["steps_completed"],
                    "epochs_completed": context.get("epochs_completed", 0),
                    "timeout_seconds": args.timeout_seconds,
                    "elapsed_seconds": round(guard.elapsed(), 3),
                    "smoke": args.steps_limit is not None,
                    "failure": context["failure"],
                    "pinned_sha256": getattr(main, "_pinned", None),
                    "source_closure": getattr(main, "_closure_launch", None),
                },
            )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def _run(args, out_dir: Path, device, started: str, guard, context: dict) -> int:
    import lightning.pytorch as pl
    import torch
    from mc_maze.multisession_datamodule import SessionBatchSampler
    from torch.nn.parameter import UninitializedParameter
    from torch.utils.data import default_collate
    from torch.utils.data import DataLoader

    # ---- sealed boundary: pinned literals verified BEFORE any data access ----
    guard.check("pinned_verification", 0)
    pinned = receipts.verify_pinned_files(REPO)
    sealed_predecessors = receipts.verify_sealed_predecessors(REPO)
    stack = receipts.load_sealed_runner_stack(REPO, ROOT)
    arm_runner = stack["arm_runner"]
    pop_runner = stack["pop_runner"]
    arm_common = stack["arm_common"]
    matched_scorer = stack["matched_scorer"]
    receipt_mod = stack["receipt"]
    pop_robust = stack["pop_robust"]
    main._receipt_stack = stack  # type: ignore[attr-defined]
    main._pinned = pinned  # type: ignore[attr-defined]

    cycle = schedule.parse_cycle(args.b3s_prefix_cycle)
    closure_launch = receipt_mod.source_closure(ROOT, plan.BOUND_PATTERNS)
    main._closure_launch = closure_launch  # type: ignore[attr-defined]
    gpu = receipts.gpu_binding(args.device, args.require_gpu_uuid or None)

    # ---- attempt receipt: BEFORE any source/model/data access ----------------
    guard.check("attempt_publication", 0)
    attempt_payload = {
        "schema": plan.SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "cell": plan.CELL,
        "arm": args.arm,
        "arm_role": plan.ARM_ROLES[args.arm],
        "started_utc": started,
        "ordering": (
            "this receipt is sealed before the datamodule, the model or any "
            "session array is touched"
        ),
        "b3s_prefix_cycle": {
            "literal": args.b3s_prefix_cycle,
            "cycle": list(cycle),
            "cycle_sha256": schedule.cycle_digest(cycle),
            "law": "M_step = cycle[global_training_forward % len(cycle)]",
            "rng_consumed": "none: pure integer arithmetic; the dropout-p stream is untouched",
        },
        "budget": {
            "epochs": args.epochs,
            "steps_limit": args.steps_limit,
            "seed": args.seed,
            "train_batch_size": args.train_batch_size,
            "timeout_seconds": args.timeout_seconds,
            "record_steps": args.record_steps,
            "full_budget_declared": {
                "epochs": plan.EPOCHS, "steps_per_epoch": plan.STEPS_PER_EPOCH,
                "total_optimizer_steps": plan.TOTAL_OPTIMIZER_STEPS,
            },
        },
        "pinned_sha256": pinned,
        "sealed_predecessors": sealed_predecessors,
        "work_order": {
            "relative": plan.WORK_ORDER_RELATIVE,
            "sha256": pinned[plan.WORK_ORDER_RELATIVE],
        },
        "source_closure": closure_launch,
        "environment": receipts.environment_payload(receipt_mod, REPO, device, gpu),
        "gate_spec": plan.gate_spec_payload(),
    }
    attempt_sha = receipts.publish_attempt(out_dir, receipt_mod, attempt_payload)
    guard_stage = receipts.attempt_guard(out_dir)

    # ---- canonical initial state: strict load + bitwise proof ---------------
    guard.check("initial_state", 0)
    initial_sidecar = Path(str(args.initial_state) + ".sha256")
    if not args.initial_state.is_file() or not initial_sidecar.is_file():
        raise SystemExit("canonical initial state artifact missing")
    initial_sha = receipt_mod.sha256_file(args.initial_state)
    if initial_sha != initial_sidecar.read_text().split()[0]:
        raise SystemExit("canonical initial state SHA mismatch")
    initial_payload = torch.load(args.initial_state, map_location="cpu", weights_only=False)
    canonical_state = initial_payload["state_dict"]

    # ---- data (arm-A contract; development rosters never opened) -------------
    guard.check("data_materialization", 0)
    guard_stage("data_materialization")
    dm, a2 = arm_runner.build_datamodule(args)
    train_dataset = dm.train_dataset
    if len(dm.session_splits["train"]) != plan.SOURCE_SESSIONS_REQUIRED:
        raise SystemExit("strict-27 roster drift")
    if receipt_mod.sha256_file(a2.MANIFEST_PATH) != plan.EXPECTED_MANIFEST_SHA256:
        raise SystemExit("manifest SHA drift")
    behavior_semantic = a2.normalizer_value_sha256(*dm._behavior_stats)
    if not behavior_semantic.startswith(plan.BEHAVIOR_NORMALIZER_SEMANTIC_PREFIX):
        raise SystemExit("source behavior normalizer semantic SHA drift")
    no_target_path = {
        "within_dev_sessions_opened": False,
        "external_sub_m_opened": False,
        "formal_or_organizer_held_data_opened": False,
        "val_paths_resolved": list(dm.session_files["val"]),
        "test_paths_resolved": list(dm.session_files["test"]),
        "val_and_test_empty": dm.session_files["val"] == [] and dm.session_files["test"] == [],
    }
    t4_authority = arm_common.t4_authority_fingerprint(train_dataset.sessions)

    pl.seed_everything(args.seed, workers=True)
    model = pop_robust.build_population_robustness_model(seed=args.seed, cell="D")
    model.load_state_dict(canonical_state, strict=True)
    state_sha_loaded = arm_common.state_sha256(model)
    if state_sha_loaded != initial_payload["state_sha256"]:
        raise SystemExit("loaded state SHA != canonical artifact state SHA")
    proof = pop_robust.initial_state_equality_proof(canonical_state, seed=args.seed)
    for cell_proof in proof.values():
        if not cell_proof["state_keys_equal_to_canonical"] or cell_proof["shape_mismatches"]:
            raise SystemExit(f"initial-state equality proof failed for {cell_proof}")
    if len({p["trainable_parameters"] for p in proof.values()}) != 1:
        raise SystemExit("parameter-count drift across cells")
    model.to(device)

    # ---- the CAL-AUG operator (work order section 2) --------------------------
    operator = CalPrefixOperator(args.arm, cycle, record_steps=args.record_steps)
    operator.attach(model)

    sampler = SessionBatchSampler(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, seed=args.seed
    )
    steps_per_epoch = len(sampler)
    epochs = args.epochs
    if args.steps_limit is None and (steps_per_epoch != plan.STEPS_PER_EPOCH or epochs != plan.EPOCHS):
        raise SystemExit("budget drift: expected 33,925 steps/epoch over 48 epochs")
    loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ---- throughput probe state (epoch 0, steps [20, 20+N)) -------------------
    probe_state = {
        "warmup": plan.PROBE_WARMUP_STEPS,
        "n": int(args.publish_probe_at_step),
        "t0": None,
        "published": False,
        "payload": None,
    }
    epoch_holder = {"epoch": 0, "offset": 0}

    def probe_step(step_done_in_epoch: int) -> None:
        if epoch_holder["epoch"] != 0 or probe_state["published"]:
            return
        if step_done_in_epoch == probe_state["warmup"]:
            probe_state["t0"] = time.perf_counter()
        elif (
            probe_state["t0"] is not None
            and step_done_in_epoch == probe_state["warmup"] + probe_state["n"]
        ):
            seconds = time.perf_counter() - probe_state["t0"]
            payload = receipts.probe_payload(
                arm=ARM_NAMES[args.arm],
                measured_steps=probe_state["n"],
                warmup_steps=probe_state["warmup"],
                seconds=seconds,
            )
            receipt_mod.write_receipt_transactionally(
                Path(args.out_root) / plan.RESULT_SUBROOTS["probe"]
                / f"probe_{ARM_NAMES[args.arm]}.json",
                payload,
            )
            probe_state["payload"] = payload
            probe_state["published"] = True
            print(json.dumps({"probe": payload["steps_per_second"],
                              "pair_gpu_hours": payload["projected_pair_gpu_hours"]}), flush=True)

    wrapped_loader = receipts.DeadlineLoader(loader, guard, on_step=probe_step)

    fixed_batch = default_collate([train_dataset[i] for i in next(iter(sampler))])
    fx_neural, _fx_beh, fx_calib, _fx_sess, fx_side = fixed_batch[:5]
    fx_neural, fx_calib, fx_side = (
        fx_neural.to(device), fx_calib.to(device), fx_side.to(device),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=arm_common.ADAM_CONSTRUCTOR["lr"],
        betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
        eps=arm_common.ADAM_CONSTRUCTOR["eps"],
        weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
        amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
    )
    schedule_params = arm_common.schedule_params(epochs, steps_per_epoch)
    lr_fn = lambda step: arm_common.lr_at_step(step, epochs, steps_per_epoch)  # noqa: E731

    smoke = receipts.read_smoke_digests(args.smoke_receipt)
    if smoke is None and not args.allow_missing_smoke_receipt:
        raise SystemExit(
            f"smoke receipt missing at {args.smoke_receipt} (launch discipline: smoke "
            "integrity passes before any full arm; override only with "
            "--allow-missing-smoke-receipt for a non-authoritative probe)"
        )

    launch = {
        "schema": plan.SCHEMA + "_launch",
        "status": "CELL_LAUNCHED",
        "cell": plan.CELL,
        "arm": args.arm,
        "arm_name": ARM_NAMES[args.arm],
        "arm_role": plan.ARM_ROLES[args.arm],
        "predecessor_runner": "sealed run_pop_robust_cell.py cell D (sha "
                              + pinned["tfpd_exploration/scripts/run_pop_robust_cell.py"] + ")",
        "started_utc": started,
        "cal_aug_operator": operator.snapshot(),
        "initial_state": {
            "path": str(args.initial_state), "artifact_sha256": initial_sha,
            "state_dict_sha256": initial_payload["state_sha256"],
            "loaded_state_sha256": state_sha_loaded,
            "strict_load": True,
            "bitwise_equality_proof": proof,
        },
        "budget": {
            "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "steps_limit": args.steps_limit,
            "total_optimizer_steps": epochs * steps_per_epoch,
            "schedule": schedule_params, "optimizer": arm_common.ADAM_CONSTRUCTOR,
            "sampler": "mc_maze SessionBatchSampler(batch=32, shuffle, seed=42)",
            "timeout_seconds": args.timeout_seconds,
        },
        "data_contract": {
            "roster_n": len(dm.session_splits["train"]),
            "n_train_windows": len(train_dataset.window_indices),
            "within_dev_sessions_opened": False,
            "external_sub_m_opened": False,
            "formal_or_organizer_held_data_opened": False,
            "visible_side": "canonical normalized T4 (unchanged by CAL-AUG)",
            "behavior_normalizer_semantic_sha256": behavior_semantic,
            "t4_authority_sha256_first": t4_authority[
                dm.session_splits["train"][0]
            ],
            **no_target_path,
        },
        "probe_plan": {
            "warmup_steps": probe_state["warmup"],
            "measured_steps": probe_state["n"],
            "publishes_at_epoch0_step": probe_state["warmup"] + probe_state["n"],
            "note": (
                "a steps-limited run shorter than the probe endpoint cannot "
                "publish a probe; that is recorded, never fabricated"
            ),
        },
        "smoke_digest_binding": {
            "path": str(args.smoke_receipt),
            "present": smoke is not None,
            "first_epoch_digests_must_match": smoke is not None,
        },
        "attempt_sha256": attempt_sha,
        "disclosures": {
            "teacher_checkpoint_logits_or_loss_used": False,
            "pretraining_used": False,
            "width_changed": False,
            "extra_seed": False,
            "clipping": "none",
            "single_axis": (
                "ONLY the B3S calibration prefix is perturbed; query activity, "
                "T4, decoder, loss, optimizer, batch order and the dropout-p "
                "stream stay the sealed Cell-D recipe"
            ),
            "target_updates_gradients_or_optimizer_steps": 0,
        },
        "source_closure": closure_launch,
        "environment": attempt_payload["environment"],
    }
    receipt_mod.write_receipt_transactionally(out_dir / "launch_receipt.json", launch)

    swa_local = set(range(max(0, epochs - 4), epochs))
    diagnostics, checkpoints, invariant_failures = [], [], []
    phase_step = 0
    guard.check("training_start", 0)
    for epoch in range(epochs):
        epoch_holder["epoch"] = epoch
        wrapped_loader.steps_offset = phase_step
        guard.check(f"epoch_{epoch}_start", phase_step)
        t0 = time.time()
        training_forwards_before = operator.training_invocations
        with pop_robust.dynamic_dropout_recorder() as record, \
                pop_robust.fc_in_token_norm_recorder(model) as token_norms:
            stats = arm_runner.train_epoch(
                model, optimizer, wrapped_loader, "t4", lr_fn, device, phase_step,
                max_steps=args.steps_limit,
            )
        record["fc_in_token_norms"] = token_norms
        phase_step += stats["optimizer_steps"]
        context["steps_completed"] = phase_step
        context["stage"] = f"epoch_{epoch}_diagnostics"
        dropout_summary = pop_robust.summarize_dropout_record(record, 2)
        fixed_diag = pop_robust.fixed_batch_dropout_diagnostic(
            model, fx_neural[:8], fx_calib[:8], fx_side[:8]
        )
        training_forwards_this_epoch = operator.training_invocations - training_forwards_before

        w_side = arm_common.w_side_block(model)
        params_finite = all(
            bool(torch.isfinite(p.detach()).all().item())
            for p in model.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter) and p.numel()
        )
        with torch.no_grad():
            contribution = arm_common.post_pool_contribution(model, fx_calib[:8], fx_side[:8])
            model.eval()
            head_summary = pop_robust.per_head_attention_summary(
                model, fx_neural[:4], fx_calib[:4], fx_side[:4]
            )
            model.train()
        authority_ok = arm_common.t4_authority_fingerprint(train_dataset.sessions) == t4_authority

        # ---- first-epoch smoke digest binding (hard check) -------------------
        smoke_match = None
        if epoch == 0 and smoke is not None:
            n_smoke = int(smoke["steps"])
            if n_smoke <= stats["optimizer_steps"]:
                batches = sampler.batched_indices[:n_smoke]
                session_names = [
                    train_dataset.window_indices[i][0]
                    for batch in batches for i in batch
                ]
                order = schedule.batch_order_digest(batches, session_names)
                m_seq = [r["scheduled_m"] for r in operator.records[:n_smoke]]
                # the smoke receipt's equality block stores the per-arm batch
                # order digest as the FULL schedule dict and sampled-p as a
                # plain string; normalize before comparing
                raw_order = smoke["equality"]["batch_order_digest"][args.arm]
                expected_order = (
                    raw_order["combined_sha256"]
                    if isinstance(raw_order, dict) else raw_order
                )
                raw_p = smoke["equality"]["sampled_p_digest"][args.arm]
                expected_p = raw_p["sampled_p_sha256"] if isinstance(raw_p, dict) else raw_p
                smoke_match = {
                    "n": n_smoke,
                    "batch_order": {
                        "expected": expected_order,
                        "observed": order["combined_sha256"],
                        "match": order["combined_sha256"] == expected_order,
                    },
                    "sampled_p": {
                        "expected": expected_p,
                        "observed": schedule.float_stream_digest(record["sampled_p"][:n_smoke]),
                        "match": schedule.float_stream_digest(record["sampled_p"][:n_smoke])
                        == expected_p,
                    },
                    "prefix_sequence": {
                        "expected": smoke["arms"][args.arm]["prefix_sequence_sha256"],
                        "observed": schedule.sequence_digest(m_seq),
                        "match": schedule.sequence_digest(m_seq)
                        == smoke["arms"][args.arm]["prefix_sequence_sha256"],
                    },
                }
                smoke_match["all_match"] = all(
                    smoke_match[field]["match"]
                    for field in ("batch_order", "sampled_p", "prefix_sequence")
                )
            else:
                smoke_match = {
                    "skipped": True,
                    "reason": "smoke covered more steps than this epoch ran",
                }

        diag = {
            "epoch": epoch,
            "visible_side_mode": "t4",
            "duration_s": round(time.time() - t0, 3),
            **stats,
            "optimizer_steps_total": phase_step,
            "lr_expected_first": schedule_params["lr_at_step_0"],
            "lr_expected_last": schedule_params["lr_at_final_step"],
            "clipping_authorized": False,
            "w_side_norm": float(w_side.norm().item()),
            "norm_alpha_times_w_side": float(w_side.norm().item()),
            **contribution,
            "cal_aug_operator": {
                "training_invocations_total": operator.training_invocations,
                "training_invocations_this_epoch": training_forwards_this_epoch,
                "eval_invocations_total": operator.eval_invocations,
                "expected_training_forwards": stats["optimizer_steps"] + 1,
                "note": (
                    "the +1 is the fixed-batch dropout diagnostic's own train-mode "
                    "forward, which consumes one deterministic schedule slot"
                ),
            },
            "dynamic_dropout_summary": dropout_summary,
            "fixed_batch_dropout_diagnostic": fixed_diag,
            "per_head_attention": head_summary,
            "t4_authority_unchanged": authority_ok,
            "parameters_finite": params_finite,
            "optimizer_state_finite": arm_runner.optimizer_state_finite(optimizer),
            "state_dict_sha256": arm_common.state_sha256(model),
            "optimizer_state_sha256": arm_common.optimizer_sha256(optimizer),
            "smoke_digest_match": smoke_match,
            "checkpoint_saved": None,
        }
        ok_steps = (
            stats["optimizer_steps"] == min(args.steps_limit, steps_per_epoch)
            if args.steps_limit is not None
            else stats["optimizer_steps"] == steps_per_epoch
        )
        loss_finite = (
            stats["nonfinite_loss_steps"] == 0
            and stats["train_loss_mean_per_step"] == stats["train_loss_mean_per_step"]
        )
        if not all([
            stats["visible_side_violation_count"] == 0,
            stats["nonfinite_loss_steps"] == 0,
            stats["nonfinite_grad_steps"] == 0,
            params_finite, diag["optimizer_state_finite"], authority_ok, ok_steps,
            head_summary["n_heads"] == 2,
            dropout_summary["n_recorded_unit_mask_calls"] == stats["optimizer_steps"],
            dropout_summary["n_forwards_with_sampled_p"] == stats["optimizer_steps"],
            training_forwards_this_epoch == stats["optimizer_steps"] + 1,
            loss_finite,
            smoke_match is None or bool(smoke_match.get("all_match", True)),
        ]):
            invariant_failures.append(epoch)
        if epoch in swa_local:
            name = f"epoch{epoch:03d}.ckpt"
            path = out_dir / name
            torch.save(
                {
                    "kind": "cal_aug_v1_ckpt",
                    "arm": args.arm,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "state_dict_sha256": diag["state_dict_sha256"],
                    "optimizer_state_sha256": diag["optimizer_state_sha256"],
                },
                path,
            )
            sha = arm_runner.seal_file(path)
            diag["checkpoint_saved"] = name
            checkpoints.append({"file": name, "epoch": epoch, "sha256": sha})
        diagnostics.append(diag)
        receipt_mod.write_receipt_transactionally(
            out_dir / f"epoch{epoch:03d}_diagnostics.json",
            {"schema": plan.SCHEMA + "_epoch_diagnostics", "arm": args.arm, **diag},
        )
        print(json.dumps({
            "arm": args.arm, "epoch": epoch,
            "loss": round(stats["train_loss_mean_per_step"], 6),
            "lr_last": stats["lr_last"],
            "p_mean": round(dropout_summary["sampled_p_mean"] or 0.0, 4),
            "prefix_forwards": training_forwards_this_epoch,
        }), flush=True)
        context["epochs_completed"] = epoch + 1
        guard.check(f"epoch_{epoch}_end", phase_step)

    operator.detach()

    # ---- final-four SWA + finite forward smoke --------------------------------
    guard.check("swa", phase_step)
    swa_path = None
    swa_sha = None
    swa_manifest = None
    swa_finite = None
    if len(checkpoints) == 4:
        swa_path = out_dir / "swa_final4.pt"
        swa_manifest = matched_scorer.build_swa_final_four(
            [out_dir / c["file"] for c in checkpoints], swa_path
        )
        swa_sha = arm_runner.seal_file(swa_path)
        swa_state = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]
        swa_model = pop_robust.build_population_robustness_model(seed=args.seed, cell="D")
        swa_model.load_state_dict(swa_state, strict=True)
        swa_model.to(device).eval()
        with torch.no_grad():
            pred, _ = swa_model(
                fx_neural[:4], calib_trials=fx_calib[:4], side_features=fx_side[:4]
            )
            swa_finite = bool(torch.isfinite(pred).all().item())
        del swa_model
    elif args.steps_limit is None:
        raise SystemExit(f"expected 4 SWA checkpoints, got {len(checkpoints)}")

    closure_final = receipt_mod.source_closure(ROOT, plan.BOUND_PATTERNS)
    closure_equal = closure_final["closure_sha256"] == closure_launch["closure_sha256"]
    full_run = args.steps_limit is None
    status = (
        ("CAL_AUG_CELL_TERMINAL" if full_run
         else "CAL_AUG_CELL_STEPS_LIMITED_COMPLETE__NON_AUTHORITATIVE")
        if not invariant_failures and closure_equal and len(diagnostics) == epochs
        and (swa_finite is True or (not full_run and swa_finite is None))
        else "CAL_AUG_CELL_INVARIANT_OR_CLOSURE_FAILURE"
    )
    context["terminal_published"] = True
    receipt_mod.write_receipt_transactionally(
        out_dir / "terminal.json",
        {
            "schema": plan.SCHEMA + "_cell",
            "status": status,
            "cell": plan.CELL,
            "arm": args.arm,
            "arm_name": ARM_NAMES[args.arm],
            "arm_role": plan.ARM_ROLES[args.arm],
            "steps_limit": args.steps_limit,
            "timeout_seconds": args.timeout_seconds,
            "wall_seconds": round(guard.elapsed(), 3),
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attempt_sha256": attempt_sha,
            "initial_state": launch["initial_state"],
            "budget": launch["budget"],
            "data_contract": launch["data_contract"],
            "cal_aug_operator_final": operator.snapshot(),
            "throughput_probe": probe_state["payload"] or {"published": False},
            "epochs_run": len(diagnostics),
            "diagnostics_per_epoch": diagnostics,
            "invariant_failures": invariant_failures,
            "checkpoints": checkpoints,
            "swa": (
                None if swa_path is None
                else {
                    "path": str(swa_path), "sha256": swa_sha,
                    "window_epochs": [c["epoch"] for c in checkpoints],
                    "manifest": swa_manifest,
                    "strict_reload_finite_forward_smoke": swa_finite,
                }
            ),
            "disclosures": launch["disclosures"],
            "smoke_digest_binding": launch["smoke_digest_binding"],
            "source_closure": {"launch": closure_launch, "final": closure_final,
                               "launch_final_closure_equal": closure_equal},
            "environment": launch["environment"],
        },
    )
    print(json.dumps({
        "arm": args.arm, "status": status,
        "epochs_run": len(diagnostics),
        "final_loss": round(diagnostics[-1]["train_loss_mean_per_step"], 6),
        "swa_sha256": (swa_sha or "")[:16],
        "probe_steps_per_second": (
            probe_state["payload"]["steps_per_second"] if probe_state["payload"] else None
        ),
    }, indent=1))
    return 0 if status in (
        "CAL_AUG_CELL_TERMINAL", "CAL_AUG_CELL_STEPS_LIMITED_COMPLETE__NON_AUTHORITATIVE"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
