#!/usr/bin/env python3
"""Fail-closed, one-date-at-a-time remote executor for future H1 date-LODO pairs.

This is deliberately *not* an H1 evaluation launcher.  It only knows how to
run one already-admitted source-training pair, in the non-bypassable order

``H-S -> H-S runtime-init probe -> no-target H-S terminal checker -> H-C
-> no-target pair checker``.

The future dates are kept separate from the active 19250108 pair: the script
rejects that date, uses a date-scoped tmux session/lock/run root, refuses to
overwrite any output, and consumes the immutable per-date CPU preflight and
prepare-only launch receipts.  It also refuses a repository installed at a
different absolute path from the one bound into those receipts; silently
rewriting a receipt path on the remote would destroy the chain of custody.

``--plan`` is the default and is target-free/read-only apart from stdout.
``--launch`` and ``--run-worker`` exist only for a later explicit operator
decision.  They are not used by the test suite.  Each frozen confirmation date
is launched separately rather than chained automatically, but the observed
H-C--H-S sign is never an authorisation or veto for the next date.  Review the
immutable terminal/evaluation receipts and only predeclared catastrophic
validity failures before that separate operator action; do not select dates by
intermediate effect signs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FUTURE_DATES = ("19250113", "19250115", "19250119", "19250120")
PAIR_SCHEMA = "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1"
PAIR_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED"
LAUNCH_SCHEMA = "h1_carrierid_date_lodo_phase2_paired_source_launch_receipt_v1"
LAUNCH_STATUS = "PASS_PAIRED_SOURCE_TRAINING_PREPARED_NOT_LAUNCHED"
SINGLE_SCHEMA = "h1_carrierid_date_lodo_phase2_single_terminal_check_v1"
SINGLE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_SOURCE_E49_CHECKPOINT_NO_TARGET"
PAIR_CHECK_SCHEMA = "h1_carrierid_date_lodo_phase2_paired_terminal_check_v1"
PAIR_CHECK_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIRED_SOURCE_E49_CHECKPOINTS_NO_TARGET"
RUNTIME_INIT_PROBE_SCHEMA = "h1_carrierid_date_lodo_phase2_hs_runtime_init_probe_v1"
RUNTIME_INIT_PROBE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_HS_RUNTIME_INIT_PROBE_SOURCE_ONLY_NO_TARGET"


class FutureQueueError(RuntimeError):
    """A remote queue precondition failed; no later arm may start."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise FutureQueueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_json(path: Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    path = path.resolve()
    _need(path.is_file(), f"required receipt is missing: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"receipt is not immutable mode 0444: {path}")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FutureQueueError(f"receipt is not valid JSON: {path}") from error
    _need(isinstance(body, dict), f"receipt must be an object: {path}")
    _need(body.get("schema") == schema and body.get("status") == status,
          f"receipt schema/status drift: {path}")
    return path, body, _sha256(path)


def _receipt_paths(repo_root: Path, outer_date: str) -> tuple[Path, Path]:
    _need(outer_date in FUTURE_DATES,
          f"future executor accepts only {FUTURE_DATES}; 19250108 and unadmitted dates are forbidden")
    receipt_dir = repo_root / "pilot_artifacts" / "h1_carrierid_date_lodo_phase2"
    return (
        receipt_dir / f"H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_PAIR_CPU_PREFLIGHT_v1.json",
        receipt_dir / f"H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_PAIRED_SOURCE_LAUNCH_RECEIPT_v1.json",
    )


def _require_receipt_bound_code(repo_root: Path, launch: Mapping[str, Any]) -> dict[str, str]:
    code = launch.get("code_sha256")
    _need(isinstance(code, Mapping), "launch receipt lacks code_sha256")
    expected_paths = {
        "data": repo_root / "src/data/h1_carrierid_date_lodo_phase2.py",
        "model": repo_root / "src/models/h1_carrierid_date_lodo_phase2_module.py",
        "hs_experiment": repo_root / "configs/experiment/h1_carrierid_date_lodo_hs_phase2.yaml",
        "hc_experiment": repo_root / "configs/experiment/h1_carrierid_date_lodo_hc_phase2.yaml",
        "terminal_callback": repo_root / "configs/callbacks/h1_carrierid_date_lodo_phase2_terminal.yaml",
    }
    actual: dict[str, str] = {}
    for key, path in expected_paths.items():
        _need(isinstance(code.get(key), str) and len(str(code[key])) == 64,
              f"launch receipt lacks a valid {key} code SHA")
        _need(path.is_file(), f"receipt-bound code path is missing: {path}")
        actual[key] = _sha256(path)
        _need(actual[key] == code[key], f"receipt-bound code SHA mismatch for {key}: {path}")
    return actual


def _validate_closure(*, repo_root: Path, outer_date: str) -> dict[str, Any]:
    """Validate immutable evidence and code without opening any NWB/data record."""

    repo_root = repo_root.resolve()
    pair_path, launch_path = _receipt_paths(repo_root, outer_date)
    pair_path, pair, pair_sha = _immutable_json(pair_path, schema=PAIR_SCHEMA, status=PAIR_STATUS)
    launch_path, launch, launch_sha = _immutable_json(launch_path, schema=LAUNCH_SCHEMA, status=LAUNCH_STATUS)

    _need(pair.get("outer_date") == outer_date, "pair preflight outer date drift")
    source = pair.get("source_binding")
    _need(isinstance(source, Mapping), "pair preflight lacks source binding")
    _need(source.get("outer_date") == outer_date, "pair preflight/source binding outer-date mismatch")
    _need(source.get("target_recordings_opened") == 0 and source.get("target_bytes_read") == 0,
          "pair preflight violates source-only target boundary")
    _need(source.get("warm_start_forbidden") is True, "pair preflight permits a warm start")
    _need(launch.get("outer_date") == outer_date, "launch receipt outer date drift")
    _need(launch.get("pair_preflight_sha256") == pair_sha,
          "launch receipt does not bind the immutable pair preflight bytes")
    _need(launch.get("source_binding_sha256") == pair.get("source_binding_sha256"),
          "launch receipt/source preflight binding SHA drift")

    # The frozen receipts bind absolute paths.  A remote stage at a different
    # root cannot be made equivalent by string substitution; require an exact
    # install instead.  This leaves current 19250108 stage paths untouched.
    _need(Path(str(launch.get("pair_preflight_path", ""))).resolve() == pair_path,
          "remote repo root differs from frozen launch-receipt pair path; do not remap it")
    phase1_path = Path(str(source.get("preflight_path", ""))).resolve()
    _need(phase1_path.is_file() and stat.S_IMODE(phase1_path.stat().st_mode) == 0o444,
          "source binding Phase-1 preflight is missing or mutable")
    _need(_sha256(phase1_path) == source.get("preflight_sha256"),
          "source binding Phase-1 preflight byte SHA drift")
    _need(phase1_path == repo_root / "pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json",
          "source binding requires a different repo root; do not rewrite it")

    pair_contract = pair.get("phase2_training_contract")
    launch_pair = launch.get("pair")
    _need(isinstance(pair_contract, Mapping) and isinstance(launch_pair, Mapping),
          "pair/launch receipts lack training contract")
    _need(pair_contract.get("arms") == ["H-S", "H-C"] and pair_contract.get("fresh_seed") == 42,
          "pair preflight arm/seed contract drift")
    _need(pair_contract.get("fixed_terminal_epoch_zero_based") == 49 and pair_contract.get("epochs") == 50,
          "pair preflight e49 contract drift")
    _need(launch_pair.get("fresh_seed") == 42 and launch_pair.get("terminal_checkpoint") == "epoch_049 only after 50 source epochs",
          "launch receipt seed/e49 contract drift")
    arms = launch_pair.get("arms")
    _need(isinstance(arms, Mapping), "launch receipt lacks arm contracts")
    expected = {
        "H-S": ("h1_carrierid_date_lodo_hs_phase2", "src.models.components.spint.SpintModel"),
        "H-C": ("h1_carrierid_date_lodo_hc_phase2", "src.models.components.h1_carrierid_spint.H1CarrierIdSpint"),
    }
    for arm, (config, consumer) in expected.items():
        arm_row = arms.get(arm)
        _need(isinstance(arm_row, Mapping), f"launch receipt lacks {arm} arm")
        _need(arm_row.get("experiment_config") == config and arm_row.get("consumer_target") == consumer,
              f"launch receipt {arm} consumer/config drift")
        _need(arm_row.get("fresh_seed") == 42 and arm_row.get("epochs") == 50 and
              arm_row.get("fixed_terminal_epoch_zero_based") == 49 and
              arm_row.get("checkpoint_warm_start_forbidden") is True,
              f"launch receipt {arm} fixed training contract drift")
        overrides = arm_row.get("compose_overrides")
        _need(isinstance(overrides, list) and f"phase2.outer_date={outer_date}" in overrides and
              f"phase2.pair_preflight_path={pair_path}" in overrides and
              f"phase2.phase1_preflight_path={phase1_path}" in overrides,
              f"launch receipt {arm} does not bind this date/receipt pair")

    checker = repo_root / "scripts/h1_carrierid_date_lodo_phase2_terminal_checker.py"
    _need(checker.is_file(), f"required no-target terminal checker is missing: {checker}")
    checker_text = checker.read_text(encoding="utf-8")
    for token in ("def check_single", "def check_pair", "SINGLE_STATUS", "PAIR_STATUS", "--hs-runtime-init-probe"):
        _need(token in checker_text, f"terminal checker is incompatible/missing {token}")
    preflight = repo_root / "scripts/h1_carrierid_date_lodo_phase2_preflight.py"
    _need(preflight.is_file(), f"required H-S runtime-init probe producer is missing: {preflight}")
    preflight_text = preflight.read_text(encoding="utf-8")
    for token in ("def run_hs_runtime_init_probe", "RUNTIME_INIT_PROBE_SCHEMA", "max_steps=1"):
        _need(token in preflight_text, f"runtime-init probe producer is incompatible/missing {token}")

    return {
        "repo_root": str(repo_root),
        "outer_date": outer_date,
        "phase1_preflight": str(phase1_path),
        "pair_preflight": str(pair_path),
        "pair_preflight_sha256": pair_sha,
        "launch_receipt": str(launch_path),
        "launch_receipt_sha256": launch_sha,
        "source_binding_sha256": pair["source_binding_sha256"],
        "code_sha256": _require_receipt_bound_code(repo_root, launch),
        "checker_path": str(checker),
        "runtime_probe_path": str(preflight),
    }


def _command_for_arm(*, closure: Mapping[str, Any], arm: str, run_dir: Path, python: Path) -> list[str]:
    _need(arm in ("H-S", "H-C"), f"unsupported arm: {arm}")
    repo_root = Path(str(closure["repo_root"]))
    config = "h1_carrierid_date_lodo_hs_phase2" if arm == "H-S" else "h1_carrierid_date_lodo_hc_phase2"
    return [
        str(python), str(repo_root / "src/train.py"), f"experiment={config}",
        f"hydra.run.dir={run_dir}", f"paths.root_dir={repo_root}", f"paths.work_dir={repo_root}",
        f"paths.data_dir={repo_root / 'data'}",
        f"phase2.outer_date={closure['outer_date']}",
        f"phase2.phase1_preflight_path={closure['phase1_preflight']}",
        f"phase2.pair_preflight_path={closure['pair_preflight']}",
        "ckpt_path=null", "test=false", "trainer.accelerator=gpu", "trainer.devices=1",
        "trainer.max_epochs=50", "trainer.min_epochs=50", "trainer.precision=32-true", "seed=42",
        f"task_name=h1_carrierid_date_lodo_{arm.lower().replace('-', '')}_{closure['outer_date']}",
    ]


def _runtime_probe_command(*, closure: Mapping[str, Any], hs_config: Path, probe_root: Path,
                           output: Path, python: Path) -> list[str]:
    """Build the one-step H-S lifecycle capture without selecting a target route."""

    return [
        str(python), str(closure["runtime_probe_path"]), "--run-hs-runtime-init-probe",
        "--pair-preflight", str(closure["pair_preflight"]), "--hs-config", str(hs_config),
        "--probe-root", str(probe_root), "--output", str(output),
    ]


def _checker_command(*, closure: Mapping[str, Any], mode: str, output: Path,
                     hs_checkpoint: Path, hc_checkpoint: Path | None,
                     hs_runtime_init_probe: Path, python: Path) -> list[str]:
    base = [str(python), str(closure["checker_path"]), "--pair-preflight", str(closure["pair_preflight"]),
            "--launch-receipt", str(closure["launch_receipt"]), "--output", str(output),
            "--hs-runtime-init-probe", str(hs_runtime_init_probe)]
    if mode == "single":
        return [*base, "--checkpoint", str(hs_checkpoint), "--arm", "H-S"]
    _need(mode == "pair" and hc_checkpoint is not None, "pair checker requires H-C checkpoint")
    return [*base, "--pair", "--hs-checkpoint", str(hs_checkpoint), "--hc-checkpoint", str(hc_checkpoint)]


def build_plan(*, repo_root: Path, outer_date: str, run_root: Path, python: Path) -> dict[str, Any]:
    """Return a target-free, side-effect-free plan for exactly one future date."""

    closure = _validate_closure(repo_root=repo_root, outer_date=outer_date)
    run_root = run_root.resolve()
    date_root = run_root / outer_date
    hs_run = date_root / "hs_s42_e49_v1"
    hc_run = date_root / "hc_s42_e49_v1"
    hs_checkpoint = hs_run / "checkpoints/fixed_epoch50/epoch_049.ckpt"
    hc_checkpoint = hc_run / "checkpoints/fixed_epoch50/epoch_049.ckpt"
    hs_config = hs_run / ".hydra/config.yaml"
    hs_probe_root = date_root / "hs_runtime_init_probe_v1"
    hs_probe = hs_probe_root / f"H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_HS_RUNTIME_INIT_PROBE_v1.json"
    hs_check = date_root / f"H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_HS_TERMINAL_CHECK_v1.json"
    pair_check = date_root / f"H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_PAIR_TERMINAL_CHECK_v1.json"
    forbidden = (hs_run, hc_run, hs_probe_root, hs_check, pair_check)
    return {
        "mode": "PLAN_ONLY_NO_TMUX_NO_GPU_NO_TARGET_EVALUATION",
        "closure": closure,
        "date_root": str(date_root),
        "tmux_session": f"h1_date_lodo_future_{outer_date}_pair_s42_e49",
        "lock_dir": str(Path("/tmp") / f"h1_date_lodo_future_{outer_date}_pair_s42_e49.lock"),
        "refuse_if_exists": [str(path) for path in forbidden],
        "runtime_init_probe": {"root": str(hs_probe_root), "receipt": str(hs_probe), "hs_config": str(hs_config)},
        "steps": [
            {"name": "H-S source-only e49", "command": _command_for_arm(closure=closure, arm="H-S", run_dir=hs_run, python=python)},
            {"name": "H-S source-only runtime-init probe", "command": _runtime_probe_command(
                closure=closure, hs_config=hs_config, probe_root=hs_probe_root, output=hs_probe, python=python)},
            {"name": "H-S no-target terminal checker", "command": _checker_command(
                closure=closure, mode="single", output=hs_check, hs_checkpoint=hs_checkpoint, hc_checkpoint=None,
                hs_runtime_init_probe=hs_probe, python=python)},
            {"name": "H-C source-only e49", "command": _command_for_arm(closure=closure, arm="H-C", run_dir=hc_run, python=python)},
            {"name": "H-S/H-C no-target pair checker", "command": _checker_command(
                closure=closure, mode="pair", output=pair_check, hs_checkpoint=hs_checkpoint,
                hc_checkpoint=hc_checkpoint, hs_runtime_init_probe=hs_probe, python=python)},
        ],
        "next_date_policy": "NO_AUTOMATIC_CHAINING: each frozen confirmation date requires a separate explicit launch after review of immutable terminal/evaluation receipts and predeclared catastrophic validity failures; the observed H-C--H-S sign never authorizes or vetoes a next date.",
        "target_evaluation": "NOT_IMPLEMENTED_AND_NOT_INVOKED",
    }


def _assert_outputs_absent(plan: Mapping[str, Any]) -> None:
    existing = [path for path in plan["refuse_if_exists"] if Path(path).exists()]
    _need(not existing, f"refusing any overwrite/resume; date-scoped outputs already exist: {existing}")


def _verify_terminal_receipt(path: Path, *, schema: str, status: str, outer_date: str) -> None:
    _, body, _ = _immutable_json(path, schema=schema, status=status)
    _need(body.get("outer_date") == outer_date, f"terminal checker outer-date drift: {path}")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0 and scope.get("target_bytes_read") == 0,
          f"terminal checker receipt violates target boundary: {path}")


def _verify_runtime_init_probe_receipt(path: Path, *, closure: Mapping[str, Any], outer_date: str,
                                       hs_config: Path) -> None:
    """Require the exact date's source-only bridge before H-S terminal checking."""

    _, body, _ = _immutable_json(path, schema=RUNTIME_INIT_PROBE_SCHEMA, status=RUNTIME_INIT_PROBE_STATUS)
    _need(body.get("arm") == "H-S" and body.get("outer_date") == outer_date,
          f"runtime-init probe arm/date drift: {path}")
    _need(body.get("pair_preflight") == {
        "path": str(closure["pair_preflight"]), "sha256": closure["pair_preflight_sha256"],
    }, f"runtime-init probe binds another pair receipt: {path}")
    _need(body.get("source_binding_sha256") == closure["source_binding_sha256"],
          f"runtime-init probe source binding drift: {path}")
    _need(body.get("production_hs_config") == {
        "path": str(hs_config), "sha256": _sha256(hs_config),
    }, f"runtime-init probe uses another H-S config: {path}")
    runtime = body.get("runtime_initialization")
    _need(isinstance(runtime, Mapping)
          and runtime.get("capture_hook") == "on_train_batch_start(epoch=0,batch_idx=0)"
          and runtime.get("capture_precedes_training_step_and_first_optimizer_step") is True
          and runtime.get("optimizer_steps_before_capture") == 0
          and runtime.get("backward_steps_before_capture") == 0
          and runtime.get("probe_optimizer_steps_after_capture") == 1,
          f"runtime-init probe lacks pre-optimizer capture evidence: {path}")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
          and scope.get("target_bytes_read") == 0 and scope.get("formal_heldout_opened") is False
          and scope.get("minival_opened") is False and scope.get("evalai_opened") is False
          and scope.get("checkpoint_loaded") is False and scope.get("checkpoint_created") is False,
          f"runtime-init probe violates source-only/no-checkpoint boundary: {path}")


def run_worker(*, repo_root: Path, outer_date: str, run_root: Path, python: Path, lock_dir: Path) -> None:
    """Execute one source-only pair.  Any nonzero subprocess stops the sequence."""

    _need(lock_dir.is_dir(), f"queue lock missing; worker is not an authorised tmux child: {lock_dir}")
    try:
        plan = build_plan(repo_root=repo_root, outer_date=outer_date, run_root=run_root, python=python)
        _assert_outputs_absent(plan)
        date_root = Path(plan["date_root"])
        # The root is made only after immutable receipt/code closure is rechecked.
        date_root.mkdir(parents=True, exist_ok=False)
        hs_checkpoint = date_root / "hs_s42_e49_v1/checkpoints/fixed_epoch50/epoch_049.ckpt"
        hc_checkpoint = date_root / "hc_s42_e49_v1/checkpoints/fixed_epoch50/epoch_049.ckpt"
        probe = plan["runtime_init_probe"]
        _need(isinstance(probe, Mapping), "queue plan lacks H-S runtime-init probe artifacts")
        hs_probe = Path(str(probe["receipt"]))
        hs_config = Path(str(probe["hs_config"]))
        hs_check = date_root / f"H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_HS_TERMINAL_CHECK_v1.json"
        pair_check = date_root / f"H1_CARRIERID_DATE_LODO_PHASE2_{outer_date}_PAIR_TERMINAL_CHECK_v1.json"
        steps: Sequence[Mapping[str, Any]] = plan["steps"]
        gpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        subprocess.run(steps[0]["command"], cwd=str(repo_root), check=True, env=gpu_env)
        _need(hs_checkpoint.is_file(), f"H-S did not produce required e49 checkpoint: {hs_checkpoint}")
        _need(hs_config.is_file(), f"H-S did not produce resolved config for runtime-init probe: {hs_config}")
        subprocess.run(steps[1]["command"], cwd=str(repo_root), check=True, env=gpu_env)
        _verify_runtime_init_probe_receipt(hs_probe, closure=plan["closure"], outer_date=outer_date, hs_config=hs_config)
        subprocess.run(steps[2]["command"], cwd=str(repo_root), check=True, env=cpu_env)
        _verify_terminal_receipt(hs_check, schema=SINGLE_SCHEMA, status=SINGLE_STATUS, outer_date=outer_date)
        # H-S can run for hours.  Recheck the complete immutable receipt/code
        # closure before spending a second GPU block on H-C, so an unrelated
        # concurrent source-tree mutation fails closed instead of silently
        # changing one arm of the pair.
        revalidated_closure = _validate_closure(repo_root=repo_root, outer_date=outer_date)
        _need(revalidated_closure == plan["closure"],
              "receipt/code closure changed after H-S; refusing to start H-C")
        subprocess.run(steps[3]["command"], cwd=str(repo_root), check=True, env=gpu_env)
        _need(hc_checkpoint.is_file(), f"H-C did not produce required e49 checkpoint: {hc_checkpoint}")
        subprocess.run(steps[4]["command"], cwd=str(repo_root), check=True, env=cpu_env)
        _verify_terminal_receipt(pair_check, schema=PAIR_CHECK_SCHEMA, status=PAIR_CHECK_STATUS, outer_date=outer_date)
        print(json.dumps({"status": "PASS_ONE_FUTURE_DATE_SOURCE_PAIR_NO_TARGET", "outer_date": outer_date,
                          "pair_terminal_receipt": str(pair_check),
                          "next_date_policy": plan["next_date_policy"]}, sort_keys=True), flush=True)
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            # Preserve a nonempty/corrupt lock for operator inspection; never remove it recursively.
            pass


def launch(*, plan: Mapping[str, Any], repo_root: Path, outer_date: str, run_root: Path,
           python: Path, predecessor_tmx: str | None) -> None:
    """Create exactly one detached worker after all no-overwrite checks pass."""

    _assert_outputs_absent(plan)
    _need(bool(predecessor_tmx),
          "--launch requires --predecessor-tmux for the active 19250108 H-C session; refuse ambiguous GPU overlap")
    session = str(plan["tmux_session"])
    prior = subprocess.run(["tmux", "has-session", "-t", predecessor_tmx], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _need(prior.returncode != 0,
          f"predecessor tmux session is still present: {predecessor_tmx}; no GPU overlap allowed")
    existing = subprocess.run(["tmux", "has-session", "-t", session], check=False,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _need(existing.returncode != 0, f"future tmux session already exists: {session}")
    lock_dir = Path(str(plan["lock_dir"]))
    try:
        lock_dir.mkdir()
    except FileExistsError as error:
        raise FutureQueueError(f"future queue lock already exists: {lock_dir}") from error
    # The source file is intentionally not required to carry an executable
    # permission bit after a minimal remote sync.  Invoke it through the
    # receipt-validated Python explicitly, rather than asking tmux/the shell
    # to execute the .py file directly.
    worker = [str(python), str(Path(__file__).resolve()), "--run-worker", "--repo-root", str(repo_root),
              "--outer-date", outer_date, "--run-root", str(run_root), "--python", str(python),
              "--lock-dir", str(lock_dir)]
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, *worker], check=True)
    except BaseException:
        try:
            lock_dir.rmdir()
        except OSError:
            pass
        raise
    print(json.dumps({"status": "LAUNCHED_ONE_FUTURE_DATE_SOURCE_PAIR", "tmux_session": session,
                      "outer_date": outer_date, "next_date_policy": plan["next_date_policy"]}, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="default: validate and print a target-free plan")
    mode.add_argument("--launch", action="store_true", help="explicitly create one detached future-date source worker")
    mode.add_argument("--run-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--outer-date", choices=FUTURE_DATES, default=FUTURE_DATES[0])
    parser.add_argument("--run-root", type=Path,
                        default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase2/future_remote_gpu_runs")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--predecessor-tmux", help="active 19250108 H-C session; mandatory for --launch and must be absent")
    parser.add_argument("--lock-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    _need(args.python.is_file(), f"Python executable is missing: {args.python}")
    if args.run_worker:
        _need(args.lock_dir is not None, "--run-worker requires the launch-created lock")
        run_worker(repo_root=args.repo_root, outer_date=args.outer_date, run_root=args.run_root,
                   python=args.python, lock_dir=args.lock_dir)
        return
    plan = build_plan(repo_root=args.repo_root, outer_date=args.outer_date, run_root=args.run_root, python=args.python)
    if args.launch:
        launch(plan=plan, repo_root=args.repo_root.resolve(), outer_date=args.outer_date,
               run_root=args.run_root.resolve(), python=args.python.resolve(), predecessor_tmx=args.predecessor_tmux)
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
