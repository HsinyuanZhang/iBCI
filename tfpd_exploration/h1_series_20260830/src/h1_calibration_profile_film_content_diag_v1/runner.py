"""Diagnostic runner: re-execute the sealed V2 LODO runner under a FiLM
context mode (full / empty / rowshuffle) and apply the preregistered
decision law.

Nothing sealed is modified: the patch rebinds two module-namespace names
(`film_identity`, `build_film`) inside
`h1_calibration_profile_film_v1.evaluate` for the lifetime of one mode-run,
then restores them.  Every internal anchor of the sealed runner (zero-init
bitwise-equality, frozen-substrate hashes, LP-R3 r2 anchors) stays active.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from . import plan

REPO_ROOT = Path(__file__).resolve().parents[4]
LOCAL_SRC = REPO_ROOT / "tfpd_exploration/h1_series_20260830/src"
SPINT_ROOT = REPO_ROOT / "SPINT-main"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_paths() -> None:
    for value in (str(REPO_ROOT), str(SPINT_ROOT), str(LOCAL_SRC)):
        if value not in sys.path:
            sys.path.insert(0, value)


def preflight(gpu_index: int) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_index):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must pin exactly the target GPU")
    for variable in THREAD_VARIABLES:
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable} must be 1")
    gpu_line = subprocess.check_output(
        ["nvidia-smi", "-i", str(int(gpu_index)), "--query-gpu=uuid,name,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).strip()
    uuid, name, free_mib, utilization = (part.strip() for part in gpu_line.split(","))
    if int(free_mib) < 4096:
        raise RuntimeError(f"GPU{gpu_index} has less than 4 GiB free")
    if int(utilization) > 10:
        raise RuntimeError(f"GPU{gpu_index} is not idle enough")
    apps = subprocess.check_output(
        ["nvidia-smi", "-i", str(int(gpu_index)), "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if apps:
        raise RuntimeError(f"GPU{gpu_index} has foreign compute processes: {apps!r}")
    return {"gpu_index": int(gpu_index), "gpu_uuid": uuid, "gpu_name": name,
            "gpu_free_mib_at_preflight": int(free_mib),
            "gpu_utilization_at_preflight": int(utilization),
            "gpu_foreign_compute_processes_at_preflight": apps}


def _gate(per_date_gains: list[float]) -> dict[str, Any]:
    """V2 gate semantics (v1 plan _gate) applied to a per-date delta series."""
    mean = sum(per_date_gains) / len(per_date_gains)
    nonnegative = sum(value >= 0.0 for value in per_date_gains)
    worst = min(per_date_gains)
    passed = (mean >= plan.GATE_MEAN and nonnegative >= plan.GATE_NONNEGATIVE_DATES
              and worst >= plan.GATE_WORST)
    return {"mean": mean, "nonnegative_dates": nonnegative, "worst": worst, "pass": passed,
            "per_date": list(per_date_gains)}


def make_patch(mode: str) -> dict[str, Any]:
    """Rebind film_identity/build_film in the v1 runner namespace for `mode`."""
    import numpy as np
    import torch

    import h1_calibration_profile_film_v1.evaluate as v1_eval
    from src.h1_m4_cce_contract import state_hash

    if mode not in plan.MODES:
        raise RuntimeError(f"unknown mode {mode!r}")
    original_film_identity = v1_eval.film_identity
    original_build_film = v1_eval.build_film
    permutation = np.random.default_rng(plan.PERM_SEED).permutation(plan.PERM_UNITS)
    perm_sha = hashlib.sha256(json.dumps([int(v) for v in permutation]).encode("ascii")).hexdigest()
    perm_tensor = torch.as_tensor(permutation, dtype=torch.long)
    state: dict[str, Any] = {"perm_checked": False, "init_film_state_shas": []}

    def transform(profile: Any) -> Any:
        if mode == "full":
            return profile
        if mode == "empty":
            return torch.zeros_like(profile)
        out = profile[:, perm_tensor.to(profile.device), :]
        if not state["perm_checked"]:
            rows_original = torch.sort(profile.detach().reshape(-1, profile.shape[-1]), dim=0).values
            rows_moved = torch.sort(out.detach().reshape(-1, profile.shape[-1]), dim=0).values
            if not torch.allclose(rows_original, rows_moved, atol=0.0, rtol=0.0):
                raise RuntimeError("rowshuffle transform is not a true row permutation")
            state["perm_checked"] = True
        return out

    def film_identity_wrapped(net: Any, activity: Any, carrier: Any, profile: Any, film: Any, *, late: bool) -> Any:
        if late:
            return original_film_identity(net, activity, carrier, profile, film, late=True)
        return original_film_identity(net, activity, carrier, transform(profile), film, late=False)

    def build_film_wrapped() -> Any:
        module = original_build_film()
        state["init_film_state_shas"].append(state_hash(module.state_dict()))
        return module

    v1_eval.film_identity = film_identity_wrapped
    v1_eval.build_film = build_film_wrapped

    def restore() -> None:
        v1_eval.film_identity = original_film_identity
        v1_eval.build_film = original_build_film

    return {"restore": restore, "perm_sha256": perm_sha, "state": state,
            "patched": {"h1_calibration_profile_film_v1.evaluate.film_identity": mode,
                        "h1_calibration_profile_film_v1.evaluate.build_film": "receipted"}}


def sealed_v2_reference() -> dict[str, Any]:
    """Load and SHA-verify the sealed V2 score; extract paired references."""
    root = REPO_ROOT / plan.V2_ROOT_RELATIVE
    score_path = root / "score.json"
    if sha256_file(score_path) != plan.V2_SCORE_SHA256:
        raise RuntimeError("sealed V2 score SHA drift")
    score = json.loads(score_path.read_text(encoding="utf-8"))
    folds = score["folds"]
    return {
        "outer_dates": [fold["outer_date"] for fold in folds],
        "ep_zero_per_date": [fold["scores"]["EP-ZERO"] for fold in folds],
        "ep_film_per_date": [fold["scores"]["EP-FILM"] for fold in folds],
        "ep_gain_per_date": [fold["early_film_gain"] for fold in folds],
        "lp_gain_per_date": [fold["late_film_gain"] for fold in folds],
        "ep_gain_mean": sum(fold["early_film_gain"] for fold in folds) / len(folds),
        "lp_gain_mean": sum(fold["late_film_gain"] for fold in folds) / len(folds),
        "first_fold_film_state_sha256": {
            arm: folds[0]["checkpoints"][arm]["film_state_sha256"] for arm in ("EP-FILM", "LP-FILM")
        },
    }


def run_mode(mode: str, repo_root: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    install_paths()
    import torch  # noqa: F401  (device sanity happens inside run_v1)

    from h1_calibration_profile_film_v1.evaluate import run as run_v1
    from h1_calibration_profile_film_v2.plan import ANCHOR_R2_TOLERANCE

    receipt_root = REPO_ROOT / plan.RESULT_ROOT_RELATIVE / mode
    if receipt_root.exists():
        raise RuntimeError(f"mode receipt root is not fresh: {receipt_root}")
    receipt_root.mkdir(parents=True)
    patch = make_patch(mode)
    attempt = {
        "schema": f"{plan.SCHEMA}_attempt", "mode": mode,
        "status": "ATTEMPT_MODE_RUN_OVER_SEALED_V2_RUNNER",
        "created_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "perm_sha256": patch["perm_sha256"], "perm_seed": plan.PERM_SEED,
        "patched": patch["patched"], "anchor_r2_tolerance": ANCHOR_R2_TOLERANCE,
        "require_historical_prediction_sha": False,
        "formal_heldout_opened": False, "evalai_opened": False,
    }
    (receipt_root / "attempt.json").write_text(json.dumps(attempt, indent=1, sort_keys=True), encoding="utf-8")
    try:
        result = run_v1(repo_root, device=device, receipt_root=receipt_root,
                        require_historical_prediction_sha=False,
                        anchor_r2_tolerance=ANCHOR_R2_TOLERANCE)
    finally:
        patch["restore"]()
    # build_film call sites per fold: one train-time build in train_pair
    # (right after torch.manual_seed(SEED) -> deterministic, must equal the
    # sealed V2 initial ff2263b5...) and two scratch builds inside _load_film
    # (immediately overwritten by strict loads; values unconstrained).
    from collections import Counter

    init_counter = dict(Counter(patch["state"]["init_film_state_shas"]))
    folds = result["folds"]
    if init_counter.get(plan.V2_INITIAL_FILM_STATE_SHA256, 0) < len(folds):
        raise RuntimeError(
            f"training-time initial film state drift under mode {mode}: {init_counter}")
    ep_gains = [fold["early_film_gain"] for fold in folds]
    lp_gains = [fold["late_film_gain"] for fold in folds]
    summary = {
        "schema": f"{plan.SCHEMA}_mode_result", "mode": mode,
        "ep_gain_per_date": ep_gains,
        "ep_gain_mean": sum(ep_gains) / len(ep_gains),
        "ep_gate": _gate(ep_gains),
        "lp_gain_per_date": lp_gains,
        "lp_gain_mean": sum(lp_gains) / len(lp_gains),
        "ep_zero_per_date": [fold["scores"]["EP-ZERO"] for fold in folds],
        "ep_film_per_date": [fold["scores"]["EP-FILM"] for fold in folds],
        "first_fold_film_state_sha256": {
            arm: folds[0]["checkpoints"][arm]["film_state_sha256"] for arm in ("EP-FILM", "LP-FILM")
        },
        "init_film_state_sha_counter": init_counter,
        "perm_sha256": patch["perm_sha256"],
        "decision_classification": result["decision"]["classification"],
    }
    (receipt_root / "mode_result.json").write_text(json.dumps(
        {"schema": summary["schema"], **summary, "run_status": result["status"]},
        indent=1, sort_keys=True), encoding="utf-8")
    return summary


def decide(summaries: dict[str, dict[str, Any]], sealed: dict[str, Any]) -> dict[str, Any]:
    full, empty, rowshuffle = summaries["full"], summaries["empty"], summaries["rowshuffle"]
    drift = abs(full["ep_gain_mean"] - plan.SEALED_V2_OOF_MEAN_GAIN)
    drift_ok = drift <= plan.DRIFT_TOLERANCE
    lp_canary = {
        f"{mode}": {
            "mean_abs_delta": abs(summaries[mode]["lp_gain_mean"] - sealed["lp_gain_mean"]),
            "max_per_date_abs_delta": max(abs(a - b) for a, b in
                                          zip(summaries[mode]["lp_gain_per_date"], sealed["lp_gain_per_date"])),
        }
        for mode in plan.MODES
    }
    lp_canary_ok = all(
        entry["mean_abs_delta"] <= plan.LP_CANARY_MEAN_TOLERANCE
        and entry["max_per_date_abs_delta"] <= plan.LP_CANARY_MAX_TOLERANCE
        for entry in lp_canary.values()
    )
    empty_mean = empty["ep_gain_mean"]
    empty_gate_pass = bool(empty["ep_gate"]["pass"])
    if not drift_ok or not lp_canary_ok:
        branch = "STOP_ENVIRONMENT_DRIFT"
    elif empty_gate_pass:
        branch = "A_BUDGET_ADAPTATION"
    elif empty_mean < plan.CONTENT_FLOOR:
        branch = "B_PROFILE_CONTENT"
    else:
        branch = "C_MIXED"
    ratio = (empty_mean / full["ep_gain_mean"]) if abs(full["ep_gain_mean"]) > 1e-12 else None
    return {
        "schema": f"{plan.SCHEMA}_decision", "branch": branch, "drift_abs": drift,
        "drift_ok": drift_ok, "lp_canary": lp_canary, "lp_canary_ok": lp_canary_ok,
        "empty_mean": empty_mean, "empty_gate_pass": empty_gate_pass,
        "rowshuffle_mean": rowshuffle["ep_gain_mean"],
        "rowshuffle_gate": rowshuffle["ep_gate"],
        "full_mean": full["ep_gain_mean"], "empty_over_full_ratio": ratio,
        "narrative": {
            "A_BUDGET_ADAPTATION": "C3 permanently cancelled; H1 FiLM story becomes the zero-init adapter result",
            "B_PROFILE_CONTENT": "profile content is real; C3 may open and must add a C3-ZERO same-machine control",
            "C_MIXED": "both stories one notch weaker, proportional attribution by empty/full ratio",
            "STOP_ENVIRONMENT_DRIFT": "no interpretation; diagnose the environment first",
        }[branch],
    }


def publish_terminal(result_root: Path, summaries: dict[str, dict[str, Any]],
                     sealed: dict[str, Any], decision: dict[str, Any],
                     preflight_info: dict[str, Any]) -> str:
    terminal = {
        "schema": f"{plan.SCHEMA}_terminal",
        "status": f"COMPLETE_H1_FILM_CONTENT_DIAGNOSTIC_{decision['branch']}",
        "modes": {mode: {key: summaries[mode][key] for key in
                         ("ep_gain_per_date", "ep_gain_mean", "ep_gate", "lp_gain_mean",
                          "first_fold_film_state_sha256", "perm_sha256")}
                  for mode in plan.MODES},
        "sealed_v2": sealed,
        "decision": decision,
        "preflight": preflight_info,
        "formal_heldout_opened": False, "evalai_opened": False,
        "target_optimizer_steps": 0, "target_backward_steps": 0, "target_model_updates": 0,
    }
    body = json.dumps(terminal, indent=1, sort_keys=True)
    path = result_root / "terminal.json"
    path.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    side = path.with_name(path.name + ".sha256")
    side.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.chmod(path, 0o444)
    os.chmod(side, 0o444)
    return digest


__all__ = ("decide", "install_paths", "make_patch", "preflight", "publish_terminal",
           "run_mode", "sealed_v2_reference", "sha256_file")
