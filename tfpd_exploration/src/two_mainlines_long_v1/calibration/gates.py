"""P1–P5 gate runner. Must fail on the old _PAIR / fake-resume path."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer as sealed_normalizer

from . import constants as C
from . import checkpoint as ckpt
from . import hashes
from . import loop
from . import optimizer as optmod
from . import parent_parity
from . import source_query
from .factory import build_fresh_arm
from .pfix_cache import build_pfix_cache


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def gate_p1(repo_root: Path, frozen: sealed_normalizer.FrozenSourceNormalizer, device: torch.device) -> dict[str, Any]:
    model_adapter._PAIR = None
    old_a = model_adapter.build_p_pair(repo_root, frozen)
    old_b = model_adapter.build_p_pair(repo_root, frozen)
    old_same_object = old_a is old_b
    before_ca = hashes.state_sha256(hashes.consumer_state(old_b.student))
    with torch.no_grad():
        old_a.student.carrier_projection_weight.add_(0.01)
    after_ca = hashes.state_sha256(hashes.consumer_state(old_b.student))
    old_mutates_unstarted = before_ca != after_ca
    model_adapter._PAIR = None
    del old_a, old_b
    gc.collect()

    fix = build_fresh_arm(repo_root, frozen, "P-FIX", device=device, seed=C.FORMAL_SEED)
    ca = build_fresh_arm(repo_root, frozen, "P-CA", device=device, seed=C.FORMAL_SEED)
    step0_equal = fix.consumer_sha256() == ca.consumer_sha256()
    before = ca.consumer_sha256()
    learned = hashes.learned_state(ca.student, ca.basis)
    with torch.no_grad():
        fix.student.carrier_projection_weight.add_(0.25)
    after = ca.consumer_sha256()
    after_learned = hashes.state_sha256(hashes.learned_state(ca.student, ca.basis))
    isolated = before == after and hashes.state_sha256(learned) == after_learned
    distinct_objects = fix.student is not ca.student and fix.basis is not ca.basis and fix.optimizer is not ca.optimizer
    passed = bool(distinct_objects and isolated and step0_equal and old_same_object and old_mutates_unstarted)
    del fix, ca
    gc.collect()
    return {
        "passed": passed,
        "old_path_same_object": old_same_object,
        "old_path_mutates_unstarted_other_arm": old_mutates_unstarted,
        "fresh_objects_distinct": distinct_objects,
        "step0_consumer_sha_equal": step0_equal,
        "training_pfix_leaves_unstarted_pca": isolated,
        "forbidden_factory": C.FORBIDDEN_OLD_FACTORY,
    }


def gate_p2(repo_root: Path, frozen: sealed_normalizer.FrozenSourceNormalizer, device: torch.device) -> dict[str, Any]:
    seed_receipt = optmod.seed_all(C.PROFILE_SEED, device=device)
    arm = build_fresh_arm(repo_root, frozen, "P-CA", device=device, seed=C.PROFILE_SEED)
    mode = arm.enter_train()
    old_lit, old_student = model_adapter._load_sfix_student(repo_root)
    old_eval = (not bool(old_lit.training)) and (not bool(old_student.training))
    old_dyn = bool(old_student.decoder.dynamic_dropout)
    old_tf = float(getattr(old_student.decoder, "tf_drop_rate", 0.0))
    del old_lit, old_student
    gc.collect()
    passed = bool(
        mode["student_training"]
        and mode["decoder_training"]
        and mode["dynamic_dropout"]
        and mode["basis_requires_grad"]
        and any(group["weight_decay"] == C.WEIGHT_DECAY for group in mode["adamw_groups"])
        and any(group["weight_decay"] == 0.0 for group in mode["adamw_groups"])
        and seed_receipt["python"]
        and seed_receipt["numpy"]
        and seed_receipt["torch"]
        and old_eval
    )
    del arm
    gc.collect()
    return {
        "passed": passed,
        "seed_receipt": seed_receipt,
        "train_mode": mode,
        "old_sfix_loader_leaves_eval": old_eval,
        "old_dynamic_dropout": old_dyn,
        "old_tf_drop_rate": old_tf,
    }


def gate_p3(repo_root: Path, frozen: sealed_normalizer.FrozenSourceNormalizer, device: torch.device) -> dict[str, Any]:
    data, loader = source_query.load_real_source_batches(repo_root, frozen, n_batches=3)
    if len(loader) < 3:
        raise RuntimeError("need at least 3 source batches for resume")
    arm = build_fresh_arm(repo_root, frozen, "P-CA", device=device, seed=C.PROFILE_SEED)
    cache = None
    rec0 = loop.one_update(arm, loader[0], cache=cache, steps_per_warmup=C.PROFILE_WARMUP_STEPS)
    path = C.OWNED_RESULT_ROOT / "gates" / "p3_resume.pt"
    ckpt.save_arm(arm, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    twin = build_fresh_arm(repo_root, frozen, "P-CA", device=device, seed=None)
    ckpt.load_into_arm(twin, path)
    ckpt._restore_rng(saved["rng"], arm.device)
    rec_a = loop.one_update(arm, loader[1], cache=cache, steps_per_warmup=C.PROFILE_WARMUP_STEPS)
    ckpt._restore_rng(saved["rng"], twin.device)
    rec_b = loop.one_update(twin, loader[1], cache=cache, steps_per_warmup=C.PROFILE_WARMUP_STEPS)
    loss_ok = abs(rec_a["loss"] - rec_b["loss"]) <= 1.0e-7
    lr_ok = rec_a["lr"] == rec_b["lr"]
    sha_ok = arm.learned_sha256() == twin.learned_sha256()
    old = model_adapter.PConsumerPair(repo_root, frozen, arm.lit, arm.student)
    old_payload = old.new_stage_payload()
    old_resume = old.new_stage_resume_roundtrip()
    old_empty = not bool(old_payload["optimizer"].get("state"))
    old_sampler_zero = int(old_payload["sampler"]["index"]) == 0
    old_no_cuda = "cuda" not in old_payload["rng"]
    old_no_arrays = "mu0" not in old_payload["normalizer"]
    adamw_ok = ckpt.adamw_nonempty(arm.optimizer)
    passed = bool(
        adamw_ok
        and loss_ok
        and lr_ok
        and sha_ok
        and old_empty
        and old_sampler_zero
        and old_no_arrays
        and old_resume.get("roundtrip_ok") is True
    )
    del data, arm, twin, old
    gc.collect()
    return {
        "passed": passed,
        "warmup_step_loss": rec0["loss"],
        "next_loss_uninterrupted": rec_a["loss"],
        "next_loss_restored": rec_b["loss"],
        "next_lr_match": lr_ok,
        "next_loss_match": loss_ok,
        "next_state_sha_match": sha_ok,
        "adamw_nonempty": adamw_ok,
        "old_resume_is_dict_copy": bool(old_resume.get("roundtrip_ok")),
        "old_adamw_empty": old_empty,
        "old_sampler_index_zero": old_sampler_zero,
        "old_rng_missing_cuda": old_no_cuda,
        "old_normalizer_receipt_only": old_no_arrays,
        "checkpoint": str(path),
    }


def gate_p4(repo_root: Path, frozen: sealed_normalizer.FrozenSourceNormalizer) -> dict[str, Any]:
    report = parent_parity.independent_parent_parity(repo_root, frozen)
    report["old_path_uses_new_raw_plus_old_mean"] = report["used_new_raw_plus_old_mean_as_parent"]
    return report


def gate_p5(repo_root: Path, frozen: sealed_normalizer.FrozenSourceNormalizer, device: torch.device) -> dict[str, Any]:
    data, batch = source_query.load_one_real_source_batch(repo_root, frozen)
    split = source_query.window_id_support_query_split(data, frozen)
    arm = build_fresh_arm(repo_root, frozen, "P-CA", device=device, seed=C.PROFILE_SEED)
    grad = source_query.real_source_query_dictionary_grad(arm, batch)
    target = source_query.target_fit_learned_state_unchanged(arm)
    old_pair = model_adapter.PConsumerPair(repo_root, frozen, arm.lit, arm.student)
    old_grad = old_pair.source_query_loss_dictionary_grad()
    old_split = old_pair.query_history_disjointness()
    old_target = old_pair.target_fit_no_backward()
    old_synthetic = old_grad.get("loss") == "consumer_mse_query_emg"
    old_ignores_labels = bool(old_split.get("query_labels_ignored"))
    old_no_hash = "before_sha256" not in old_target
    passed = bool(split["passed"] and grad["passed"] and target["passed"] and old_synthetic and old_ignores_labels and old_no_hash)
    del data, arm, old_pair
    gc.collect()
    return {
        "passed": passed,
        "window_id_split": split,
        "real_query_grad": {k: v for k, v in grad.items() if k != "passed"} | {"passed": grad["passed"]},
        "target_hash": target,
        "old_grad_uses_zero_target": old_synthetic,
        "old_split_ignores_query_labels_arg": old_ignores_labels,
        "old_target_fit_omits_state_hash": old_no_hash,
    }


def run_all_gates(*, device: str | torch.device = "cpu") -> dict[str, Any]:
    repo = C.REPO_ROOT
    device = torch.device(device)
    frozen = sealed_normalizer.materialize(repo)
    receipts: dict[str, Any] = {}
    out = C.OWNED_RESULT_ROOT / "gates" / "p_gates.json"
    for name, runner in (
        ("P4", lambda: gate_p4(repo, frozen)),
        ("P1", lambda: gate_p1(repo, frozen, device)),
        ("P2", lambda: gate_p2(repo, frozen, device)),
        ("P3", lambda: gate_p3(repo, frozen, device)),
        ("P5", lambda: gate_p5(repo, frozen, device)),
    ):
        receipts[name] = runner()
        gc.collect()
        _write(
            out,
            {
                "schema": "two_mainlines_p_gates_v1",
                "partial": True,
                "gates": receipts,
            },
        )
    all_pass = all(bool(item["passed"]) for item in receipts.values())
    summary = {
        "schema": "two_mainlines_p_gates_v1",
        "all_pass": all_pass,
        "gates": receipts,
        "formal_training_authorized": all_pass,
        "red_stop": (not receipts["P4"]["passed"]) or (not all_pass and receipts["P4"].get("red_stop")),
    }
    out = C.OWNED_RESULT_ROOT / "gates" / "p_gates.json"
    _write(out, summary)
    _write(C.SLOT_ROOT / "p_gates.json", summary)
    return summary
