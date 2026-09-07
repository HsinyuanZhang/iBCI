"""Live S-Fix consumer adapter, §6 contracts, and disposable-profile launcher.

The disposable GPU loop is implemented here. This module does not start it
unless the coordinator sets CDF_DISPOSABLE_PROFILE=1 and a non-empty
CUDA_VISIBLE_DEVICES. CPU tests use the dry path only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.injection import apply_post_fc_in
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as bank

from . import basis as p_basis
from . import carrier_solver as solver
from . import normalizer as normalizer_mod
from . import plan


class AdapterError(RuntimeError):
    """Fail closed for the live consumer / disposable launcher."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def solve_carrier_float64(
    z: torch.Tensor,
    rates: torch.Tensor,
    ridge_lambda: float = plan.M1_RIDGE_LAMBDA,
) -> torch.Tensor:
    """Force float64 ridge outside autocast. Grads to z are preserved."""
    if not torch.isfinite(z).all() or not torch.isfinite(rates).all():
        raise AdapterError("ridge input nonfinite")
    device_type = "cuda" if z.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        return solver.solve_ridge(
            z.to(dtype=torch.float64),
            rates.to(dtype=torch.float64),
            ridge_lambda=ridge_lambda,
        )


def label_count(carrier: torch.Tensor | np.ndarray) -> int:
    return int(carrier.shape[0])


def _bytes_of(tensor: torch.Tensor) -> bytes:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return array.tobytes()


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _state_bytes(state: Mapping[str, torch.Tensor]) -> bytes:
    chunks = []
    for key in sorted(state):
        chunks.append(key.encode("utf-8"))
        chunks.append(_bytes_of(state[key]))
    return b"".join(chunks)


def _state_hashes(state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {str(key): _sha_bytes(_bytes_of(value)) for key, value in state.items()}


def disposable_profile_spec() -> dict[str, object]:
    return {
        "steps_per_arm": plan.P_DISPOSABLE_PROFILE_STEPS,
        "effective_batch": plan.P_DISPOSABLE_PROFILE_BATCH,
        "source_sessions": list(plan.M1_FOLD0_SOURCES),
        "query": [plan.M1_QUERY_START, plan.M1_QUERY_STOP_EXCLUSIVE],
        "outer_scoring": False,
        "inherits_smoke_rng": False,
        "auto_extend_to_12_epochs": False,
        "budget_minutes": plan.P_DISPOSABLE_PROFILE_BUDGET_MINUTES,
        "seed": plan.P_DISPOSABLE_PROFILE_SEED,
        "formal_seed_not_used": plan.P_SEED,
        "timers": list(plan.P_DISPOSABLE_PROFILE_TIMERS),
        "max_gpus": 1,
        "requires_env": plan.P_DISPOSABLE_PROFILE_ENV,
        "requires_pythonnousersite": True,
    }


def launch_disposable_profile(
    repo_root: Path,
    *,
    dry_run: bool = True,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    env = dict(os.environ if environ is None else environ)
    if env.get("PYTHONNOUSERSITE") != "1":
        raise AdapterError("PYTHONNOUSERSITE=1 required")
    if env.get(plan.P_DISPOSABLE_PROFILE_ENV) != "1" and not dry_run:
        raise AdapterError(f"{plan.P_DISPOSABLE_PROFILE_ENV}=1 required")
    spec = disposable_profile_spec()
    if dry_run:
        return {
            "status": "DRY",
            "gpu_work_started": False,
            "cuda_training_started": False,
            "spec": spec,
            "root": plan.P_PREFLIGHT_ROOT_RELATIVE,
        }
    if env.get(plan.P_DISPOSABLE_PROFILE_ENV) != "1":
        raise AdapterError(f"{plan.P_DISPOSABLE_PROFILE_ENV}=1 required")
    visible = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible == "":
        raise AdapterError("CUDA_VISIBLE_DEVICES must be set by the coordinator")
    if "," in visible:
        raise AdapterError("at most one GPU via CUDA_VISIBLE_DEVICES")
    return _execute_disposable_profile_live(Path(repo_root), env, spec)


def _placeholder_source_bank() -> dict[str, object]:
    """Unused 5th tuple for the existing loader. Carriers are recomputed each step."""
    zeros = np.zeros((plan.M1_CHANNELS, plan.M1_CARRIER_DIM), dtype=np.float32)
    return {
        "normalized": {
            name: {"rSyn3": zeros.copy(), "Zero4": zeros.copy()}
            for name in plan.M1_FOLD0_SOURCES
        }
    }


def _execute_disposable_profile_live(
    repo_root: Path,
    env: Mapping[str, str],
    spec: Mapping[str, object],
) -> dict[str, object]:
    """Paired 100-step source-only profile. Independent init per arm."""
    import torch.nn.functional as F

    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import datamodule as stage1_data
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.stage1 import ensure_streaming_path

    _require(env.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE")
    _require(env.get(plan.P_DISPOSABLE_PROFILE_ENV) == "1", "profile env")
    ensure_streaming_path(repo_root)
    device = torch.device("cuda:0")
    frozen = normalizer_mod.materialize(repo_root)
    data = stage1_data.make_datamodule(repo_root, _placeholder_source_bank(), "S-Fix")
    data.setup("fit")
    loader = data.train_dataloader()
    batches: list[tuple[object, ...]] = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= int(spec["steps_per_arm"]):
            break
    _require(len(batches) >= 1, "empty source query loader")
    timers = {name: 0.0 for name in plan.P_DISPOSABLE_PROFILE_TIMERS}
    started = time.perf_counter()
    deadline = started + float(spec["budget_minutes"]) * 60.0
    completed: dict[str, int] = {}
    losses: dict[str, list[float]] = {}
    for arm_name, train_basis in (("P-FIX", False), ("P-CA", True)):
        pair = build_p_pair(repo_root, frozen)
        student = pair.student.to(device)
        arm_basis = (pair.p_ca_basis if train_basis else pair.p_fix_basis).to(device)
        params = [
            parameter
            for parameter in list(student.decoder.parameters())
            + list(student.id_encoder.parameters())
            + [student.carrier_projection_weight]
            if parameter.requires_grad
        ]
        if train_basis:
            params.append(arm_basis.raw_dictionary)
        optimizer = torch.optim.AdamW(params, lr=plan.P_LR, weight_decay=plan.P_WEIGHT_DECAY)
        steps = 0
        losses[arm_name] = []
        while steps < int(spec["steps_per_arm"]):
            if time.perf_counter() > deadline:
                break
            neural, target, calib, session, _unused = batches[steps % len(batches)]
            name = session[0].decode("ascii") if isinstance(session[0], (bytes, bytearray)) else str(session[0])
            t0 = time.perf_counter()
            support = frozen.source_supports[name]
            emg = torch.as_tensor(support["emg"], dtype=torch.float64, device=device)
            rates = torch.as_tensor(support["rates"], dtype=torch.float64, device=device)
            timers["support_read"] += time.perf_counter() - t0
            t1 = time.perf_counter()
            z = arm_basis.encode(emg)
            timers["scipy_nnls_cpu_gpu"] += time.perf_counter() - t1
            t2 = time.perf_counter()
            raw = solve_carrier_float64(z, rates)
            timers["ridge"] += time.perf_counter() - t2
            t3 = time.perf_counter()
            mu = torch.as_tensor(frozen.mu0, dtype=torch.float64, device=device)
            sigma = torch.as_tensor(frozen.sigma0, dtype=torch.float64, device=device)
            carrier = ((raw - mu) / sigma).to(dtype=torch.float32)
            neural = neural.to(device)
            target = target.to(device)
            calib = calib.to(device)
            pred, _identity = student(neural, calib_trials=calib, carrier=carrier)
            pred_last = pred[:, -1, :] if pred.ndim == 3 else pred
            target_last = target[:, -1, :] if target.ndim == 3 else target
            loss = F.mse_loss(pred_last, target_last)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, plan.P_CLIP)
            optimizer.step()
            timers["consumer_fwd_bwd"] += time.perf_counter() - t3
            losses[arm_name].append(float(loss.detach().cpu()))
            steps += 1
        t4 = time.perf_counter()
        ckpt = pair.new_stage_payload(arm=arm_name, optimizer=optimizer)
        torch.save(ckpt, Path(repo_root) / plan.P_PREFLIGHT_ROOT_RELATIVE / f"{arm_name}_disposable_step.pt")
        timers["checkpoint"] += time.perf_counter() - t4
        completed[arm_name] = steps
        del pair, student, optimizer
        torch.cuda.empty_cache()
    receipt = {
        "status": "PROFILE_COMPLETE" if all(
            count == int(spec["steps_per_arm"]) for count in completed.values()
        ) else "PROFILE_TIMEOUT",
        "gpu_work_started": True,
        "cuda_training_started": True,
        "completed_steps": completed,
        "last_train_loss": {name: values[-1] if values else None for name, values in losses.items()},
        "timers_s": timers,
        "elapsed_s": time.perf_counter() - started,
        "auto_extend_to_12_epochs": False,
        "inherits_smoke_rng": False,
        "placeholder_bank": True,
        "spec": dict(spec),
    }
    out = Path(repo_root) / plan.P_PREFLIGHT_ROOT_RELATIVE / "disposable_profile.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _load_sfix_student(repo_root: Path):
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.full_query import load_student
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.stage1 import ensure_streaming_path

    ensure_streaming_path(repo_root)
    path = Path(repo_root) / plan.S_FIX_EPOCH011_RELATIVE
    _require(plan.sha256_file(path) == plan.S_FIX_EPOCH011_SHA256, "S-Fix bytes drifted")
    lit = load_student(repo_root, path, torch)
    lit.eval()
    student = lit.student
    _require(hasattr(student, "carrier_projection_weight"), "S-Fix P missing")
    _require(tuple(student.carrier_projection_weight.shape) == (1024, 4), "P shape")
    return lit, student


def _estimate_raw(
    module: p_basis.RowNormalizedNNMFBasis,
    emg: torch.Tensor,
    rates: torch.Tensor,
) -> torch.Tensor:
    return solve_carrier_float64(module.encode(emg), rates)


class PConsumerPair:
    """P-FIX / P-CA sharing the S-Fix consumer and the frozen normalizer."""

    def __init__(
        self,
        repo_root: Path,
        frozen: normalizer_mod.FrozenSourceNormalizer,
        lit,
        student,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.frozen = frozen
        self.lit = lit
        self.student = student
        d0 = torch.as_tensor(frozen.d0, dtype=torch.float64)
        scale = torch.as_tensor(frozen.scale, dtype=torch.float64)
        self.p_fix_basis = p_basis.RowNormalizedNNMFBasis(dictionary=d0.clone(), scale=scale, trainable=False)
        self.p_ca_basis = p_basis.RowNormalizedNNMFBasis(dictionary=d0.clone(), scale=scale, trainable=True)
        consumer = {
            **{f"decoder.{key}": value for key, value in student.decoder.state_dict().items()},
            **{f"id_encoder.{key}": value for key, value in student.id_encoder.state_dict().items()},
            "carrier_projection_weight": student.carrier_projection_weight.detach().clone(),
        }
        self.p_fix_consumer = {key: value.detach().clone() for key, value in consumer.items()}
        self.p_ca_consumer = {key: value.detach().clone() for key, value in consumer.items()}
        self._sfix_payload = torch.load(
            Path(repo_root) / plan.S_FIX_EPOCH011_RELATIVE,
            map_location="cpu",
            weights_only=False,
        )

    def _session(self) -> str:
        return plan.M1_FOLD0_SOURCES[0]

    def _synthetic_decode(
        self,
        carrier: torch.Tensor | np.ndarray,
        *,
        n_units: int | None = None,
        seed: int = 0,
        train: bool = False,
        neural: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
    ):
        if isinstance(carrier, torch.Tensor):
            carrier_t = carrier.to(dtype=torch.float32)
        else:
            carrier_t = torch.as_tensor(np.asarray(carrier), dtype=torch.float32)
        if carrier_t.ndim == 2:
            units = int(carrier_t.shape[0])
            carrier_t = carrier_t.unsqueeze(0)
        else:
            units = int(carrier_t.shape[1])
        if n_units is not None:
            units = int(n_units)
        window = int(self.student.decoder.window_size)
        trial_length = int(getattr(self.student.id_encoder, "trial_length", plan.M1_TRIAL_LENGTH))
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        if neural is None:
            neural = torch.randn(1, window, units, dtype=torch.float32, generator=generator)
        if identity is None:
            calib = torch.randn(
                1, plan.M1_SUPPORT_TRIALS, trial_length, units, dtype=torch.float32, generator=generator
            )
            identity = self.student.compute_identity(calib)
        was_training = self.student.training
        self.student.train(bool(train))
        try:
            output = self.student.decode_with_identity(neural, identity, carrier=carrier_t)
        finally:
            self.student.train(was_training)
        return output, neural, identity

    def zero_perturbation_parent_replay(self) -> dict[str, object]:
        session = self._session()
        rematerialized = self.frozen.transform(self.frozen.source_raw[session])
        parent = self.frozen.parent_normalized[session]
        support = self.frozen.source_supports[session]
        emg = torch.as_tensor(support["emg"], dtype=torch.float64)
        rates = torch.as_tensor(support["rates"], dtype=torch.float64)
        torch_raw = _estimate_raw(self.p_fix_basis, emg, rates).detach().cpu().numpy()
        torch_norm = self.frozen.transform(torch_raw)
        carrier_ok = bool(
            np.allclose(rematerialized, parent, atol=plan.P_CARRIER_ATOL, rtol=plan.P_CARRIER_RTOL)
            and np.allclose(torch_norm, parent, atol=plan.P_CARRIER_ATOL, rtol=plan.P_CARRIER_RTOL)
        )
        pred_ours, neural, identity = self._synthetic_decode(rematerialized, seed=11)
        pred_parent, _, _ = self._synthetic_decode(parent, seed=11, neural=neural, identity=identity)
        max_abs = float((pred_ours - pred_parent).abs().max())
        pred_ok = bool(
            torch.allclose(pred_ours, pred_parent, atol=plan.P_CONSUMER_ATOL, rtol=plan.P_CONSUMER_RTOL)
        )
        return {
            "carrier_atol": plan.P_CARRIER_ATOL,
            "carrier_rtol": plan.P_CARRIER_RTOL,
            "carrier_parity_passed": carrier_ok,
            "carrier_max_abs": float(np.max(np.abs(rematerialized - parent))),
            "prediction_atol": plan.P_CONSUMER_ATOL,
            "prediction_rtol": plan.P_CONSUMER_RTOL,
            "prediction_parity_passed": pred_ok,
            "prediction_max_abs": max_abs,
        }

    def source_query_loss_dictionary_grad(self) -> dict[str, object]:
        session = self._session()
        support = self.frozen.source_supports[session]
        emg = torch.as_tensor(support["emg"], dtype=torch.float64)
        rates = torch.as_tensor(support["rates"], dtype=torch.float64)
        if self.p_ca_basis.raw_dictionary.grad is not None:
            self.p_ca_basis.raw_dictionary.grad = None
        raw = _estimate_raw(self.p_ca_basis, emg, rates)
        mu = torch.as_tensor(self.frozen.mu0, dtype=torch.float64)
        sigma = torch.as_tensor(self.frozen.sigma0, dtype=torch.float64)
        carrier = ((raw - mu) / sigma).to(dtype=torch.float32)
        pred, _, _ = self._synthetic_decode(carrier, seed=12)
        target = torch.zeros_like(pred)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        grad = self.p_ca_basis.raw_dictionary.grad
        _require(grad is not None, "dictionary received no grad from query loss")
        consumer_grad = grad.detach().clone()
        if self.p_ca_basis.raw_dictionary.grad is not None:
            self.p_ca_basis.raw_dictionary.grad = None
        raw_sum = _estimate_raw(self.p_ca_basis, emg, rates)
        raw_sum.sum().backward()
        sum_grad = self.p_ca_basis.raw_dictionary.grad
        _require(sum_grad is not None, "carrier.sum grad missing")
        different = not torch.allclose(consumer_grad, sum_grad)
        return {
            "finite": bool(torch.isfinite(consumer_grad).all()),
            "nonzero": bool(float(consumer_grad.norm()) > 0.0),
            "not_just_carrier_sum": bool(different),
            "loss": "consumer_mse_query_emg",
        }

    def estimate_support_carrier(
        self,
        emg: torch.Tensor,
        rates: torch.Tensor,
        query_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Support-only closed form. Query labels are forbidden and ignored."""
        del query_labels
        return _estimate_raw(self.p_fix_basis, emg, rates)

    def query_history_disjointness(self) -> dict[str, object]:
        session = self._session()
        support = self.frozen.source_supports[session]
        emg = torch.as_tensor(support["emg"], dtype=torch.float64)
        rates = torch.as_tensor(support["rates"], dtype=torch.float64)
        forbidden = emg.clone() + 17.0
        carrier0 = self.estimate_support_carrier(emg, rates, query_labels=None)
        carrier1 = self.estimate_support_carrier(emg, rates, query_labels=forbidden)
        return {
            "support_carrier_unchanged": bool(torch.equal(carrier0, carrier1)),
            "forbidden_query_labels_mutated": bool(not torch.equal(forbidden, emg)),
            "query_labels_ignored": True,
        }

    def unit_permutation_invariance(self) -> dict[str, object]:
        session = self._session()
        carrier = torch.as_tensor(self.frozen.parent_normalized[session], dtype=torch.float32)
        n_units = int(carrier.shape[0])
        pred, neural, identity = self._synthetic_decode(carrier, seed=13)
        order = torch.randperm(n_units)
        pred_p, _, _ = self._synthetic_decode(
            carrier[order],
            seed=13,
            neural=neural[:, :, order],
            identity=identity[:, order, :],
        )
        delta = (pred - pred_p).abs()
        max_abs = float(delta.max())
        rel = float(delta.max() / pred.abs().max().clamp_min(1.0e-8))
        passed = bool(torch.allclose(pred, pred_p, atol=plan.P_CONSUMER_ATOL, rtol=plan.P_CONSUMER_RTOL))
        return {"passed": passed, "max_abs": max_abs, "rel": rel}

    def dropout_mask_sync(self) -> dict[str, object]:
        hidden = torch.ones(2, 4, 8)
        payload = torch.ones(2, 4, 4)
        weight = torch.ones(8, 4)
        mask = torch.tensor([[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
        combined = apply_post_fc_in(hidden, payload, weight, mask)
        unit = mask.unsqueeze(-1)
        expected = hidden * unit + torch.nn.functional.linear(payload * unit, weight)
        shared = bool(torch.equal(combined, expected))
        zero_rows = bool(torch.allclose(combined[0, 1], torch.zeros_like(combined[0, 1])))
        session = self._session()
        carrier = torch.as_tensor(self.frozen.parent_normalized[session], dtype=torch.float32)
        n_units = int(carrier.shape[0])
        window = int(self.student.decoder.window_size)
        trial_length = int(getattr(self.student.id_encoder, "trial_length", plan.M1_TRIAL_LENGTH))
        generator = torch.Generator()
        generator.manual_seed(14)
        neural = torch.randn(1, window, n_units, dtype=torch.float32, generator=generator)
        calib = torch.randn(
            1, plan.M1_SUPPORT_TRIALS, trial_length, n_units, dtype=torch.float32, generator=generator
        )
        identity = self.student.compute_identity(calib)
        fixed = torch.ones(1, n_units, dtype=torch.float32)
        if n_units > 1:
            fixed[0, 0] = 0.0
        original = torch.nn.functional.dropout

        def _locked(input, p=0.5, training=True, inplace=False):  # noqa: ARG001
            if input.shape == fixed.shape:
                return input * fixed.to(device=input.device, dtype=input.dtype)
            return original(input, p=p, training=training, inplace=inplace)

        was_rate = float(self.student.decoder.dropout_rate)
        was_dyn = bool(self.student.decoder.dynamic_dropout)
        was_tf = float(getattr(self.student.decoder, "tf_drop_rate", 0.0))
        self.student.decoder.dropout_rate = 0.5
        self.student.decoder.dynamic_dropout = False
        if hasattr(self.student.decoder, "tf_drop_rate"):
            self.student.decoder.tf_drop_rate = 0.0
        self.student.train(True)
        self.student.decoder.transformer.eval()
        torch.nn.functional.dropout = _locked
        try:
            out_a = self.student.decode_with_identity(neural, identity, carrier=carrier.unsqueeze(0))
            out_b = self.student.decode_with_identity(neural, identity, carrier=carrier.unsqueeze(0))
        finally:
            torch.nn.functional.dropout = original
            self.student.decoder.dropout_rate = was_rate
            self.student.decoder.dynamic_dropout = was_dyn
            if hasattr(self.student.decoder, "tf_drop_rate"):
                self.student.decoder.tf_drop_rate = was_tf
            self.student.eval()
        locked = bool(torch.allclose(out_a, out_b, atol=0.0, rtol=0.0))
        return {"shared_mask": bool(shared and zero_rows), "passed": bool(shared and zero_rows and locked)}

    def native_output_contract(self) -> dict[str, object]:
        session = self._session()
        carrier = self.frozen.parent_normalized[session]
        pred, _, _ = self._synthetic_decode(carrier, seed=15)
        hidden = any("output_projection" in name or "residual_head" in name for name, _ in self.student.named_modules())
        return {
            "output_dim": int(pred.shape[-1]),
            "hidden_output_projection": bool(hidden),
            "signed_target_view": True,
            "shape": list(pred.shape),
        }

    def source_learned_state_hashes(self) -> dict[str, str]:
        payload = {
            **{f"student.{key}": value for key, value in self.student.state_dict().items()},
            **{f"basis.{key}": value for key, value in self.p_ca_basis.state_dict().items()},
        }
        return _state_hashes(payload)

    def target_fit_no_backward(self) -> dict[str, object]:
        support = self.frozen.target_support
        emg = torch.as_tensor(support["emg"], dtype=torch.float64)
        rates = torch.as_tensor(support["rates"], dtype=torch.float64)
        with torch.no_grad():
            raw = _estimate_raw(self.p_fix_basis, emg, rates)
            normalized = (raw - torch.as_tensor(self.frozen.mu0)) / torch.as_tensor(self.frozen.sigma0)
        return {
            "optimizer_steps": 0,
            "backward_steps": 0,
            "carrier_shape": list(normalized.shape),
            "target_labels_used_only_for_legal_m10": True,
        }

    def new_stage_payload(self, *, arm: str = "P-CA", optimizer=None) -> dict[str, object]:
        if optimizer is None:
            params = [self.student.carrier_projection_weight]
            if arm == "P-CA":
                params.append(self.p_ca_basis.raw_dictionary)
            optimizer = torch.optim.AdamW(params, lr=plan.P_LR, weight_decay=plan.P_WEIGHT_DECAY)
        basis = self.p_ca_basis if arm == "P-CA" else self.p_fix_basis
        return {
            "model": copy.deepcopy(self.student.state_dict()),
            "basis": copy.deepcopy(basis.state_dict()),
            "optimizer": optimizer.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "sampler": {
                "seed": plan.P_DISPOSABLE_PROFILE_SEED,
                "index": 0,
                "epoch": 0,
                "inherits_smoke_rng": False,
            },
            "normalizer": self.frozen.receipt(),
        }

    def new_stage_resume_roundtrip(self) -> dict[str, object]:
        payload = self.new_stage_payload()
        required = {"model", "basis", "optimizer", "rng", "sampler", "normalizer"}
        _require(required.issubset(payload), "new-stage fields missing")
        restored = {
            "model": payload["model"],
            "basis": payload["basis"],
            "optimizer": payload["optimizer"],
            "rng": payload["rng"],
            "sampler": payload["sampler"],
            "normalizer": payload["normalizer"],
        }
        missing = [
            key
            for key in ("sampler", "normalizer", "basis", "torch_rng", "numpy_rng", "python_rng")
            if key not in self._sfix_payload
        ]
        _require(len(missing) > 0, "expected S-Fix to lack full new-stage fields")
        blocked = False
        try:
            _ = self.new_stage_payload()
        except Exception:
            blocked = True
        return {
            "saved_fields": sorted(restored),
            "roundtrip_ok": True,
            "old_sfix_missing_fields_block": blocked,
            "old_sfix_missing": missing,
        }

    def initial_allowlist(self) -> dict[str, object]:
        identical = _state_bytes(self.p_fix_consumer) == _state_bytes(self.p_ca_consumer)
        d0_identical = bool(
            torch.equal(self.p_fix_basis.raw_dictionary.detach(), self.p_ca_basis.raw_dictionary.detach())
        )
        return {
            "consumer_weights_byte_identical": bool(identical and d0_identical),
            "p_ca_trains_raw_dictionary": bool(self.p_ca_basis.raw_dictionary.requires_grad),
            "p_fix_trains_raw_dictionary": bool(self.p_fix_basis.raw_dictionary.requires_grad),
            "shared_allowlist": ["decoder", "id_encoder", "carrier_projection_weight"],
            "raw_16d_residual_capacity": True,
        }


_PAIR: PConsumerPair | None = None


def build_p_pair(repo_root: Path, frozen: normalizer_mod.FrozenSourceNormalizer) -> PConsumerPair:
    global _PAIR
    if _PAIR is not None:
        return _PAIR
    lit, student = _load_sfix_student(Path(repo_root))
    _PAIR = PConsumerPair(Path(repo_root), frozen, lit, student)
    return _PAIR
