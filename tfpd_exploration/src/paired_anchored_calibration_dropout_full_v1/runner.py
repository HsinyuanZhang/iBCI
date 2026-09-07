"""Streaming full-epoch evidence; every update delegates to reviewed PACD V1."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from . import plan

# Tests may inject a tiny reviewed-operator stand-in.  Production resolves the
# sole scientific operator lazily, after the immutable attempt is present.
paired_train_step = None


class FullInvariantError(RuntimeError):
    pass


def _req(condition: bool, message: str) -> None:
    if not condition:
        raise FullInvariantError(message)


def _chain(previous: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (previous + json.dumps(row, sort_keys=True, separators=(",", ":"))).encode()
    ).hexdigest()


@dataclass
class _Stats:
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def add(self, value: float) -> None:
        value = float(value)
        _req(value == value and value not in (float("inf"), float("-inf")), "non-finite statistic")
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def receipt(self) -> dict[str, float]:
        _req(self.count > 0, "empty statistic")
        return {"mean": self.total / self.count, "min": self.minimum, "max": self.maximum}


def _stat_set() -> dict[str, _Stats]:
    return {name: _Stats() for name in (
        "anchor", "short", "combined", "anchor_encoder", "anchor_decoder",
        "combined_encoder", "combined_decoder",
    )}


def _record_statistic(stats: dict[str, _Stats], row: dict[str, Any], key: str, row_key: str) -> None:
    # The V1 evidence seam is additive.  A source checkout predating the seam
    # cannot silently serve full training: the missing datum is a hard error.
    _req(row_key in row, f"reviewed PACD step omitted {row_key}")
    stats[key].add(float(row[row_key]))


def run_epoch(*, model, optimizer, loader, arm, device, torch, arm_common,
              encoder_parameters, decoder_parameters, epoch: int, pad_value: float,
              steps_per_epoch: int = plan.STEPS_PER_EPOCH, probe_sink=None,
              zero_encoder_policy: str = "reject") -> dict[str, Any]:
    """Run one exact epoch with O(sentinels), not O(steps), receipt memory."""
    global paired_train_step
    if paired_train_step is None:
        from src.paired_anchored_calibration_dropout_v1.core import paired_train_step as operator
        paired_train_step = operator
    _req(steps_per_epoch == plan.STEPS_PER_EPOCH, "steps/epoch drift")
    _req(epoch in range(plan.EPOCHS), "epoch index drift")

    stats = _stat_set()
    mask_chain = rng_chain = prefix_chain = ""
    sentinels: list[dict[str, Any]] = []
    p0_prediction_mismatches = p0_identity_mismatches = 0
    rng_violations = prefix_mutations = finite_violations = 0
    zero_encoder_steps = zero_decoder_steps = 0
    accepted_zero_encoder_steps = 0
    positive_encoder_steps = positive_decoder_steps = 0
    valid_min, valid_max = float("inf"), float("-inf")
    p_min, p_max = float("inf"), float("-inf")
    materialized_min, materialized_max = float("inf"), float("-inf")
    lazy_min, lazy_max = float("inf"), float("-inf")

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    count = 0
    for step, batch in enumerate(loader):
        if step >= steps_per_epoch:
            break
        neural, behavior, calibration, sessions, side = batch[:5]
        if step == 0 and probe_sink is not None:
            # A detached exact-shape source probe for the terminal SWA proof.
            # It never enters a receipt and never changes sampler/update order.
            probe_sink((neural.detach().clone(), calibration.detach().clone(), side.detach().clone()))
        lr = float(arm_common.lr_at_step(epoch * steps_per_epoch + step, plan.EPOCHS, steps_per_epoch))
        _req(len(optimizer.param_groups) == 1, "Adam parameter-group drift")
        optimizer.param_groups[0]["lr"] = lr
        evidence = paired_train_step(
            model=model, optimizer=optimizer, neural=neural.to(device), behavior=behavior.to(device),
            calibration=calibration.to(device), side_features=side.to(device), short_m=arm["short_m"],
            pad_value=pad_value, encoder_parameters=encoder_parameters, decoder_parameters=decoder_parameters,
            collect_branch_evidence=step in plan.SENTINELS,
            zero_encoder_policy=zero_encoder_policy,
        )
        _req(evidence.get("optimizer_steps") == 1, "paired operator update-count drift")
        _record_statistic(stats, evidence, "anchor", "loss_anchor")
        _record_statistic(stats, evidence, "short", "loss_short")
        _record_statistic(stats, evidence, "combined", "loss_combined")
        for key, row_key in (
            ("anchor_encoder", "anchor_encoder_grad_norm"), ("anchor_decoder", "anchor_decoder_grad_norm"),
            ("combined_encoder", "combined_encoder_grad_norm"), ("combined_decoder", "combined_decoder_grad_norm"),
        ):
            _record_statistic(stats, evidence, key, row_key)
        combined_encoder = float(evidence["combined_encoder_grad_norm"])
        combined_decoder = float(evidence["combined_decoder_grad_norm"])
        if combined_encoder == 0.0:
            zero_encoder_steps += 1
            if zero_encoder_policy != "reject":
                _req(
                    evidence.get("combined_encoder_zero_accepted") is True
                    and evidence.get("combined_encoder_zero_reason") == "all_units_dropped_valid_zero"
                    and evidence.get("paired_unit_mask_empty") is True,
                    "unjustified zero encoder gradient",
                )
                accepted_zero_encoder_steps += 1
        else:
            positive_encoder_steps += 1
        if combined_decoder == 0.0:
            zero_decoder_steps += 1
        else:
            positive_decoder_steps += 1
        valid = int(evidence["valid_bins"]); valid_min = min(valid_min, valid); valid_max = max(valid_max, valid)
        probability = float(evidence["dropout"]["p"]); p_min = min(p_min, probability); p_max = max(p_max, probability)
        mask_chain = _chain(mask_chain, evidence["dropout"])
        rng_chain = _chain(rng_chain, {"before": evidence["rng_before_sha256"], "after": evidence["rng_after_pair_sha256"]})
        prefix_chain = _chain(prefix_chain, {"full": evidence["calibration_anchor_sha256"], "short": evidence["calibration_short_sha256"]})
        rng_violations += int(not evidence.get("rng_short_transition_equal")) + int(not evidence.get("rng_pair_transition_equal"))
        prefix_mutations += int(evidence.get("calibration_full_sha256") != evidence.get("calibration_full_after_sha256", evidence.get("calibration_full_sha256")))
        p0_prediction_mismatches += int(not evidence["prediction_pair_equal"])
        p0_identity_mismatches += int(not evidence["identity_pair_equal"])
        finite = evidence["parameter_finiteness"]
        materialized = int(finite["materialized"]); lazy = int(finite["skipped_uninitialized_lazy"])
        materialized_min = min(materialized_min, materialized); materialized_max = max(materialized_max, materialized)
        lazy_min = min(lazy_min, lazy); lazy_max = max(lazy_max, lazy)
        finite_violations += int((materialized, lazy) != (29, 2))
        if step in plan.SENTINELS:
            _req("short_encoder_grad_norm" in evidence and "short_decoder_grad_norm" in evidence,
                 "sentinel branch evidence absent")
            sentinels.append({"step": step, "lr": lr, "evidence": evidence})
        count += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    _req(count == steps_per_epoch, "short epoch")
    _req((p0_prediction_mismatches, p0_identity_mismatches) == (0, 0) if arm["name"] == "p0" else True, "P0 equality")
    _req(rng_violations == prefix_mutations == finite_violations == 0, "full step invariant drift")
    if zero_encoder_policy != "reject":
        _req(zero_decoder_steps == 0, "zero decoder-gradient coverage")
        _req(positive_encoder_steps > 0, "all-epoch zero encoder-gradient coverage")
        _req(positive_decoder_steps > 0, "all-epoch zero decoder-gradient coverage")
    _req([item["step"] for item in sentinels] == [x for x in plan.SENTINELS if x < steps_per_epoch], "sentinel topology")
    receipt = {
        "epoch": epoch, "optimizer_steps": count, "cumulative_optimizer_steps": (epoch + 1) * steps_per_epoch,
        "loss": {name: stats[name].receipt() for name in ("anchor", "short", "combined")},
        "grad_norm": {name: stats[name].receipt() for name in (
            "anchor_encoder", "anchor_decoder", "combined_encoder", "combined_decoder")},
        "branch_gradient_cosine": {"collected_at_sentinels_only": True,
                                   "sentinel_count": len(sentinels),
                                   "encoder": [row["evidence"]["encoder_branch_gradient_cosine"] for row in sentinels],
                                   "decoder": [row["evidence"]["decoder_branch_gradient_cosine"] for row in sentinels]},
        "valid_bins": {"min": valid_min, "max": valid_max},
        "dropout": {"p_min": p_min, "p_max": p_max, "mask_chain": mask_chain},
        "rng_chain": rng_chain, "rng_violations": rng_violations, "prefix_chain": prefix_chain,
        "prefix_mutations": prefix_mutations, "p0_prediction_mismatches": p0_prediction_mismatches,
        "p0_identity_mismatches": p0_identity_mismatches,
        "parameter_finiteness": {"materialized_min": materialized_min, "materialized_max": materialized_max,
                                  "lazy_min": lazy_min, "lazy_max": lazy_max, "violations": finite_violations},
        "sentinels": sentinels, "wall_seconds": elapsed, "paired_steps_per_second": count / elapsed if elapsed else float("inf"),
    }
    if zero_encoder_policy != "reject":
        receipt["gradient_coverage"] = {
            "zero_encoder_steps": zero_encoder_steps,
            "accepted_zero_encoder_steps": accepted_zero_encoder_steps,
            "zero_encoder_reason": "all_units_dropped_valid_zero",
            "zero_decoder_steps": zero_decoder_steps,
            "positive_encoder_steps": positive_encoder_steps,
            "positive_decoder_steps": positive_decoder_steps,
        }
    return receipt
