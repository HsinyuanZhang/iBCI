#!/usr/bin/env python3
"""Source-only post-training gauge-aware audit for a H1-EST4 B-C/L-C pair.

EST4 has exact latent-basis and carrier-coordinate gauges.  In particular,
raw ``pcs``, ``U``, ``mu``, ``lambda`` and ``tau2`` values are not a valid
mechanistic attribution after an L-C checkpoint has co-trained the consumer.
This CPU-only audit instead binds each terminal checkpoint to (i) rank,
condition number and normalized spectrum, (ii) frozen-to-trained projection
subspaces, (iii) scale/rotation-invariant scalar ratios, and (iv) deterministic
source-carrier and carrier-to-consumer-affine RMS values over every immutable
M=4 source-cache entry in manifest order.

It opens source recordings only after validating the immutable pair preflight
and checkpoints.  It has no target, minival, formal-test, EvalAI, Trainer,
CUDA, subprocess, optimizer, or prediction-score path.  Therefore its output
is a forward-behaviour/provenance binding, never B-C-versus-L-C efficacy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_est4_pair_preflight import (
    MODEL_KWARGS,
    PAIR_ARMS,
    PREFLIGHT_SCHEMA,
    PREFLIGHT_STATUS,
)
from src.data.h1_carrierid_date_lodo_est4 import H1CarrierIdDateLodoEst4DataModule
from src.h1_m4_cce_contract import canonical_sha256, sha256_file, write_immutable_json
from src.models.components.h1_carrierid_est4_spint import H1CarrierIdEst4Spint
from src.models.h1_carrierid_date_lodo_est4_module import EST4_CHECKPOINT_SCHEMA


AUDIT_SCHEMA = "h1_carrierid_date_lodo_est4_pair_posttraining_gauge_audit_v1"
AUDIT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_EST4_PAIR_SOURCE_FORWARD_GAUGE_AUDIT_NOT_PREDICTIVE"


class Est4PairGaugeAuditError(ValueError):
    """A source-only B-C/L-C checkpoint/audit binding is incomplete."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4PairGaugeAuditError(message)


def _immutable_json(path: Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    candidate = path.resolve()
    _need(candidate.is_file() and not candidate.is_symlink() and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          f"immutable mode-0444 receipt required: {candidate}")
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Est4PairGaugeAuditError(f"invalid immutable JSON receipt: {candidate}") from error
    _need(isinstance(body, dict) and body.get("schema") == schema and body.get("status") == status,
          f"receipt schema/status drift: {candidate}")
    return candidate, body, sha256_file(candidate)


def _array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    _need(np.isfinite(result).all(), "audit tensor is nonfinite")
    return result


def _matrix_metrics(value: torch.Tensor | np.ndarray, *, expected_rank: int, label: str) -> dict[str, Any]:
    matrix = _array(value)
    _need(matrix.ndim == 2, f"{label} must be a matrix")
    singular = np.linalg.svd(matrix, compute_uv=False)
    _need(singular.size >= expected_rank and singular[0] > 0.0, f"{label} has zero spectral norm")
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * float(singular[0])
    rank = int(np.count_nonzero(singular > tolerance))
    _need(rank == expected_rank, f"{label} rank={rank}, expected {expected_rank}")
    retained = singular[:rank]
    return {
        "shape": list(matrix.shape), "rank": rank, "rank_tolerance": float(tolerance),
        "condition_number": float(retained[0] / retained[-1]),
        # This spectrum, unlike its absolute magnitude, is invariant to the
        # positive scale in the exact EST4 gauge.
        "normalized_singular_values": [float(item / retained[0]) for item in retained],
    }


def _row_basis(value: torch.Tensor | np.ndarray, *, rank: int, label: str) -> np.ndarray:
    matrix = _array(value)
    _need(matrix.ndim == 2, f"{label} must be a matrix")
    basis, _ = np.linalg.qr(matrix.T, mode="reduced")
    _need(basis.shape[1] >= rank, f"{label} has insufficient QR rank")
    return basis[:, :rank]


def _subspace_metrics(
    frozen: torch.Tensor | np.ndarray, trained: torch.Tensor | np.ndarray, *, rank: int, label: str,
) -> dict[str, Any]:
    first, second = _row_basis(frozen, rank=rank, label=f"{label} frozen"), _row_basis(trained, rank=rank, label=f"{label} trained")
    cosine = np.clip(np.linalg.svd(first.T @ second, compute_uv=False), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return {
        "principal_cosines": [float(item) for item in cosine],
        "principal_angles_degrees": [float(item) for item in angle],
        "max_principal_angle_degrees": float(np.max(angle)),
        "rms_principal_angle_degrees": float(math.sqrt(float(np.mean(np.square(angle))))),
        "projector_frobenius_distance": float(np.linalg.norm(first @ first.T - second @ second.T, ord="fro")),
    }


def gauge_aware_estimator_metrics(
    *, trained_pcs: torch.Tensor | np.ndarray, trained_u: torch.Tensor | np.ndarray,
    trained_mu: torch.Tensor | np.ndarray, trained_lambda: float, trained_tau2: float,
    frozen_pcs: torch.Tensor | np.ndarray, frozen_u: torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    """Return quantities invariant to EST4's declared basis/scale gauges.

    ``pcs -> alpha Q pcs, lambda -> alpha² lambda`` preserves its latent
    ridge coordinates; ``U -> beta U R, mu -> beta mu R, tau2 -> beta² tau2``
    is absorbed by the first consumer carrier affine map.  No raw basis or
    scale is therefore emitted as an attribution result.
    """

    pcs, u, mu = _array(trained_pcs), _array(trained_u), _array(trained_mu)
    base_pcs, base_u = _array(frozen_pcs), _array(frozen_u)
    _need(pcs.shape == (16, 176) and base_pcs.shape == (16, 176), "pcs shape drift")
    _need(u.shape == (7, 4) and base_u.shape == (7, 4) and mu.shape == (4,), "U/mu shape drift")
    _need(math.isfinite(float(trained_lambda)) and float(trained_lambda) > 0.0,
          "trained ridge lambda is invalid")
    _need(math.isfinite(float(trained_tau2)) and float(trained_tau2) > 0.0,
          "trained shrinkage tau2 is invalid")
    pcs_frob_sq = float(np.square(np.linalg.norm(pcs, ord="fro")))
    u_frob_sq = float(np.square(np.linalg.norm(u, ord="fro")))
    _need(pcs_frob_sq > 0.0 and u_frob_sq > 0.0, "trained projection scale is zero")
    return {
        "pcs": _matrix_metrics(pcs, expected_rank=16, label="trained pcs"),
        "U": _matrix_metrics(u, expected_rank=4, label="trained U"),
        "frozen_to_trained_subspace": {
            "pcs_row_space": _subspace_metrics(base_pcs, pcs, rank=16, label="pcs"),
            # The coordinate gauge acts on U's columns, so U's column space
            # is represented as the row space of U.T.
            "U_column_space": _subspace_metrics(base_u.T, u.T, rank=4, label="U"),
        },
        "gauge_invariant_scalar_ratios": {
            "lambda_over_pcs_frobenius_squared": float(float(trained_lambda) / pcs_frob_sq),
            "tau2_over_U_frobenius_squared": float(float(trained_tau2) / u_frob_sq),
            "mu_l2_over_U_frobenius": float(np.linalg.norm(mu) / math.sqrt(u_frob_sq)),
        },
        "interpretation": (
            "rank, condition, normalized spectrum, subspaces, and listed ratios are gauge-aware; "
            "raw pcs/U/mu/lambda/tau2 coordinates are not a mechanism attribution"
        ),
    }


def _checkpoint(
    path: Path, *, arm: str, pair_preflight: Mapping[str, Any], pair_preflight_sha: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    candidate = path.resolve()
    _need(candidate.is_file() and not candidate.is_symlink(), f"checkpoint missing: {candidate}")
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and int(payload.get("epoch", -1)) == 49 and int(payload.get("global_step", 0)) > 0,
          f"{arm} is not fixed terminal e49")
    metadata = payload.get("h1_carrierid_date_lodo_est4")
    _need(isinstance(metadata, Mapping), f"{arm} lacks EST4 checkpoint metadata")
    aggregate = pair_preflight.get("five_date_aggregate")
    frozen = pair_preflight.get("frozen_estimator_initialization")
    source = pair_preflight.get("source_binding")
    _need(isinstance(aggregate, Mapping) and isinstance(frozen, Mapping) and isinstance(source, Mapping),
          "pair preflight lacks aggregate/frozen/source binding")
    required = {
        "schema": EST4_CHECKPOINT_SCHEMA, "arm": arm,
        "outer_date": pair_preflight.get("outer_date"), "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
        "target_optimizer_steps": 0, "target_backward_steps": 0, "checkpoint_warm_start": False,
        "target_evaluator_status": "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED",
        "est4_preflight_sha256": pair_preflight_sha,
        "five_date_aggregate_sha256": aggregate.get("sha256"),
        "phase2_base_source_binding_sha256": canonical_sha256(source),
        "frozen_plan_sha256": frozen.get("sha256"),
    }
    for key, value in required.items():
        _need(metadata.get(key) == value, f"{arm} checkpoint metadata drift: {key}")
    fresh = pair_preflight.get("fresh_models", {}).get(arm)
    _need(isinstance(fresh, Mapping)
          and metadata.get("component_initial_state_sha256") == fresh.get("initial_state_sha256")
          and metadata.get("shared_backbone_initial_state_sha256") == fresh.get("shared_backbone_initial_state_sha256"),
          f"{arm} checkpoint does not bind the pair preflight's fresh model initialization")
    mode = "baseline" if arm == "B-C" else "learned"
    added = 0 if mode == "baseline" else 2_850
    _need(metadata.get("estimator_mode") == mode and metadata.get("estimator_added_learned_parameters") == added,
          f"{arm} estimator accounting drift")
    state = payload.get("state_dict")
    _need(isinstance(state, Mapping), f"{arm} checkpoint state_dict is missing")
    net_state = {str(name)[4:]: value for name, value in state.items() if str(name).startswith("net.")}
    _need(net_state and len(net_state) == len(state), f"{arm} checkpoint contains non-net state")
    _need(all(isinstance(value, torch.Tensor) for value in net_state.values()), f"{arm} non-tensor net checkpoint value")
    config = candidate.parent.parent.parent / ".hydra/config.yaml"
    _need(config.is_file() and sha256_file(config) == metadata.get("config_sha256"),
          f"{arm} saved Hydra config SHA mismatch")
    text = config.read_text(encoding="utf-8")
    for fragment in (
        "train: true", "test: false", "ckpt_path: null", "seed: 42", f"arm: {arm}",
        "H1CarrierIdDateLodoEst4DataModule", "H1CarrierIdDateLodoEst4LitModule",
        "max_epochs: 50", "min_epochs: 50",
    ):
        _need(fragment in text, f"{arm} saved config lacks {fragment!r}")
    return {
        "path": str(candidate), "sha256": sha256_file(candidate), "metadata": dict(metadata),
    }, net_state


def _model_from_checkpoint(
    *, arm: str, state: Mapping[str, torch.Tensor], frozen_plan: Path,
) -> H1CarrierIdEst4Spint:
    kwargs: dict[str, Any] = {**MODEL_KWARGS, "estimator_mode": "baseline" if arm == "B-C" else "learned"}
    if arm == "L-C":
        # The persisted source-normalizer buffer is loaded from the checkpoint;
        # this seed value cannot become a target-derived quantity.
        kwargs.update(frozen_plan_path=str(frozen_plan), source_normalizer=1.0)
    model = H1CarrierIdEst4Spint(**kwargs).cpu().eval()
    missing, unexpected = model.load_state_dict(dict(state), strict=False)
    _need(not missing and not unexpected, f"{arm} checkpoint/model topology drift: missing={missing}, unexpected={unexpected}")
    return model


def _rms(accumulator: float, elements: int) -> float:
    _need(elements > 0 and accumulator >= 0.0 and math.isfinite(accumulator), "invalid RMS accumulator")
    return float(math.sqrt(accumulator / elements))


def _source_forward_rms(
    *, dataset: Any, baseline: H1CarrierIdEst4Spint, learned: H1CarrierIdEst4Spint,
) -> dict[str, Any]:
    """Forward exactly all immutable M=4 cache entries in manifest order."""

    _need(learned.estimator is not None, "L-C model has no estimator")
    sums = {"b_carrier": 0.0, "l_carrier": 0.0, "b_affine": 0.0, "l_affine": 0.0}
    counts = {key: 0 for key in sums}
    b_weight = baseline.carrier_post_pool[0].weight[:, baseline.carrier_hidden_dim:].detach().cpu()
    l_weight = learned.carrier_post_pool[0].weight[:, learned.carrier_hidden_dim:].detach().cpu()
    _need(tuple(b_weight.shape) == tuple(l_weight.shape) == (32, 4), "carrier consumer affine topology drift")
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for entry in dataset.cache.entries:
            session, start = str(entry.session_name), int(entry.start_index)
            b_carrier = torch.as_tensor(dataset._baseline_carriers[(session, start)], dtype=torch.float32).unsqueeze(0)
            rates, labels, mask = dataset._ridge_inputs(session=session, calibration_start=start)
            l_carrier = learned.estimator(
                torch.as_tensor(rates, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(labels, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0),
            ).cpu()
            b_affine = torch.matmul(b_carrier, b_weight.T)
            l_affine = torch.matmul(l_carrier, l_weight.T)
            for key, tensor in (("b_carrier", b_carrier), ("l_carrier", l_carrier),
                                ("b_affine", b_affine), ("l_affine", l_affine)):
                value = tensor.detach().cpu().double()
                _need(torch.isfinite(value).all().item(), f"{key} source forward is nonfinite")
                sums[key] += float(torch.sum(torch.square(value)).item())
                counts[key] += int(value.numel())
            rows.append({"session": session, "start_index": start, "trial_values": [float(v) for v in entry.trial_values]})
    _need(rows and len(rows) == len(dataset.cache.entries), "source cache enumeration drift")
    return {
        "enumeration": "all immutable M=4 source-cache entries in manifest order",
        "entry_count": len(rows),
        "entries_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        # Carrier RMS by itself is coordinate-scale dependent; it catches
        # numerical collapse/explosion but is not called a mechanism result.
        "carrier_output_rms": {"B-C": _rms(sums["b_carrier"], counts["b_carrier"]),
                               "L-C": _rms(sums["l_carrier"], counts["l_carrier"])},
        # This is invariant under the paired carrier-coordinate/consumer
        # reparameterization and is the appropriate consumption-scale guard.
        "carrier_to_first_consumer_affine_rms": {
            "B-C": _rms(sums["b_affine"], counts["b_affine"]),
            "L-C": _rms(sums["l_affine"], counts["l_affine"]),
        },
        "interpretation": (
            "no decoding R2/loss is computed; carrier-output RMS is a deterministic numerical binding, "
            "whereas carrier-to-first-consumer-affine RMS is gauge-invariant under the declared output gauge"
        ),
    }


def run(
    *, data_dir: Path, phase1_preflight: Path, pair_preflight_path: Path,
    checkpoints: Mapping[str, Path], output: Path,
) -> dict[str, Any]:
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""),
          "EST4 pair post-training audit requires CUDA_VISIBLE_DEVICES unset/empty")
    _need(tuple(checkpoints) == PAIR_ARMS, "audit requires exactly B-C then L-C checkpoints")
    _need(not output.exists() and not os.path.lexists(str(output)), "refusing to overwrite EST4 pair audit")
    preflight_path, pair, pair_sha = _immutable_json(pair_preflight_path, schema=PREFLIGHT_SCHEMA, status=PREFLIGHT_STATUS)
    frozen = pair.get("frozen_estimator_initialization")
    source = pair.get("source_binding")
    _need(isinstance(frozen, Mapping) and isinstance(source, Mapping), "pair preflight frozen/source binding missing")
    frozen_plan = Path(str(frozen.get("path", ""))).resolve()
    _need(frozen_plan.is_file() and sha256_file(frozen_plan) == frozen.get("sha256"), "frozen plan hash drift")
    checked: dict[str, dict[str, Any]] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for arm in PAIR_ARMS:
        checked[arm], states[arm] = _checkpoint(checkpoints[arm], arm=arm, pair_preflight=pair, pair_preflight_sha=pair_sha)
    b_meta, l_meta = checked["B-C"]["metadata"], checked["L-C"]["metadata"]
    _need(b_meta.get("shared_backbone_initial_state_sha256") == l_meta.get("shared_backbone_initial_state_sha256"),
          "B-C/L-C did not start from the same common consumer backbone")
    baseline = _model_from_checkpoint(arm="B-C", state=states["B-C"], frozen_plan=frozen_plan)
    learned = _model_from_checkpoint(arm="L-C", state=states["L-C"], frozen_plan=frozen_plan)
    _need(learned.estimator is not None, "loaded L-C checkpoint lacks estimator")
    with np.load(frozen_plan, allow_pickle=False) as values:
        frozen_pcs = np.asarray(values["pcs"], dtype=np.float64)[:16]
        frozen_u = np.asarray(values["U"], dtype=np.float64)
    estimator = learned.estimator
    estimator_metrics = gauge_aware_estimator_metrics(
        trained_pcs=estimator.pcs, trained_u=estimator.U, trained_mu=estimator.mu,
        trained_lambda=float(torch.exp(estimator.log_lambda).detach().cpu().item()),
        trained_tau2=float(torch.exp(estimator.log_tau2).detach().cpu().item()),
        frozen_pcs=frozen_pcs, frozen_u=frozen_u,
    )
    # Source I/O begins only here, after every immutable provenance and
    # checkpoint check above.  The DataModule itself has no target loader.
    dm = H1CarrierIdDateLodoEst4DataModule(
        task="h1", data_dir=str(data_dir), phase1_preflight_path=str(phase1_preflight),
        outer_date=str(pair["outer_date"]), est4_arm="L-C", est4_preflight_path=str(preflight_path),
        five_date_aggregate_path=str(pair["five_date_aggregate"]["path"]), frozen_plan_path=str(frozen_plan),
        batch_size=32, window_size=700, calibration_n_trials=4, max_trial_length=1024,
        seed=42, fixed_epochs=50, num_workers=0, pin_memory=False,
    )
    dm.setup("fit")
    _need(canonical_sha256(dm.binding.manifest()) == canonical_sha256(source), "actual source binding drifted from pair preflight")
    source_rms = _source_forward_rms(dataset=dm.train_dataset, baseline=baseline, learned=learned)
    payload = {
        "schema": AUDIT_SCHEMA, "status": AUDIT_STATUS,
        "mode": "explicit_cpu_source_forward_audit_no_trainer_no_gpu_no_target_no_prediction_metric",
        "pair_preflight": {"path": str(preflight_path), "sha256": pair_sha, "outer_date": pair["outer_date"]},
        "checkpoints": checked,
        "source_binding_sha256": canonical_sha256(source),
        "frozen_plan": {"path": str(frozen_plan), "sha256": sha256_file(frozen_plan)},
        "gauge_aware_l_c_estimator": estimator_metrics,
        "deterministic_source_forward": source_rms,
        "scope": {
            "source_recordings_opened": len(dm.binding.source_sessions),
            "target_recordings_opened": 0, "target_bytes_read": 0,
            "minival_opened": 0, "formal_test_opened": 0, "evalai_opened": 0,
            "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
            "prediction_r2_or_training_loss_reported": False,
        },
        "conclusion_boundary": (
            "This receipt validates checkpoint provenance and source forward behaviour only. "
            "It cannot establish B-C-versus-L-C prediction gain and cannot attribute any later gain to raw estimator parameters."
        ),
        "code_sha256": {
            "audit": sha256_file(Path(__file__).resolve()),
            "data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_est4.py"),
            "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_est4_module.py"),
            "component": sha256_file(ROOT / "src/models/components/h1_carrierid_est4_spint.py"),
            "pair_preflight": sha256_file(ROOT / "scripts/h1_carrierid_date_lodo_est4_pair_preflight.py"),
        },
    }
    written, digest = write_immutable_json(output, payload)
    return {"status": AUDIT_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    # Gate source I/O before argparse validates any path arguments; a bare
    # call must never progress to a potentially data-bearing invocation.
    if "--run-source-forward-audit" not in sys.argv:
        raise SystemExit("refusing implicit source/NWB access; pass --run-source-forward-audit explicitly")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-source-forward-audit", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--phase1-preflight", type=Path,
                        default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json")
    parser.add_argument("--pair-preflight", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", metavar="ARM=PATH", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints: dict[str, Path] = {}
    for item in args.checkpoint:
        arm, separator, path = item.partition("=")
        if not separator or arm in checkpoints:
            raise SystemExit("each --checkpoint must be one unique ARM=PATH")
        checkpoints[arm] = Path(path)
    print(json.dumps(run(
        data_dir=args.data_dir, phase1_preflight=args.phase1_preflight,
        pair_preflight_path=args.pair_preflight, checkpoints=checkpoints, output=args.output,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
