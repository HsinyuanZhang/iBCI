#!/usr/bin/env python3
"""Forward-only cross-session carrier transfer control for Context Full fold-0.

Scores each target recording with the correctly fitted carrier from the other
recording (same checkpoint, query windows, identity, and normalizer). No training.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset
from pynwb import NWBHDF5IO

from src.data.h1_context_event_carrier import (
    H1ContextStrictTargetDataset,
    build_context_target_dataset,
    H1ContextEventDataModule,
)
from src.data.h1_context_event_source_snapshot import load_snapshot
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1_M4_FOLD0_TARGET,
    H1PilotRecord,
    WINDOW,
)
from src.h1_m4_eb_normalized_v2_contract import (
    assert_state_immutable,
    sha256_file,
    state_hash,
    write_immutable_json,
)
from src.models.h1_sparse_event_module import CHECKPOINT_SCHEMA
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1


SCHEMA = "h1_cross_session_carrier_transfer_fold0_v1"
SEALED_FULL_R2 = 0.516518
SEALED_ROW_R2 = 0.499152
SEALED_ZERO_R2 = 0.484059
EXPECTED_QUERY_SHA = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
EXPECTED_QUERY_COUNT = 8965
FULL_INTEGRITY_TOLERANCE = 1.0e-6
DEFAULT_SOURCE_SNAPSHOT_RECEIPT = (
    PROJECT_ROOT / "pilot_artifacts/h1_context_event_carrier/source_snapshot/H1_CONTEXT_SER_Q4_FOLD0_SOURCE_v3.json"
)
DEFAULT_FULL_CHECKPOINT = (
    PROJECT_ROOT / "logs/h1_context_event_carrier_m4_full_s42_v2/checkpoints/fixed_epoch50/epoch_049.ckpt"
)
DEFAULT_FULL_CONFIG = PROJECT_ROOT / "logs/h1_context_event_carrier_m4_full_s42_v2/.hydra/config.yaml"
ARMS = ("full", "cross_session", "row", "zero")
CARRIER_DIM = 5


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def cross_session_transfer_map(recordings: Sequence[str]) -> dict[str, str]:
    """Return a derangement: each recording receives the next recording's carrier donor."""

    names = tuple(recordings)
    _need(len(names) >= 2, "cross-session transfer requires at least two target recordings")
    mapping = {names[index]: names[(index + 1) % len(names)] for index in range(len(names))}
    _need(all(name != donor for name, donor in mapping.items()), "transfer map must have no fixed points")
    _need(set(mapping.keys()) == set(names) and set(mapping.values()) == set(names), "transfer map must be a permutation")
    return mapping


def _unit_id_vector(path: Path) -> np.ndarray:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        _need(nwb.units is not None, f"{path}: units table missing")
        return np.asarray(nwb.units.id[:], dtype=np.int64)


def verify_same_array_channel_contract(records: Mapping[str, H1PilotRecord]) -> dict[str, Any]:
    """Fail closed unless all recordings share subject, channel count, and unit ordering."""

    names = tuple(records)
    _need(len(names) >= 2, "channel contract requires at least two recordings")
    for name in names:
        count = int(records[name].num_neurons)
        _need(count == EXPECTED_NEURONS, f"{name}: expected {EXPECTED_NEURONS} channels, got {count}")
    subjects: list[str] = []
    unit_ids: list[np.ndarray] = []
    for name in names:
        record = records[name]
        with NWBHDF5IO(str(record.path), "r", load_namespaces=True) as io:
            nwb = io.read()
            subjects.append(str(nwb.subject.subject_id))
        unit_ids.append(_unit_id_vector(record.path))
    reference_subject = subjects[0]
    reference_ids = unit_ids[0]
    _need(all(subject == reference_subject for subject in subjects), "target recordings must share one subject")
    for index, name in enumerate(names):
        _need(np.array_equal(unit_ids[index], reference_ids), f"{name}: unit/channel ordering differs from reference")
    return {
        "same_subject": True,
        "subject_id": reference_subject,
        "channel_count": EXPECTED_NEURONS,
        "carrier_shape": [EXPECTED_NEURONS, CARRIER_DIM],
        "recordings": list(names),
        "unit_id_sha256": event_v1.array_sha256(reference_ids),
        "unit_id_vector": reference_ids.tolist(),
        "transfer_kind": "same_array_same_subject_row_aligned",
    }


def build_cross_session_carriers(
    target: H1ContextStrictTargetDataset,
    transfer_map: Mapping[str, str],
) -> dict[str, np.ndarray]:
    full_carriers = {name: np.asarray(target.support[name].carriers["full"], np.float64) for name in H1_M4_FOLD0_TARGET}
    for name, donor in transfer_map.items():
        carrier = full_carriers[donor]
        _need(carrier.shape == (EXPECTED_NEURONS, CARRIER_DIM), f"{name}: donor carrier shape drift")
        _need(np.array_equal(carrier, full_carriers[donor]), f"{name}: cross-session carrier must equal donor full carrier")
    return {name: full_carriers[donor] for name, donor in transfer_map.items()}


def pooled_r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth64 = np.asarray(truth, np.float64)
    estimate64 = np.asarray(estimate, np.float64)
    sse = float(np.square(truth64 - estimate64).sum())
    tss = float(np.square(truth64 - truth64.mean(axis=0, keepdims=True)).sum())
    _need(np.isfinite(sse) and np.isfinite(tss) and tss > 0, "pooled R2 undefined")
    return 1.0 - sse / tss


def aggregate_pooled_r2(per_session_truth: Mapping[str, np.ndarray], per_session_estimate: Mapping[str, np.ndarray]) -> float:
    names = sorted(per_session_truth)
    truth = np.concatenate([np.asarray(per_session_truth[name], np.float64) for name in names])
    estimate = np.concatenate([np.asarray(per_session_estimate[name], np.float64) for name in names])
    return pooled_r2(truth, estimate)


@dataclass(frozen=True)
class InterventionViewDataset(Dataset):
    base: H1ContextStrictTargetDataset
    intervention: str
    carrier_override: Mapping[str, np.ndarray] | None = None

    @property
    def window_indices_sha256(self) -> str:
        return self.base.window_indices_sha256

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        name, start = self.base.window_indices[int(index)]
        record = self.base.records[name]
        item = self.base.support[name]
        end = start + WINDOW
        _need(start >= item.query_first_bin and record.eval_mask[end - 1], "query boundary violation")
        if self.carrier_override is not None:
            carrier = np.asarray(self.carrier_override[name], np.float32)
        else:
            carrier = np.asarray(item.carriers[self.intervention], np.float32)
        return (
            record.neural[start:end],
            record.velocity[start:end],
            item.identity,
            name,
            carrier,
        )


def _load_checkpoint(path: Path, config_path: Path) -> tuple[dict[str, Any], Any, Mapping[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("h1_sparse_event_endpoint")
    _need(isinstance(checkpoint, dict) and "state_dict" in checkpoint, "checkpoint missing state_dict")
    _need(isinstance(metadata, dict) and metadata.get("schema") == CHECKPOINT_SCHEMA, "checkpoint metadata missing")
    _need(metadata.get("arm") == "full", "cross-session control requires Context Full checkpoint")
    _need(int(checkpoint.get("epoch", -1)) == 49 and int(metadata.get("epochs_completed", 0)) == 50, "checkpoint not terminal epoch 49")
    _need(metadata.get("fold_date") == "19250101" and metadata.get("carrier_dim") == 5, "checkpoint fold/topology drift")
    _need(metadata.get("target_session_optimizer_steps") == 0 and metadata.get("target_session_backward_steps") == 0, "checkpoint target legality drift")
    _need(metadata.get("config_sha256") == sha256_file(config_path), "checkpoint config SHA mismatch")
    config = OmegaConf.load(config_path)
    _need(config.pilot.arm == "full" and int(config.model.net.carrier_dim) == 5, "resolved config drift")
    return checkpoint, config, metadata


def _instantiate_model(config: Any, checkpoint: Mapping[str, Any], device: torch.device):
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _score(model, dataset: Dataset, device: torch.device, label: str) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sessions: list[str] = []
    before = state_hash(model.state_dict())
    with torch.no_grad():
        for neural, target, identity, name, carrier in loader:
            output = model(
                neural.to(device=device, dtype=torch.float32),
                calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
                carrier=carrier.to(device=device, dtype=torch.float32),
            )
            if model.hparams.decode_last_timestep_only:
                output, target = output[:, -1:, :], target[:, -1:, :]
            if model.hparams.predict_scaled_behavior:
                output = output / model.hparams.behavior_scaling_factor
            predictions.append(output[:, -1].cpu().numpy())
            targets.append(target[:, -1].cpu().numpy())
            sessions.extend(list(name))
    after = state_hash(model.state_dict())
    assert_state_immutable(before, after, label)
    prediction = np.concatenate(predictions)
    target_values = np.concatenate(targets)
    per_session = {}
    for session_name in H1_M4_FOLD0_TARGET:
        mask = np.asarray([item == session_name for item in sessions], dtype=bool)
        per_session[session_name] = {
            "samples": int(mask.sum()),
            "r2": pooled_r2(target_values[mask], prediction[mask]),
        }
    return {
        "pooled_r2": pooled_r2(target_values, prediction),
        "samples": len(dataset),
        "per_session": per_session,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "state_immutable": before == after,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def _carrier_sha_by_arm(
    target: H1ContextStrictTargetDataset,
    transfer_map: Mapping[str, str],
    cross_session_carriers: Mapping[str, np.ndarray],
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for name in H1_M4_FOLD0_TARGET:
        support = target.support[name]
        donor = transfer_map[name]
        out[name] = {
            "full": support.carrier_sha256["full"],
            "cross_session": support.carrier_sha256["full"] if donor == name else target.support[donor].carrier_sha256["full"],
            "row": support.carrier_sha256["row"],
            "zero": support.carrier_sha256["zero"],
            "cross_session_donor": donor,
            "cross_session_array_sha256": event_v1.array_sha256(np.asarray(cross_session_carriers[name], np.float64)),
        }
        _need(
            out[name]["cross_session"] == target.support[donor].carrier_sha256["full"],
            f"{name}: cross_session SHA must equal donor full SHA",
        )
        _need(
            out[name]["cross_session_array_sha256"] == event_v1.array_sha256(np.asarray(cross_session_carriers[name], np.float64)),
            f"{name}: cross_session carrier array SHA mismatch",
        )
    return out


def _distance_anchors(cross_session_r2: float, full_r2: float, row_r2: float, zero_r2: float) -> dict[str, float]:
    return {
        "cross_session_to_full_abs": abs(cross_session_r2 - full_r2),
        "cross_session_to_row_abs": abs(cross_session_r2 - row_r2),
        "cross_session_to_zero_abs": abs(cross_session_r2 - zero_r2),
        "cross_session_minus_full": cross_session_r2 - full_r2,
        "cross_session_minus_row": cross_session_r2 - row_r2,
        "cross_session_minus_zero": cross_session_r2 - zero_r2,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve()
    checkpoint, config, metadata = _load_checkpoint(checkpoint_path, config_path)
    snapshot = load_snapshot(args.source_snapshot_receipt)
    source = H1ContextEventDataModule(
        task="h1",
        data_dir=str(args.data_dir.resolve()),
        cache_dir=str((PROJECT_ROOT / "pilot_artifacts/h1_context_event_carrier/shared_source_cache").resolve()),
        source_snapshot_receipt=str(args.source_snapshot_receipt),
    )
    source.setup("fit")
    _need(metadata["source_manifest_sha256"] == source.pilot_manifest_sha256, "checkpoint/source manifest mismatch")
    _need(metadata["normalizer_sha256"] == source.normalizer.normalizer_sha256, "checkpoint/normalizer mismatch")
    target = build_context_target_dataset(data_dir=args.data_dir, source_module=source)
    _need(target.window_indices_sha256 == EXPECTED_QUERY_SHA, "query window SHA drift")
    _need(len(target) == EXPECTED_QUERY_COUNT, f"expected {EXPECTED_QUERY_COUNT} query windows, got {len(target)}")
    channel_contract = verify_same_array_channel_contract(target.records)
    transfer_map = cross_session_transfer_map(H1_M4_FOLD0_TARGET)
    cross_session_carriers = build_cross_session_carriers(target, transfer_map)
    carrier_sha_by_arm = _carrier_sha_by_arm(target, transfer_map, cross_session_carriers)
    model = _instantiate_model(config, checkpoint, device)
    del checkpoint
    datasets = {
        "full": InterventionViewDataset(target.with_intervention("full"), "full"),
        "row": InterventionViewDataset(target.with_intervention("row"), "row"),
        "zero": InterventionViewDataset(target.with_intervention("zero"), "zero"),
        "cross_session": InterventionViewDataset(target.with_intervention("full"), "cross_session", cross_session_carriers),
    }
    scores = {arm: _score(model, datasets[arm], device, f"cross-session/{arm}") for arm in ARMS}
    query_sha = scores["full"]["query_window_indices_sha256"]
    for arm in ARMS:
        _need(scores[arm]["query_window_indices_sha256"] == query_sha, f"{arm}: query SHA differs across arms")
        _need(scores[arm]["samples"] == EXPECTED_QUERY_COUNT, f"{arm}: sample count drift")
    full_r2 = float(scores["full"]["pooled_r2"])
    full_delta = full_r2 - SEALED_FULL_R2
    _need(abs(full_delta) <= FULL_INTEGRITY_TOLERANCE, f"full arm failed integrity gate: {full_r2} vs {SEALED_FULL_R2} (delta={full_delta})")
    contrasts = {
        "full_minus_cross_session": scores["full"]["pooled_r2"] - scores["cross_session"]["pooled_r2"],
        "cross_session_minus_row": scores["cross_session"]["pooled_r2"] - scores["row"]["pooled_r2"],
        "cross_session_minus_zero": scores["cross_session"]["pooled_r2"] - scores["zero"]["pooled_r2"],
    }
    distance_anchors = _distance_anchors(
        float(scores["cross_session"]["pooled_r2"]),
        float(scores["full"]["pooled_r2"]),
        float(scores["row"]["pooled_r2"]),
        float(scores["zero"]["pooled_r2"]),
    )
    evaluator_path = Path(__file__).resolve()
    input_nwb = {
        name: {
            "path": str(target.records[name].path),
            "sha256": target.records[name].input_sha256,
        }
        for name in H1_M4_FOLD0_TARGET
    }
    receipt = {
        "schema": SCHEMA,
        "status": "PASS_CROSS_SESSION_TRANSFER",
        "fold_date": "19250101",
        "seed": 42,
        "evaluation_device": str(device),
        "evaluator": {"path": str(evaluator_path), "sha256": sha256_file(evaluator_path)},
        "checkpoint_binding_completed_before_target_open": True,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "metadata": dict(metadata),
            "resolved_config": str(config_path),
            "resolved_config_sha256": sha256_file(config_path),
        },
        "source_manifest": source.pilot_manifest(),
        "source_manifest_sha256": source.pilot_manifest_sha256,
        "source_snapshot": {
            "receipt": str(snapshot["receipt_path"]),
            "receipt_sha256": sha256_file(snapshot["receipt_path"]),
            "snapshot": str(snapshot["snapshot_path"]),
            "snapshot_sha256": sha256_file(snapshot["snapshot_path"]),
        },
        "target": {
            "sessions": list(H1_M4_FOLD0_TARGET),
            "query_window_count": EXPECTED_QUERY_COUNT,
            "query_window_indices_sha256": query_sha,
            "support_and_carrier_hashes": target.support_and_carrier_hashes(),
            "post_four_trial_query": True,
            "channel_contract": channel_contract,
            "cross_session_transfer_map": dict(transfer_map),
            "carrier_sha256_by_recording_by_arm": carrier_sha_by_arm,
        },
        "metrics": scores,
        "sealed_reference": {
            "context_full_pooled_r2": SEALED_FULL_R2,
            "context_row_pooled_r2": SEALED_ROW_R2,
            "context_zero_pooled_r2": SEALED_ZERO_R2,
            "full_reproduction_delta": full_delta,
            "full_integrity_tolerance": FULL_INTEGRITY_TOLERANCE,
        },
        "contrasts": contrasts,
        "distance_anchors": distance_anchors,
        "input_nwb": input_nwb,
        "scope": {
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training_launched": False,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
        },
        "integrity": {
            "full_integrity_gate_pass": abs(full_delta) <= FULL_INTEGRITY_TOLERANCE,
            "query_sha_identical_across_arms": True,
            "model_state_immutable": all(scores[arm]["state_immutable"] for arm in ARMS),
            "model_state_sha256_before": scores["full"]["state_sha256_before"],
            "model_state_sha256_after": scores["full"]["state_sha256_after"],
        },
    }
    output, digest = write_immutable_json(args.output, receipt)
    return {**receipt, "receipt_path": str(output), "receipt_sha256": digest}


def _print_summary_table(scores: Mapping[str, Mapping[str, Any]], contrasts: Mapping[str, float], distance_anchors: Mapping[str, float]) -> None:
    sessions = list(H1_M4_FOLD0_TARGET)
    header = ["arm", "pooled_r2"] + [f"{session}_r2" for session in sessions]
    rows = []
    for arm in ARMS:
        row = [arm, f"{scores[arm]['pooled_r2']:.6f}"]
        for session in sessions:
            row.append(f"{scores[arm]['per_session'][session]['r2']:.6f}")
        rows.append(row)
    widths = [max(len(str(cell)) for cell in column) for column in zip(header, *rows)]
    def fmt_line(cells: Sequence[str]) -> str:
        return "  ".join(str(cell).rjust(widths[index]) for index, cell in enumerate(cells))
    print(fmt_line(header))
    print(fmt_line(["-" * width for width in widths]))
    for row in rows:
        print(fmt_line(row))
    print()
    print("contrasts:")
    for key, value in contrasts.items():
        print(f"  {key}: {value:.6f}")
    print("distance_anchors:")
    for key, value in distance_anchors.items():
        print(f"  {key}: {value:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/000954")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_FULL_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_FULL_CONFIG)
    parser.add_argument("--source-snapshot-receipt", type=Path, default=DEFAULT_SOURCE_SNAPSHOT_RECEIPT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "pilot_artifacts/h1_cross_session_transfer/H1_CROSS_SESSION_CARRIER_TRANSFER_FOLD0_v1.json",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = evaluate(args)
    _print_summary_table(result["metrics"], result["contrasts"], result["distance_anchors"])
    print(json.dumps({"status": result["status"], "receipt": result["receipt_path"], "sha256": result["receipt_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
