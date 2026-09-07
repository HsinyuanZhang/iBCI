"""Read-only, fixed-208 input/gradient diagnostic for H1 capacity probes.

This is deliberately not a fit or a selector.  It loads a named immutable
checkpoint, evaluates only the source-train windows frozen by that probe, and
writes one non-overwritable receipt.  Perturbations are applied in native
neural-input units; predictions and targets in the receipt are native velocity
units (the model's 20x runtime target is divided back out before reporting).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from .cache import ROOT as CACHE_ROOT, build_or_load, validate_authority
from .capacity_probe import W, batch
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def r2(p: np.ndarray, y: np.ndarray) -> float:
    p, y = p.astype("float64"), y.astype("float64")
    return float(1 - np.square(p - y).sum() / np.square(y - y.mean(axis=0)).sum())


def covariance(a: np.ndarray, b: np.ndarray) -> dict:
    # Flattening keeps this a compact scalar diagnostic without silently
    # choosing a velocity dimension; the per-dimension values are also saved.
    flat = float(np.cov(a.ravel(), b.ravel(), ddof=0)[0, 1])
    per_dim = [float(np.cov(a[:, i], b[:, i], ddof=0)[0, 1]) for i in range(a.shape[1])]
    return {"flattened": flat, "per_velocity_dimension": per_dim}


def fixed_examples(cache: dict, ids: dict, dev: torch.device):
    out = []
    for session, starts in sorted(ids.items()):
        row = cache["train"][session]
        x, _, y = batch(row, np.asarray(starts, dtype=int), dev)
        bank = H1Bank(row["bank"]["E0"].to(dev), row["bank"]["T"].to(dev), row["bank"]["unit_mask"].to(dev))
        out.extend((session, x[i : i + 1], y[i : i + 1], bank) for i in range(len(x)))
    return out


def predict(model, examples, alteration=None):
    values, targets = [], []
    model.eval()
    with torch.no_grad():
        for i, (_, x, y, bank) in enumerate(examples):
            z = x if alteration is None else alteration(x, i)
            values.append((model.forward_last(z, bank) / model.prediction_divisor).cpu().numpy())
            targets.append(np.asarray(y, dtype=np.float32))
    return np.concatenate(values), np.concatenate(targets)


def perturbation_summary(model, examples, input_mean):
    base, target = predict(model, examples)
    count = len(examples)
    perm = np.random.default_rng(20260905).permutation(count)
    # A permutation is intentionally cross-example: the session's immutable
    # bank remains fixed, so this measures dependence on its neural input.
    source_x = [x for _, x, _, _ in examples]
    variants = {
        "x_permuted_across_examples": lambda x, i: source_x[int(perm[i])],
        "x_zero": lambda x, i: torch.zeros_like(x),
        "last_bin_only": lambda x, i: torch.cat((torch.zeros_like(x[:, :-1]), x[:, -1:]), dim=1),
        "history_only_last_bin_zeroed": lambda x, i: torch.cat((x[:, :-1], torch.zeros_like(x[:, -1:])), dim=1),
        "train_mean_per_neural_dimension": lambda x, i: input_mean.expand_as(x),
    }
    result = {
        "baseline": {
            "r2_concat": r2(base, target), "prediction_std": float(base.std()),
            "prediction_target_covariance": covariance(base, target),
        },
        "perturbations": {},
    }
    for name, op in variants.items():
        p, _ = predict(model, examples, op)
        delta = p - base
        result["perturbations"][name] = {
            "r2_concat_vs_unmodified_target": r2(p, target),
            "prediction_std": float(p.std()),
            "mean_abs_prediction_change": float(np.abs(delta).mean()),
            "rms_prediction_change": float(np.sqrt(np.square(delta).mean())),
            "max_abs_prediction_change": float(np.abs(delta).max()),
            "prediction_target_covariance": covariance(p, target),
        }
    return result


def gradient_summary(model, examples):
    """Aggregate exact source loss gradients, preserving the frozen inputs."""
    model.train(False)
    model.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for _, x, y, bank in examples:
        prediction = model.forward_last(x, bank)
        loss = torch.nn.functional.mse_loss(
            prediction,
            torch.as_tensor(y * model.prediction_divisor, device=prediction.device),
        )
        loss.backward()
        loss_sum += float(loss.detach())
    selected = {}
    for name, parameter in model.named_parameters():
        if name.startswith("frontend.") or name.startswith("readout."):
            selected[name] = float(parameter.grad.norm().cpu()) if parameter.grad is not None else 0.0
    groups = {
        "frontend": float(np.sqrt(sum(v * v for k, v in selected.items() if k.startswith("frontend.")))),
        "readout": float(np.sqrt(sum(v * v for k, v in selected.items() if k.startswith("readout.")))),
    }
    model.zero_grad(set_to_none=True)
    return {"loss_space": "20x native velocity MSE", "mean_per_example_loss": loss_sum / len(examples), "group_l2_norm": groups, "parameter_l2_norm": selected}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--revision", choices=("v2", "v3", "v4"), required=True,
                   help="Exact model revision used to construct the checkpoint before strict loading.")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic receipt: {args.output}")
    frozen = json.loads((args.probe / "frozen_ids.json").read_text())
    cache = build_or_load(); validate_authority(cache, frozen["source_authority"])
    dev = torch.device(args.device)
    examples = fixed_examples(cache, frozen["ids"], dev)
    # Mean is calculated over precisely the frozen windows, without targets.
    input_mean = torch.cat([x for _, x, _, _ in examples], dim=0).mean((0, 1), keepdim=True).view(1, 1, -1)
    module = importlib.import_module(f"tfpd_exploration.src.h1_optimized_{args.revision}.model")
    full, query = module.make_matched_pair(activity_scale=1.0 if args.revision == "v4" else 32.0)
    arms = {"full": full.to(dev), "t": query.to(dev)}
    result = {"schema": "h1_source_capacity_probe_input_gradient_diagnostic_v2", "status": "COMPLETE_READ_ONLY", "probe": str(args.probe), "revision": args.revision, "model_module": module.__name__, "code_sha256": {"diagnostic": sha256(Path(__file__)), "model": sha256(Path(module.__file__))}, "checkpoint_loading": "strict=True, all checkpoint model keys required; no keys omitted", "probe_report_sha256": sha256(args.probe / "report.json"), "frozen_ids_sha256": sha256(args.probe / "frozen_ids.json"), "checkpoint_sha256": {}, "examples": len(examples), "input_space": "native binned neural activity", "prediction_target_space": "native velocity", "input_mean_scope": "all values of the fixed 208 source windows", "arms": {}}
    for name, model in arms.items():
        checkpoint = args.probe / f"{name}_latest.pt"
        payload = torch.load(checkpoint, map_location=dev, weights_only=True)
        # The revision constructor must expose every stored key, including
        # persistent contract buffers such as V3's frontend_contract_version.
        # Do not filter, rename, or otherwise weaken this boundary.
        model.load_state_dict(payload["model"], strict=True)
        result["checkpoint_sha256"][name] = sha256(checkpoint)
        result["arms"][name] = {"input_sensitivity": perturbation_summary(model, examples, input_mean), "source_loss_gradients": gradient_summary(model, examples)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "examples": len(examples), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
