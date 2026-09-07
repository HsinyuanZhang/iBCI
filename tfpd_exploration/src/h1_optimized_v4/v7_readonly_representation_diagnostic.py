"""Read-only V4 signed-frontend versus immutable-C2 representation audit.

This is deliberately a diagnostic, not a model-selection or training entry
point.  It evaluates a sealed V4 EMA state and the immutable C2 deployment on
matched, deterministic source train/minival endpoint samples.  C2 is driven
through its real chronological API, including all preceding bins, before a
feature snapshot is taken.  The report exposes amplitude and temporal-change
statistics only; it never opens an optimiser or writes a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.c2_reference import (
    PACKAGE, PACKAGE_SHA256, FalconConfig, FalconTask, H1EPFiLMSpintDecoder,
    _reset_tag, _sha256_file, _verify_bank,
)
from tfpd_exploration.src.h1_optimized_v2.capacity_probe import r2
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .model import H1SignedFull


V4_ATTEMPT = ROOT / "v4_full_continue24_deterministic_v1"
V4_EMA = V4_ATTEMPT / "independent_score_export/full_selected_plain_ema_model_state.pt"
DEFAULT_OUT = ROOT / "v7_readonly_v4_vs_c2_representation_v2/report.json"
SAMPLES_PER_SESSION = 16


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarize(values: list[np.ndarray]) -> dict[str, float]:
    """Summarize [example, time, feature] tensors without retaining examples."""
    x = np.concatenate(values, axis=0).astype(np.float64, copy=False)
    temporal_std = x.std(axis=1)
    return {
        "examples": int(x.shape[0]), "time_bins": int(x.shape[1]), "features": int(x.shape[2]),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "global_std": float(x.std()),
        "per_feature_mean_temporal_std": float(temporal_std.mean()),
        "per_feature_median_temporal_std": float(np.median(temporal_std)),
        "adjacent_delta_rms": float(np.sqrt(np.mean(np.square(np.diff(x, axis=1))))),
        "first_last_delta_rms": float(np.sqrt(np.mean(np.square(x[:, -1] - x[:, 0])))),
        "endpoint_feature_std": float(x[:, -1].std(axis=0).mean()),
    }


def _summarize_unit_feature(values: list[np.ndarray]) -> dict[str, float]:
    """Summarize C2's post-fc_in [example, unit, latent] representation.

    C2's `fc_in` is a linear projection of the W700 axis.  Its middle axis is
    therefore unit, not time; presenting it as a time statistic would be a
    category error.  Temporal comparisons below intentionally use C2's
    pre-fc_in W700 `src` only.
    """
    x = np.concatenate(values, axis=0).astype(np.float64, copy=False)
    return {
        "examples": int(x.shape[0]), "units": int(x.shape[1]), "latent_features": int(x.shape[2]),
        "rms": float(np.sqrt(np.mean(np.square(x)))), "global_std": float(x.std()),
        "per_unit_mean_latent_std": float(x.std(axis=2).mean()),
        "per_latent_mean_unit_std": float(x.std(axis=1).mean()),
    }


def _starts(row: dict) -> np.ndarray:
    candidates = np.asarray(row["query_starts"], dtype=np.int64)
    if len(candidates) < SAMPLES_PER_SESSION:
        raise RuntimeError("insufficient frozen query starts")
    # Positions, rather than values, make this deterministic on both cache
    # partitions without using a response, prediction, or validation metric.
    positions = np.linspace(0, len(candidates) - 1, SAMPLES_PER_SESSION, dtype=np.int64)
    return candidates[positions]


def _v4_partition(model: H1SignedFull, cache: dict, partition: str, device: torch.device) -> dict:
    z_values: list[np.ndarray] = []; prediction: list[np.ndarray] = []; target: list[np.ndarray] = []
    with torch.inference_mode():
        for session, row in sorted(cache[partition].items()):
            starts = _starts(row)
            x = torch.stack([torch.as_tensor(row["neural"][s:s + 700]) for s in starts]).to(device=device, dtype=torch.float32)
            bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
            z = model.encode_frontend(x, bank)
            p = model.readout(model.final_norm(model.forward_hidden_from_frontend(z))) / 20.0
            z_values.append(z.cpu().numpy()); prediction.append(p.cpu().numpy())
            target.append(np.asarray([row["velocity"][s + 699] for s in starts], dtype=np.float32))
    p, y = np.concatenate(prediction), np.concatenate(target)
    return {"frontend": _summarize(z_values), "r2_concat_native": r2(p, y),
            "prediction_std": float(p.std()), "target_std": float(y.std())}


def _c2_partition(decoder: H1EPFiLMSpintDecoder, cache: dict, partition: str) -> dict:
    src_values: list[np.ndarray] = []; fcin_values: list[np.ndarray] = []
    prediction: list[np.ndarray] = []; target: list[np.ndarray] = []; parity: list[dict] = []
    for session, row in sorted(cache[partition].items()):
        starts = _starts(row); endpoints = starts + 699; wanted = set(map(int, endpoints))
        parity.append(_verify_bank(decoder, row, session)); decoder.reset([_reset_tag(session)])
        for index, observation in enumerate(row["neural"][:int(endpoints.max()) + 1]):
            value = decoder.predict(np.asarray(observation, dtype=np.float32)[None])[0]
            if index not in wanted:
                continue
            # `predict` has just executed the production C2 stateful API.
            # Reconstruct exactly its immutable feature input from its actual
            # retained W700 buffer and cached EP-FILM identity.
            neural = torch.as_tensor(decoder.observation_buffer.transpose(1, 0, 2), dtype=torch.float32, device=decoder.device)
            src = neural.permute(0, 2, 1) + decoder._film_identity()
            fcin = decoder.model.fc_in(src)
            src_values.append(src.permute(0, 2, 1).detach().cpu().numpy())
            fcin_values.append(fcin.detach().cpu().numpy())
            prediction.append(value[None]); target.append(np.asarray(row["velocity"][index], dtype=np.float32)[None])
    p, y = np.concatenate(prediction), np.concatenate(target)
    return {"pre_fc_in_src": _summarize(src_values), "fc_in_unit_feature": _summarize_unit_feature(fcin_values),
            "r2_concat_native": r2(p, y), "prediction_std": float(p.std()),
            "target_std": float(y.std()), "bank_parity": parity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic receipt: {args.output}")
    if not V4_EMA.is_file() or not V4_ATTEMPT.joinpath("input_authority.json").is_file():
        raise RuntimeError("sealed V4 continuation export/authority is absent")
    if _sha256_file(PACKAGE) != PACKAGE_SHA256:
        raise RuntimeError("immutable C2 package hash drift")
    cache = build_or_load(); authority = json.loads((ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, authority)
    device = torch.device(args.device)
    v4 = H1SignedFull().to(device)
    state = torch.load(V4_EMA, map_location="cpu", weights_only=True)
    v4.load_state_dict(state, strict=True); v4.eval()
    if int(v4.frontend_contract_version.item()) != 4:
        raise RuntimeError("V4 frontend contract drift")
    c2 = H1EPFiLMSpintDecoder(FalconConfig(task=FalconTask.h1), PACKAGE, batch_size=1, device=args.device)
    report = {
        "schema": "h1_v7_readonly_representation_diagnostic_v1",
        "status": "COMPLETE_READ_ONLY_DIAGNOSTIC",
        "scope": "sealed V4 signed frontend versus immutable C2 exact deployed features; deterministic 16 W700 endpoints per each of 13 train/minival sessions",
        "prohibitions": ["no training", "no optimiser", "no selection", "no minival model choice", "no checkpoint writes"],
        "V4": {"plain_ema": str(V4_EMA), "plain_ema_sha256": sha(V4_EMA), "frontend_contract_version": 4},
        "C2": {"package": str(PACKAGE), "package_sha256": PACKAGE_SHA256,
               "streaming": "one chronological production predict call per bin through each requested endpoint; no trial resets"},
        "source_cache_authority_sha256": sha(ROOT / "source_cache_authority.json"),
        "code_sha256": sha(Path(__file__)),
        "partitions": {"train": {}, "minival": {}},
    }
    for partition in ("train", "minival"):
        report["partitions"][partition]["v4"] = _v4_partition(v4, cache, partition, device)
        report["partitions"][partition]["c2"] = _c2_partition(c2, cache, partition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({p: {m: d["r2_concat_native"] for m, d in body.items()} for p, body in report["partitions"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
