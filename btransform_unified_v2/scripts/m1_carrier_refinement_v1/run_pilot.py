#!/usr/bin/env python3
"""CPU-only, source-selected frozen-B carrier-residual M1 pilot.

This is an exploratory refinement runner.  It never modifies the cross-session
M1 program, never optimizes target data, and requires an already completed B
run scored by the strict ``m1_score.py`` scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types
import copy
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CROSS = ROOT / "btransform_unified_v2" / "scripts" / "cross_session_v1"
sys.path[:0] = [str(ROOT / "btransform_unified_v2" / "src"), str(CROSS), str(ROOT / "btransform_unified_v1" / "src"), str(ROOT)]

import aligned_carrier
import m1_data
import m1_score
import m1_train
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v2.cross_session_m1_model import B_ACTIVITY_ONLY, CrossSessionM1Decoder
from btransform_unified_v2.m1_carrier_residual_candidate import FrozenBCarrierResidual

FEATURE_DIM = 24
BATCH = 512
EPOCHS = 20
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-3
DROPOUT_P = 0.5
ALPHAS = (0.25, 0.5, 1.0)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_array(x: Any) -> str:
    a = np.ascontiguousarray(np.asarray(x))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **values)
    tmp.replace(path)


def model_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def runtime_sources() -> dict[str, dict[str, str]]:
    names = ("m1_data", "m1_train", "m1_score", "aligned_carrier")
    modules = {name: sys.modules[name] for name in names}
    modules["runner"] = sys.modules[__name__]
    modules["residual"] = sys.modules["btransform_unified_v2.m1_carrier_residual_candidate"]
    modules["falcon_datamodule"] = sys.modules.get("src.data.falcon_datamodule")
    return {name: {"file": str(Path(module.__file__).resolve()), "sha256": sha_file(Path(module.__file__).resolve())} for name, module in modules.items() if module is not None and getattr(module, "__file__", None)}


def binding(a: argparse.Namespace) -> dict[str, Any]:
    return {"protocol": protocol(a), "b_train_receipt_sha256": sha_file(a.b_dest / "train_receipt.json"),
            "b_selected_ema_sha256": sha_file(a.b_dest / "selected_ema.pt"),
            "b_resume_sha256": sha_file(a.b_dest / "resume_latest.pt"), "runtime_sources": runtime_sources()}


def protocol(a: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "m1_frozen_b_carrier_refinement_v1",
        "status": "PREDECLARED",
        "fold": a.target, "seed": a.seed, "device": "cpu", "cuda_forbidden": True,
        "exploratory_note": "The 20120924 fold is exploratory development; this pilot is not an independent hidden-test claim.",
        "base": {"arm": B_ACTIVITY_ONLY, "checkpoint_view": "selected_ema", "base_trainable": False},
        "fit": {"source_only": True, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
                "train_stride": a.train_stride, "carrier_whole_channel_dropout": DROPOUT_P,
                "carrier_visibility_interpretation": "missing-carrier exact fallback / stochastic visible-row training; false rows receive no residual-head gradient and this is not a robustness or session-overreliance claim",
                "source_sampling": "equal_session_updates; deterministic per-session samples with replacement only to equalize update count"},
        "features": {"dim": FEATURE_DIM, "law": "aligned_carrier.causal_carrier_projection on FalconDataset padded neural_data; feature endpoint index=window_start_padded+99", "range_reset": "only each independent source train/val range starts; never individual trial boundaries"},
        "selection": {"surface": "complete source validation only", "alphas": list(ALPHAS), "rule": "highest equal-session R2 with every source gain over B >= -1e-6; tie earliest epoch then alpha order", "fallback": "alpha=0 exact B"},
        "target_policy": {"optimizer_steps": 0, "labels_used_for_fit_or_selection": False, "baseline": "strict saved selected-EMA B prediction NPZ"},
        "interpretation": "exploratory network-plus-carrier candidate: its prediction-conditioned residual head means an observed gain cannot be attributed to carrier information alone",
        "runtime": {"threads": a.threads, "torch_version": torch.__version__},
    }


def configure_cpu(threads: int) -> None:
    if torch.cuda.is_available():
        # Do not silently accept an accelerator merely because one is installed.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(threads)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass


def b_args(a: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(dest=a.b_dest, target=a.target, arm=B_ACTIVITY_ONLY, seed=a.seed, device="cpu")


def load_base(a: argparse.Namespace, *, include_target: bool = False) -> tuple[CrossSessionM1Decoder, dict[str, Any], dict[str, Any], Any | None]:
    """Strictly reproduce the B selected-EMA load/install order from m1_score."""
    ba = b_args(a)
    # Constructor installs streaming_calibration_exp before the source reader
    # imports src.data.falcon_datamodule; preserve that import ordering.
    model = CrossSessionM1Decoder(B_ACTIVITY_ONLY, seed=a.seed).to("cpu")
    m1_train._configure(model)
    receipt, final, selected = m1_score._require_receipts(ba)
    for recorded_path, recorded_sha in receipt.get("source_code_sha256", {}).items():
        path = Path(recorded_path)
        if not path.is_file() or sha_file(path) != recorded_sha:
            raise RuntimeError(f"completed B source-code binding drift: {path}")
    cache = m1_score._cache_path(ba)
    if not cache.is_file(): raise FileNotFoundError(f"frozen source cache missing: {cache}")
    src = m1_data.materialize_sources(a.target, cache=cache.parent)
    if receipt.get("actual_source_arrays") != m1_train.source_evidence(src):
        raise RuntimeError("rematerialized source evidence differs from completed B receipt")
    if {k: m1_train.array_hash(v) for k, v in src["rsyn3"].items()} != receipt["actual_source_arrays"]["rsyn3_dictionary_and_normalizer"]:
        raise RuntimeError("frozen source rSyn3 arrays differ from B receipt")
    target_ds = None
    banks, calib = src["banks"], src["calib"]
    if include_target:
        target_ds = m1_score._target_dataset(a.target)
        target_calib = np.ascontiguousarray(target_ds.calib_trialized_neural_features[a.target][:10], dtype=np.float32)
        target_bank = m1_data._bank(a.target, np.zeros((64, 4), dtype=np.float32))
        banks, calib = {**banks, a.target: target_bank}, {**calib, a.target: target_calib}
    m1_train._install(model, B_ACTIVITY_ONLY, banks, calib)
    ema = DecoderEMA(model, m1_train.EMA_DECAY)
    model.load_state_dict(final["model"], strict=True)
    ema.load_state_dict(selected["ema"])
    m1_train._ema_apply(model, ema)
    model.eval()
    return model, src, receipt, target_ds


def cache_identity(model: CrossSessionM1Decoder, sessions: tuple[str, ...]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    raw = model._identity
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        for s in sessions:
            e0, direct = raw([s], torch.device("cpu"))
            cache[s] = (e0[0].detach().clone(), direct[0].detach().clone())
    def cached(this: CrossSessionM1Decoder, requested, device):
        if device.type != "cpu": raise RuntimeError("pilot identity cache is CPU-only")
        try: values = [cache[str(s)] for s in requested]
        except KeyError as e: raise RuntimeError(f"uncached identity session {e.args[0]}") from e
        return torch.stack([v[0] for v in values]), torch.stack([v[1] for v in values])
    model._identity = types.MethodType(cached, model)
    return cache


def dataset_features(ds: Any, carriers: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for session, raw in ds.neural_data.items():
        # The Falcon range has one padded stream.  There are no per-trial resets.
        result[session] = aligned_carrier.causal_carrier_projection(np.asarray(raw, dtype=np.float32), carriers[session])
    return result


class WindowSubset(Dataset):
    """A source-only, original-index preserving window subset."""
    def __init__(self, base: Any, indices: list[int]) -> None:
        self.base = base; self.indices = list(indices); self.window_size = base.window_size
        self.window_indices = [base.window_indices[i] for i in self.indices]
    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, i: int): return self.base[self.indices[i]]


def stride_subset(ds: Any, stride: int) -> tuple[WindowSubset, dict[str, dict[int, int]]]:
    groups: dict[str, list[int]] = {}
    for index, (session, _start) in enumerate(ds.window_indices): groups.setdefault(session, []).append(index)
    selected = [index for session in sorted(groups) for ordinal, index in enumerate(groups[session]) if ordinal % stride == 0]
    origin = {session: {int(ds.window_indices[i][1]): int(i) for i in indices} for session, indices in groups.items()}
    return WindowSubset(ds, selected), origin


def collect_base(model: CrossSessionM1Decoder, ds: Any, banks: Mapping[str, Any], feature_grid: Mapping[str, np.ndarray], origin: Mapping[str, Mapping[int, int]] | None = None) -> dict[str, dict[str, np.ndarray]]:
    rows: dict[str, dict[str, list[np.ndarray]]] = {}
    with torch.inference_mode():
        for batch_no, (x, y, sessions, valid, starts) in enumerate(m1_train.loader(ds, 42, shuffle=False), start=1):
            for s in dict.fromkeys(sessions):
                ix = [i for i, q in enumerate(sessions) if q == s]
                start = starts[ix].numpy().astype(np.int64)
                pred = model(x[ix].to("cpu"), banks[s], input_valid_mask=valid[ix].to("cpu")).cpu().numpy()
                endpoint = start + ds.window_size - 1
                feat = feature_grid[s][endpoint]
                row = rows.setdefault(s, {"prediction": [], "target": [], "feature": [], "start": [], "valid": [], "original_index": []})
                row["prediction"].append(pred); row["target"].append(y[ix, -1].numpy()); row["feature"].append(feat)
                row["start"].append(start); row["valid"].append(valid[ix].numpy())
                row["original_index"].append(np.asarray([origin[s][int(v)] for v in start], dtype=np.int64) if origin else np.full(len(start), -1, dtype=np.int64))
            if batch_no % 100 == 0: print(json.dumps({"stage": "source_cache", "batches": batch_no}, sort_keys=True), flush=True)
    return {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in rows.items()}


def save_source_cache(path: Path, train: Mapping[str, Mapping[str, np.ndarray]], val: Mapping[str, Mapping[str, np.ndarray]], meta: Mapping[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for split, values in (("train", train), ("val", val)):
        for s, row in values.items():
            for key, value in row.items(): arrays[f"{split}/{s}/{key}"] = value
    atomic_npz(path, **arrays)
    bound = dict(meta); bound["npz_sha256"] = sha_file(path)
    atomic_json(path.with_suffix(".json"), bound)


def load_source_cache(path: Path, sessions: tuple[str, ...]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    with np.load(path, allow_pickle=False) as z:
        out: list[dict[str, dict[str, np.ndarray]]] = []
        for split in ("train", "val"):
            out.append({s: {k: np.ascontiguousarray(z[f"{split}/{s}/{k}"]) for k in ("prediction", "target", "feature", "start", "valid", "original_index")} for s in sessions})
    return out[0], out[1]


def r2_by_session(base: Mapping[str, Mapping[str, np.ndarray]], pred: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {s: float(variance_weighted_r2(base[s]["target"], pred[s])) for s in base}


def evaluate(wrapper: FrozenBCarrierResidual, values: Mapping[str, Mapping[str, np.ndarray]], alpha: float) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    wrapper.eval(); pred: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for s, row in values.items():
            p = torch.from_numpy(row["prediction"]); f = torch.from_numpy(row["feature"])
            residual = wrapper.forward_cached(p, f, carrier_visible=torch.ones(len(p), dtype=torch.bool)) - p
            pred[s] = (p + float(alpha) * residual).cpu().numpy()
    return pred, r2_by_session(values, pred)


def carrier_stats(train: Mapping[str, Mapping[str, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
    # Equal-session rather than window-count weighting is fixed before fit.
    means = np.stack([v["feature"].mean(axis=0) for _, v in sorted(train.items())])
    vars_ = np.stack([np.mean((v["feature"] - means[i]) ** 2, axis=0) for i, (_, v) in enumerate(sorted(train.items()))])
    mean = means.mean(axis=0); scale = np.sqrt(vars_.mean(axis=0) + np.mean((means - mean) ** 2, axis=0))
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.maximum(scale, 1.0e-6).astype(np.float32))


def preflight(a: argparse.Namespace) -> None:
    model, src, receipt, _ = load_base(a, include_target=False)
    before = model_hash(model); ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}
    sessions = tuple(src["sources"]); original = model._identity
    # Raw-versus-cache proof on a real source batch before persistent patch.
    x, _y, names, valid, _st = next(iter(m1_train.loader(src["train"], 42, shuffle=False)))
    name = names[0]
    with torch.inference_mode():
        raw_ids = {s: original([s], torch.device("cpu")) for s in sessions}
        raw_repeat = original([name, name], torch.device("cpu"))
        raw_id = raw_ids[name]; raw_prediction = model(x, src["banks"][name], input_valid_mask=valid)
    cache = cache_identity(model, sessions)
    with torch.inference_mode():
        cached_ids = {s: model._identity([s], torch.device("cpu")) for s in sessions}
        cached_repeat = model._identity([name, name], torch.device("cpu"))
        cached_id = cached_ids[name]; cached_prediction = model(x, src["banks"][name], input_valid_mask=valid)
    if not all(torch.equal(a0, b0) for s in sessions for a0, b0 in zip(raw_ids[s], cached_ids[s])) or not all(torch.equal(a0, b0) for a0, b0 in zip(raw_repeat, cached_repeat)) or not torch.equal(raw_id[0], cached_id[0]) or not torch.equal(raw_prediction, cached_prediction):
        raise RuntimeError("cached identity changed selected B source prediction")
    torch.manual_seed(a.seed + 0xC411)
    wrapper = FrozenBCarrierResidual(model, feature_dim=FEATURE_DIM, carrier_dropout_p=DROPOUT_P)
    grid = dataset_features(src["train"], {s: np.asarray(src["banks"][s].carrier) for s in sessions})
    starts = _st.numpy(); feature = torch.from_numpy(grid[name][starts + 99])
    mean, scale = carrier_stats({name: {"feature": feature.numpy()}})
    wrapper.residual_head.install_source_normalization(mean, scale)
    try: wrapper.residual_head.install_source_normalization(mean, torch.zeros_like(scale))
    except ValueError: zero_scale_rejected = True
    else: raise RuntimeError("zero normalizer scale accepted")
    zero = wrapper(x, src["banks"][name], feature, input_valid_mask=valid, carrier_visible=torch.zeros(len(x), dtype=torch.bool))
    if not torch.equal(zero, raw_prediction): raise RuntimeError("false visibility is not exact B")
    visible = wrapper(x, src["banks"][name], feature, input_valid_mask=valid, carrier_visible=torch.ones(len(x), dtype=torch.bool))
    if not torch.equal(visible, raw_prediction): raise RuntimeError("zero initialized residual is not exact B")
    opt = torch.optim.AdamW(wrapper.trainable_parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    wrapper.train(); opt.zero_grad(); out = wrapper.forward_cached(raw_prediction, feature, carrier_visible=torch.ones(len(x), dtype=torch.bool)); loss = nn.functional.mse_loss(out, _y[:, -1]); loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in wrapper.parameters()): raise RuntimeError("nonfinite pilot gradient")
    if any(p.grad is not None for p in wrapper.base.parameters()): raise RuntimeError("frozen B received gradient")
    opt.step()
    after_visible = wrapper.forward_cached(raw_prediction, feature, carrier_visible=torch.ones(len(x), dtype=torch.bool))
    changed = not torch.equal(after_visible, raw_prediction)
    feature_changed = not torch.equal(after_visible, wrapper.forward_cached(raw_prediction, feature + 1.0, carrier_visible=torch.ones(len(x), dtype=torch.bool)))
    after_false = wrapper.forward_cached(raw_prediction, feature, carrier_visible=torch.zeros(len(x), dtype=torch.bool))
    if not changed or not feature_changed or not torch.equal(after_false, raw_prediction): raise RuntimeError("post-step carrier/false-fallback contract failed")
    if before != model_hash(model) or ptrs != {n: p.data_ptr() for n, p in model.named_parameters()}:
        raise RuntimeError("cache/preflight mutated frozen B")
    elapsed0 = time.perf_counter(); _ = model(x, src["banks"][name], input_valid_mask=valid); elapsed = time.perf_counter() - elapsed0
    report = {**protocol(a), "status": "PASSED", "binding": binding(a), "source_only": True, "target_opened": False,
              "b_receipt_sha256": sha_file(a.b_dest / "train_receipt.json"), "base_hash": before,
              "identity": {s: {"e0": sha_array(v[0].numpy()), "direct": sha_array(v[1].numpy())} for s, v in cache.items()},
              "checks": {"raw_cached_identity_byte_equal_all_source_sessions": True, "repeated_session_identity_stack_byte_equal": True, "raw_cached_forward_byte_equal": True, "zero_init_exact_B": True, "visibility_false_exact_B": True, "positive_normalizer_installed": True, "zero_normalizer_rejected": zero_scale_rejected, "base_no_grad": True, "optimizer_excludes_base": set(map(id, opt.param_groups[0]["params"])).isdisjoint({id(p) for p in wrapper.base.parameters()}), "visible_after_step_changed": changed, "changed_carrier_feature_changes_visible_prediction": feature_changed, "post_step_false_exact_B": True, "base_hash_unchanged": True},
              "timing": {"one_source_batch_seconds": elapsed, "rough_35000_window_seconds": elapsed * 35000.0 / len(x)}}
    atomic_json(a.dest / "preflight.json", report)
    print(json.dumps({"stage": "preflight", "status": "PASSED", "one_source_batch_seconds": elapsed,
                      "rough_35000_window_seconds": report["timing"]["rough_35000_window_seconds"]}, sort_keys=True), flush=True)


def fit_score(a: argparse.Namespace) -> None:
    pf = a.dest / "preflight.json"
    if not pf.is_file() or json.loads(pf.read_text()).get("status") != "PASSED": raise RuntimeError("run successful source-only preflight first")
    model, src, receipt, _ = load_base(a, include_target=False)
    if json.loads(pf.read_text()).get("binding") != binding(a): raise RuntimeError("preflight binding differs from current protocol/code/B checkpoints")
    sessions = tuple(src["sources"]); base_hash = model_hash(model); cache_identity(model, sessions)
    frozen = aligned_carrier.fit_source_carriers(sessions, m1_data.nwb_path)
    carriers = {s: np.asarray(frozen.source_carriers[s]) for s in sessions}
    basis = frozen.basis
    frozen_path = a.dest / "frozen_source_carriers.npz"
    atomic_npz(frozen_path, basis_dictionary=np.asarray(basis.dictionary), basis_scale=np.asarray(basis.scale), normalizer_mean=np.asarray(frozen.normalizer_mean), normalizer_scale=np.asarray(frozen.normalizer_scale), **{f"carrier/{s}": carriers[s] for s in sessions})
    frozen_sha = sha_file(frozen_path)
    grids_train, grids_val = dataset_features(src["train"], carriers), dataset_features(src["val"], carriers)
    cache_path = a.dest / "source_selected_b_cache.npz"
    train_subset, origin = stride_subset(src["train"], a.train_stride)
    cache_meta = {"binding": binding(a), "base_hash": base_hash, "source_evidence": m1_train.source_evidence(src), "feature_grid_sha256": {"train": {s: sha_array(grids_train[s]) for s in sessions}, "val": {s: sha_array(grids_val[s]) for s in sessions}}, "train_stride": a.train_stride, "feature_law": protocol(a)["features"], "base_prediction_runtime": {"device": "cpu", "dtype": "float32", "torch_version": torch.__version__, "threads": torch.get_num_threads()}}
    old_meta = json.loads(cache_path.with_suffix(".json").read_text()) if cache_path.is_file() and cache_path.with_suffix(".json").is_file() else None
    if old_meta and {k: v for k, v in old_meta.items() if k != "npz_sha256"} == cache_meta and old_meta.get("npz_sha256") == sha_file(cache_path):
        train_rows, val_rows = load_source_cache(cache_path, sessions)
    else:
        train_rows, val_rows = collect_base(model, train_subset, src["banks"], grids_train, origin), collect_base(model, src["val"], src["banks"], grids_val)
        save_source_cache(cache_path, train_rows, val_rows, cache_meta)
    # Cache already is the deterministic per-session original-ordinal stride subset.
    selected = {s: np.arange(len(train_rows[s]["start"]), dtype=np.int64) for s in sessions}
    mean, scale = carrier_stats({s: {"feature": train_rows[s]["feature"]} for s in sessions})
    # Explicitly reset head initialization after either a source-cache hit or miss.
    torch.manual_seed(a.seed + 0xC411)
    wrapper = FrozenBCarrierResidual(model, feature_dim=FEATURE_DIM, carrier_dropout_p=DROPOUT_P)
    wrapper.residual_head.install_source_normalization(mean, scale)
    opt = torch.optim.AdamW(wrapper.trainable_parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    baseline = {s: float(variance_weighted_r2(val_rows[s]["target"], val_rows[s]["prediction"])) for s in sessions}
    curve: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_state = copy.deepcopy(wrapper.residual_head.state_dict())
    max_n = max(len(selected[s]) for s in sessions); steps = int(np.ceil(max_n / BATCH))
    for epoch in range(EPOCHS + 1):
        candidates = {0.0: ({s: val_rows[s]["prediction"] for s in sessions}, baseline)}
        for alpha in ALPHAS: candidates[alpha] = evaluate(wrapper, val_rows, alpha)
        for alpha, (_pred, scores) in candidates.items():
            ok = all(scores[s] >= baseline[s] - 1.0e-6 for s in sessions)
            row = {"epoch": epoch, "alpha": alpha, "per_session": scores, "equal_session_mean": float(np.mean(list(scores.values()))), "eligible": ok}
            curve.append(row)
            if ok and (best is None or row["equal_session_mean"] > best["equal_session_mean"]):
                best = row; best_state = copy.deepcopy(wrapper.residual_head.state_dict())
        if epoch == EPOCHS: break
        wrapper.train()
        for step in range(steps):
            for pos, s in enumerate(sorted(sessions)):
                ix = selected[s][(np.arange(BATCH) + step * BATCH) % len(selected[s])]
                p = torch.from_numpy(train_rows[s]["prediction"][ix]); f = torch.from_numpy(train_rows[s]["feature"][ix]); y = torch.from_numpy(train_rows[s]["target"][ix])
                g = torch.Generator(device="cpu"); g.manual_seed(a.seed + epoch * 1000003 + step * 97 + pos)
                visible = torch.rand(len(ix), generator=g) >= DROPOUT_P
                opt.zero_grad(); pred = wrapper.forward_cached(p, f, carrier_visible=visible, visibility_generator=g); loss = nn.functional.mse_loss(pred, y)
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite residual loss")
                loss.backward()
                if not all(q.grad is None or torch.isfinite(q.grad).all() for q in wrapper.trainable_parameters()): raise FloatingPointError("nonfinite residual gradient")
                opt.step()
        print(json.dumps({"stage": "fit", "epoch": epoch + 1, "source_batches_per_session": steps}, sort_keys=True), flush=True)
    if best is None: best = {"epoch": 0, "alpha": 0.0, "per_session": baseline, "equal_session_mean": float(np.mean(list(baseline.values()))), "eligible": True}
    wrapper.residual_head.load_state_dict(best_state, strict=True)
    _selected_pred, selected_scores = evaluate(wrapper, val_rows, float(best["alpha"]))
    if any(abs(selected_scores[s] - best["per_session"][s]) > 1.0e-9 for s in sessions): raise RuntimeError("restored selected residual head does not reproduce selected source validation")
    if model_hash(model) != base_hash or any(p.grad is not None for p in wrapper.base.parameters()): raise RuntimeError("frozen B changed during source residual fit")
    selection = {"status": "SEALED", "selection": best, "restored_selected_source_validation": selected_scores, "baseline": baseline, "curve": curve, "base_hash": base_hash, "frozen_source_carriers_sha256": frozen_sha, "source_only_selection": True,
                 "source_validation_cpu_baseline_recomputed": baseline,
                 "comparison_note": "all selection values are recomputed CPU float32 values from the selected EMA; do not equate them bitwise with any prior GPU surface"}
    atomic_json(a.dest / "sealed_selection.json", selection)
    torch.save({"residual_head": wrapper.residual_head.state_dict(), "selection_sha256": sha_file(a.dest / "sealed_selection.json"), "base_hash": base_hash}, a.dest / "residual_head.pt")
    # Selection is now sealed; only now may target M10/query coordinates be opened.
    model_target, _src2, _receipt2, target_ds = load_base(a, include_target=True)
    strict_score = json.loads((a.b_dest / "score_receipt.json").read_text())
    if strict_score.get("status") != "COMPLETED" or strict_score.get("checkpoint_sha256") != receipt.get("checkpoint_sha256"):
        raise RuntimeError("strict B score receipt is not bound to completed B training")
    npz_path = a.b_dest / "target_selected_predictions.npz"
    target_audit = strict_score.get("target_audit_arrays", {}).get("selected")
    if not isinstance(target_audit, dict) or target_audit.get("file") != npz_path.name or target_audit.get("sha256") != sha_file(npz_path):
        raise RuntimeError("strict B score receipt does not bind selected target NPZ filename/SHA")
    with np.load(npz_path, allow_pickle=False) as z: saved = {k: np.ascontiguousarray(z[k]) for k in z.files}
    required = {"target", "prediction", "window_start_padded", "output_index_padded", "output_index_query_relative", "prefix_bins"}
    if set(saved) != required: raise RuntimeError("target selected NPZ is not strict m1_score coordinate schema")
    starts: list[np.ndarray] = []; targets: list[np.ndarray] = []
    for _x, y, ss, _v, st in m1_train.loader(target_ds, 42, shuffle=False):
        if set(ss) != {a.target}: raise RuntimeError("target dataset session drift")
        starts.append(st.numpy()); targets.append(y[:, -1].numpy())
    current_start, current_target = np.concatenate(starts), np.concatenate(targets)
    prefix = int(np.asarray(target_ds.trial_start_indices[a.target])[0])
    if not (np.array_equal(saved["window_start_padded"], current_start) and np.array_equal(saved["target"], current_target) and np.array_equal(saved["prefix_bins"], np.asarray([prefix], dtype=np.int64)) and np.array_equal(saved["output_index_padded"], current_start + target_ds.window_size - 1) and np.array_equal(saved["output_index_query_relative"], current_start + target_ds.window_size - 1 - prefix)):
        raise RuntimeError("saved strict B prediction coordinates/labels do not match target dataset")
    target_carrier, target_meta = aligned_carrier.project_target_carrier(m1_data.nwb_path(a.target), frozen)
    target_grid = aligned_carrier.causal_carrier_projection(np.asarray(target_ds.neural_data[a.target], dtype=np.float32), target_carrier)
    feature = torch.from_numpy(target_grid[current_start + target_ds.window_size - 1])
    wrapper.eval()
    with torch.inference_mode():
        residual = wrapper.forward_cached(torch.from_numpy(saved["prediction"]), feature, carrier_visible=torch.ones(len(feature), dtype=torch.bool)) - torch.from_numpy(saved["prediction"])
        prediction = torch.from_numpy(saved["prediction"]) + float(best["alpha"]) * residual
    candidate = prediction.numpy(); metric_b = float(variance_weighted_r2(saved["target"], saved["prediction"])); metric_c = float(variance_weighted_r2(saved["target"], candidate))
    atomic_npz(a.dest / "target_candidate_predictions.npz", target=saved["target"], b_prediction=saved["prediction"], prediction=candidate, feature=feature.numpy(), window_start_padded=saved["window_start_padded"], output_index_padded=saved["output_index_padded"], output_index_query_relative=saved["output_index_query_relative"], prefix_bins=saved["prefix_bins"])
    atomic_json(a.dest / "fit_score_receipt.json", {**protocol(a), "status": "COMPLETED", "base_hash": base_hash, "source_cache_sha256": sha_file(cache_path), "selection_sha256": sha_file(a.dest / "sealed_selection.json"), "residual_checkpoint_sha256": sha_file(a.dest / "residual_head.pt"), "source_carrier_sha256": frozen_sha, "aligned_source_support": frozen.raw_support_meta, "aligned_source_unit_id_sha256": dict(frozen.source_unit_id_sha256), "target_coordinate_sha256": sha_array(saved["window_start_padded"]), "target_optimizer_steps": 0, "target_labels_used_for_fit_or_selection": False, "target_labels_only_for_final_metric_and_coordinate_gate": True, "target": {"b_r2": metric_b, "candidate_r2": metric_c, "delta_r2": metric_c - metric_b, "carrier": target_meta, "baseline_runtime": "saved original strict B NPZ; intentionally not recomputed on CPU"}, "source_selected": best, "source_curve": curve, "source_validation_cpu_baseline_recomputed": baseline, "cpu_vs_prior_gpu_note": "source selection is CPU float32 and may differ slightly from GPU floating-point evaluation; target B is the saved original NPZ exactly", "strict_b_score_receipt_sha256": sha_file(a.b_dest / "score_receipt.json"), "target_npz_sha256": sha_file(npz_path)})
    print(json.dumps({"stage": "fit_score", "status": "COMPLETED", "selection": best,
                      "target_delta_r2": metric_c - metric_b}, sort_keys=True), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=m1_data.FOLDS, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--b-dest", type=Path, required=True, help="completed original B_ACTIVITY_ONLY run directory")
    p.add_argument("--stage", choices=("preflight", "fit_score"), required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--train-stride", type=int, choices=(8, 16), default=8)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.seed != 42 or a.epochs != EPOCHS or a.threads < 1: raise ValueError("pilot protocol requires seed=42, epochs=20, and positive CPU threads")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    a.dest = a.dest.resolve(); a.b_dest = a.b_dest.resolve(); a.dest.mkdir(parents=True, exist_ok=True)
    if (a.dest / "protocol.json").exists() and json.loads((a.dest / "protocol.json").read_text()) != protocol(a): raise RuntimeError("existing destination protocol differs")
    atomic_json(a.dest / "protocol.json", protocol(a)); configure_cpu(a.threads)
    {"preflight": preflight, "fit_score": fit_score}[a.stage](a)

if __name__ == "__main__": main()
