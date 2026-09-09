#!/usr/bin/env python3
"""Sealed ext6 EMA epoch picker for the M2 RIFT concat vstate4 run.

Same selection rule as the 581973-class ext6-epochpick: six locally visible
official held-out-calib sessions, EMA view, all 24 epochs, unweighted
six-session arithmetic R2 mean, earliest maximum.  Selection surface is the
official query bytes (X/target/starts/mapping must hash-equal the sealed
official ext6 query cache); only the identity banks are the vstate4 rebuild.

Never opens an NWB, never contacts EvalAI, never mutates the official caches.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
V1 = WORKSPACE / "btransform_unified_v1"
for p in (ROOT, ROOT / "src", ROOT / "scripts" / "rift_v1", V1 / "src", WORKSPACE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.r2 import variance_weighted_r2
from scripts.rift_v1 import m2_concat_train as concat
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data, plan as old_plan

SIX = (
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
)
EPOCHS = tuple(range(1, 25))
VIEW = "EMA"
QUERY_CACHE = ROOT / "results/carrier_v3_m2/run_vstate4/cache/ext6"
OFFICIAL_QUERY = WORKSPACE / "tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1/official_heldout_query"
RECEIPT_SCHEMA = "m2_vstate4_ext6_official_heldout_query_v2"
BASELINE_PICK = ROOT / "results/rift_v1/m2_r50_concat_s42_ext6_pick_v1"
BASELINE_ANCHORS = {
    "baseline_concat_ext6_score_receipt_sha256": "6a985e1f3fb9392198725d6920efb7ea24b41052ddae84a335ae66ea72022e31",
    "baseline_concat_ext6_epoch": 9,
    "baseline_concat_ext6_equal_session_mean": 0.3900576650553506,
    "baseline_concat_official_582189_held_out_r2_mean": 0.34654225938843214,
    "pick_method_precedent_official_581973_held_out_r2_mean": 0.3903054960458745,
    "gate": "recommend only if equal_session_mean >= baseline_concat_ext6_equal_session_mean + 0.03",
}
DISCLOSURE = {
    "dev_on_official_selected": (
        "epoch selected on the same six locally visible official held-out-calib sessions "
        "(query_start_trial=0) used by the 581973-class ext6-epochpick; this is a development "
        "selection surface, not the official held-out score"
    ),
    "e0_covariant": (
        "E0 is remelted through the frozen encoder conditioned on the vstate4 T; the ext4 gain "
        "(+0.0366 vs baseline ext4 0.3826) is the joint effect of the T carrier swap and the "
        "E0 recomputation, not T alone"
    ),
    "single_seed": "single run, seed 42, no seed ensemble",
    "dense_velocity_label_carrier": (
        "vstate4 consumes dense velocity labels of calibration trials (Falcon-allowed support "
        "labels); plan doc PLAN_CARRIER_ITERATION_M2_688_20260909.md section 3.2 disclosure: "
        "\u62ab\u9732\uff1a\u6d88\u8d39\u6821\u51c6 trial \u7684 dense \u901f\u5ea6\u6807\u7b7e\uff08FALCON \u5141\u8bb8\u7684 support \u6807\u7b7e\uff09\uff1b"
        "\u8fd9\u4e0e\u65e7\u6587\u6863\u4e2d 688 \u7684\"sparse-label\"\u7eaa\u5f8b\u4e0d\u662f\u540c\u4e00\u4e2a\u95ee\u9898\u3002"
    ),
}
SCHEMA = "m2_rift_vstate4_ext6_epoch_pick_v1"


def sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _seal(path: Path) -> None:
    os.chmod(path, 0o444)


def _source_paths() -> tuple[Path, ...]:
    import btransform_unified_v2.concat_model as concat_model
    import btransform_unified_v2.model as rift_model
    import btransform_unified_v2.temporal as temporal
    import btransform_unified_v1.bank as bank
    import btransform_unified_v1.r2 as r2
    import tfpd_exploration.src.m2_dual_track_v1.champion as champion
    import tfpd_exploration.src.m2_hold_film_probe_v1.encoder as encoder
    import btransform_unified_v2.carrier_profile_v3 as v3
    return (Path(__file__), Path(concat.__file__), Path(concat_model.__file__),
            Path(rift_model.__file__), Path(temporal.__file__), Path(bank.__file__), Path(r2.__file__),
            Path(old_data.__file__), Path(old_plan.__file__), Path(champion.__file__), Path(encoder.__file__),
            Path(v3.__file__))


def _assert_no_688(paths: Mapping[str, str]) -> None:
    bad = [path for path in paths if "688" in path]
    if bad:
        raise RuntimeError(f"DANDI 688 is forbidden by ext6 picker: {bad}")


def query_asset_hashes(cache: Path) -> dict[str, Any]:
    receipt = cache / "ext6_query_receipt.json"
    doc = json.loads(receipt.read_text(encoding="utf-8"))
    if doc.get("schema") != RECEIPT_SCHEMA or doc.get("status") != "COMPLETED":
        raise RuntimeError("unrecognized vstate4 ext6 query receipt")
    if doc.get("hidden_or_test_opened") is not False or doc.get("evalai_opened") is not False:
        raise RuntimeError("ext6 bank receipt is not read-only/non-hidden")
    if set(doc.get("sessions", {})) != set(SIX):
        raise RuntimeError("ext6 cache does not contain exactly six sessions")
    out: dict[str, Any] = {"receipt_sha256": sha(receipt), "sessions": {}}
    for session in SIX:
        d = cache / session
        files = ("mapping.json", "e0_u.pt", "T.npy", "eligible_starts.npy", "X_store.npy", "target_store.npy")
        if not d.is_dir():
            raise RuntimeError(f"missing ext6 query session {session}")
        file_hashes = {name: sha(d / name) for name in files}
        mapping = json.loads((d / "mapping.json").read_text())
        expected = doc["sessions"][session]
        if (mapping.get("query_is_padded_timeline") is not True or int(mapping.get("query_pad_bins", -1)) != 49 or
                tuple(mapping.get("support_trial_ids", ())) != tuple(range(33)) or
                int(expected.get("query_start_trial", -1)) != 0):
            raise RuntimeError(f"{session}: frozen query_start_trial=0/padding/M33 contract drift")
        starts = np.load(d / "eligible_starts.npy", mmap_mode="r")
        if starts.ndim != 1 or not len(starts) or np.any(np.diff(starts) < 0):
            raise RuntimeError(f"{session}: malformed eligible starts")
        if int(expected.get("window_count", -1)) != int(len(starts)):
            raise RuntimeError(f"{session}: frozen window count drift")
        # The model-independent query bytes must be the sealed official surface.
        official = OFFICIAL_QUERY / session
        for name in ("mapping.json", "eligible_starts.npy", "X_store.npy", "target_store.npy"):
            if file_hashes[name] != sha(official / name):
                raise RuntimeError(f"{session}/{name}: query bytes drift from official ext6 surface")
        out["sessions"][session] = {
            "files": file_hashes, "window_count": int(len(starts)),
            "eligible_starts_sha256": hashlib.sha256(np.asarray(starts, dtype=np.int64).tobytes()).hexdigest(),
            "query_start_trial": 0,
            "official_query_bytes_equal": True,
        }
    return out


def baseline_anchor_binding() -> dict[str, Any]:
    actual = sha(BASELINE_PICK / "score_receipt.json")
    if actual != BASELINE_ANCHORS["baseline_concat_ext6_score_receipt_sha256"]:
        raise RuntimeError("baseline concat ext6 score receipt sha drift")
    return {
        **BASELINE_ANCHORS,
        "baseline_concat_ext6_pick_dir": str(BASELINE_PICK),
        "baseline_selected_ema_sha256": sha(BASELINE_PICK / "selected_ema.pt"),
        "official_query_cache_receipt_sha256": sha(OFFICIAL_QUERY / "official_heldout_query_banks.json"),
    }


def load_query_pair(session: str, cache: Path) -> tuple[Any, TaskBank]:
    from tfpd_exploration.src.m2_dual_track_v1 import contracts as old_contracts
    d = cache / session
    mapping = json.loads((d / "mapping.json").read_text())
    e0_obj = torch.load(d / "e0_u.pt", map_location="cpu", weights_only=False)
    e0 = np.ascontiguousarray(e0_obj["E0"].detach().cpu().numpy(), dtype=np.float32)
    carrier = np.ascontiguousarray(np.load(d / "T.npy"), dtype=np.float32)
    starts = np.ascontiguousarray(np.load(d / "eligible_starts.npy"), dtype=np.int64)
    x_store, targets = old_data.read_memmap(d / "X_store.npy"), old_data.read_memmap(d / "target_store.npy")
    dual = old_contracts.SessionBank(session_id=session, support_trial_ids=tuple(mapping["support_trial_ids"]),
        raw_trial_ids=tuple(mapping["raw_trial_ids"]), X_store=x_store, target_store=targets,
        eligible_starts=starts, E0=torch.from_numpy(e0.copy()), T=torch.from_numpy(carrier.copy()),
        unit_mask=torch.ones(old_plan.CHANNELS, dtype=torch.bool), provenance={"surface": "official_heldout_query_vstate4", "session_id": session})
    store = np.asarray(x_store)
    x3 = np.ascontiguousarray(store[:1] if store.ndim == 3 else store[:old_plan.WINDOW][None, ...], dtype=np.float32)
    t3 = np.ascontiguousarray(np.asarray(targets)[:1], dtype=np.float32)
    bank = TaskBank(session_id=session, E0=e0, carrier=carrier, unit_mask=np.ones(old_plan.CHANNELS, dtype=np.bool_),
        X_store=x3, target_store=t3, window_ids=starts[:1],
        calibration_meta={"shape": tuple(e0.shape), "trial_count": 33, "estimator": "frozen official query vstate4 bank",
                          "budget": 33, "surface": "official_heldout_query_vstate4", "cache_root": str(d),
                          "array_sha256": array_sha256(e0),
                          "carrier_sha256": hashlib.sha256(carrier.tobytes()).hexdigest(), "query_start_trial": 0})
    return dual, bank


def _valid(starts: tuple[int, ...], device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(starts, dtype=torch.long, device=device)[:, None]
    return x + torch.arange(50, device=device)[None, :] >= 49


def _model_and_loader(run_meta: Mapping[str, Any], device: torch.device):
    cell = str(run_meta.get("cell", "")); schema = str(run_meta.get("schema", ""))
    if cell == concat.CELL and schema == "m2_rift_concat_train_v1":
        return "concat", concat._decoder(device)
    raise RuntimeError(f"unsupported formal frontend: cell={cell!r} schema={schema!r}")


def _load_ema(model: torch.nn.Module, checkpoint: Mapping[str, Any], frontend: str) -> None:
    key = "raw_state_dict" if frontend == "concat" else "model"
    model.load_state_dict(checkpoint[key])
    shadow = checkpoint["ema"].get("shadow")
    if not isinstance(shadow, Mapping):
        raise RuntimeError("checkpoint has no EMA shadow")
    with torch.no_grad():
        named = dict(model.named_parameters())
        if set(named) != set(shadow):
            raise RuntimeError("EMA parameter keys drift")
        for name, value in named.items():
            value.copy_(shadow[name].to(value.device, value.dtype))


def score(model: torch.nn.Module, duals: Mapping[str, Any], banks: Mapping[str, TaskBank], device: torch.device) -> dict[str, Any]:
    model.eval(); rows: dict[str, Any] = {}; all_t: list[np.ndarray] = []; all_p: list[np.ndarray] = []
    for session in SIX:
        pred, target = [], []
        for batch in old_data.iter_session_batches(duals[session], batch_size=32, device=device, target_space=old_plan.SCORING_TARGET_SPACE):
            with torch.inference_mode():
                raw = model(batch.X, banks[session], input_valid_mask=_valid(batch.window_ids, device))
            p = np.ascontiguousarray(raw.float().cpu().numpy() / old_plan.BEHAVIOR_SCALE, dtype=np.float32)
            t = np.ascontiguousarray(batch.last_target.float().cpu().numpy(), dtype=np.float32)
            pred.append(p); target.append(t)
        if not pred:
            raise RuntimeError(f"{session}: no scored query batches")
        p, t = np.concatenate(pred), np.concatenate(target)
        if not np.isfinite(p).all() or not np.isfinite(t).all():
            raise RuntimeError(f"{session}: non-finite prediction or target")
        r2 = float(variance_weighted_r2(t, p))
        if not math.isfinite(r2):
            raise RuntimeError(f"{session}: non-finite R2")
        rows[session] = {"r2": r2, "window_count": int(len(t)), "prediction_sha256": hashlib.sha256(p.tobytes()).hexdigest()}
        all_p.append(p); all_t.append(t)
    mean = float(np.mean([rows[s]["r2"] for s in SIX]))
    pooled = float(variance_weighted_r2(np.concatenate(all_t), np.concatenate(all_p)))
    if not math.isfinite(mean) or not math.isfinite(pooled):
        raise RuntimeError("non-finite ext6 aggregate")
    return {"view": VIEW, "partial": False, "per_session": rows, "equal_session_mean": mean, "pooled_r2": pooled,
            "n_windows": int(sum(x["window_count"] for x in rows.values()))}


def validate_complete_report(report: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    per = report.get("per_session")
    if report.get("view") != VIEW or report.get("partial") is not False or not isinstance(per, Mapping) or set(per) != set(SIX):
        raise RuntimeError("progress row is not a complete six-session EMA result")
    if not all(math.isfinite(float(report.get(k, float("nan")))) for k in ("equal_session_mean", "pooled_r2")):
        raise RuntimeError("progress aggregate is non-finite")
    for session in SIX:
        row = per[session]
        if (int(row.get("window_count", -1)) != int(expected[session]["window_count"]) or
                not math.isfinite(float(row.get("r2", float("nan")))) or not isinstance(row.get("prediction_sha256"), str)):
            raise RuntimeError(f"progress row is incomplete for {session}")
    if int(report.get("n_windows", -1)) != sum(int(expected[s]["window_count"]) for s in SIX):
        raise RuntimeError("progress total window count drift")
    recomputed = float(np.mean([float(per[s]["r2"]) for s in SIX]))
    if abs(float(report["equal_session_mean"]) - recomputed) > 1.0e-12:
        raise RuntimeError("progress equal-session mean is not the six-session arithmetic mean")


def validate_complete_curve(completed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if set(completed) != {str(epoch) for epoch in EPOCHS}:
        raise RuntimeError("incomplete 24-epoch curve cannot select/package")
    for epoch in EPOCHS:
        report = completed[str(epoch)]
        if not isinstance(report, Mapping) or not isinstance(report.get("checkpoint_sha256"), str):
            raise RuntimeError(f"epoch {epoch}: missing sealed checkpoint report")
        validate_complete_report(report, expected)


def _checkpoint_contract(state: Mapping[str, Any], meta: Mapping[str, Any], epoch: int, frontend: str) -> None:
    if (int(state.get("epoch", -1)) != epoch or int(state.get("global_step", -1)) != epoch * 3165 or
            state.get("smoke") is not False or int(state.get("epochs", -1)) != 24):
        raise RuntimeError(f"epoch {epoch}: checkpoint not formal e1..e24")
    if frontend == "concat":
        required = {"schema": "m2_rift_concat_epoch_checkpoint_v1", "cell": concat.CELL, "context_bins": 50,
                    "attention_backend": "local", "identity_interface": "concat", "bias_mode": "recency"}
    else:
        raise RuntimeError("unsupported frontend")
    identity = {"seed": meta.get("seed")}
    if (any(state.get(k) != v for k, v in required.items()) or
            any(state.get(k) != v for k, v in identity.items()) or
            state.get("source_hashes") != meta.get("source_hashes") or
            state.get("frozen_cache_hashes") != meta.get("frozen_cache_hashes")):
        raise RuntimeError(f"epoch {epoch}: checkpoint contract/source drift")


def manifest_for(run: Path, cache: Path) -> dict[str, Any]:
    meta_path, receipt_path = run / "run_meta.json", run / "train_receipt.json"
    meta = json.loads(meta_path.read_text()); receipt = json.loads(receipt_path.read_text())
    required_receipt = {"cell": meta.get("cell"), "epochs": 24, "global_step": 24 * 3165, "status": "COMPLETED",
                        "schema": "m2_rift_concat_train_receipt_v1"}
    if int(meta.get("seed", -1)) != 42:
        raise RuntimeError("formal M2 epoch picker requires seed 42")
    if (meta.get("status") != "FORMAL" or any(receipt.get(k) != v for k, v in required_receipt.items()) or
            receipt.get("source_hashes") != meta.get("source_hashes") or
            receipt.get("frozen_cache_hashes") != meta.get("frozen_cache_hashes")):
        raise RuntimeError("requires completed formal run and train receipt")
    for path, digest in meta.get("source_hashes", {}).items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"formal training source drift: {path}")
    ckpts = {str(e): {"path": str(run / f"epoch_{e:03d}.pt"), "sha256": sha(run / f"epoch_{e:03d}.pt")} for e in EPOCHS}
    src = {str(p): sha(p) for p in _source_paths()}
    _assert_no_688(src)
    return {"schema": SCHEMA + "_manifest", "run": str(run), "run_meta_sha256": sha(meta_path),
            "train_receipt_sha256": sha(receipt_path),
            "cell": meta.get("cell"), "run_schema": meta.get("schema"), "seed": meta.get("seed"), "view": VIEW,
            "epochs": list(EPOCHS), "checkpoint_bytes": ckpts, "query_cache": str(cache),
            "query_assets": query_asset_hashes(cache), "baseline_anchors": baseline_anchor_binding(),
            "disclosure": DISCLOSURE, "source_sha256": src,
            "official_test_used": False, "evalai_opened": False, "nwb_opened": False,
            "created_utc": datetime.now(timezone.utc).isoformat()}


def verify_manifest(manifest: Mapping[str, Any], run: Path, cache: Path) -> None:
    fresh = manifest_for(run, cache)
    for key in fresh:
        if key != "created_utc" and fresh[key] != manifest.get(key):
            raise RuntimeError(f"sealed manifest drift: {key}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run.resolve(); dest = args.dest.resolve(); cache = args.query_cache.resolve()
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", ""):
        raise RuntimeError("this picker is pinned to GPU0")
    candidate = manifest_for(run_dir, cache)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise RuntimeError("existing destination requires explicit --resume")
    if args.resume and not manifest_path.exists():
        raise RuntimeError("--resume requires an existing sealed manifest")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_manifest(manifest, run_dir, cache)
    else:
        manifest = candidate
        atomic_json(manifest_path, manifest)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    if (dest / "score_receipt.json").exists():
        raise RuntimeError("completed destination cannot be repeated")
    duals, banks = zip(*(load_query_pair(s, cache) for s in SIX))
    dual_map = dict(zip(SIX, duals)); bank_map = dict(zip(SIX, banks))
    device = torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    frontend, model = _model_and_loader(meta, device)
    progress_path = dest / "score_progress.json"
    if args.resume and not progress_path.exists():
        raise RuntimeError("--resume requires existing score progress")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"schema": SCHEMA + "_progress", "manifest_sha256": canonical_digest(manifest), "completed": {}}
    if progress.get("manifest_sha256") != canonical_digest(manifest):
        raise RuntimeError("progress is from another manifest")
    for epoch in EPOCHS:
        cp = Path(manifest["checkpoint_bytes"][str(epoch)]["path"]); cp_sha = sha(cp)
        prior = progress["completed"].get(str(epoch))
        if prior:
            if prior.get("checkpoint_sha256") != cp_sha or prior.get("partial") is not False:
                raise RuntimeError(f"epoch {epoch}: progress drift/partial")
            validate_complete_report(prior, manifest["query_assets"]["sessions"])
            print(f"epoch {epoch}: resumed {prior['equal_session_mean']:.4f}", flush=True)
            continue
        state = torch.load(cp, map_location=device, weights_only=False)
        _checkpoint_contract(state, meta, epoch, frontend)
        _load_ema(model, state, frontend)
        report = score(model, dual_map, bank_map, device)
        validate_complete_report(report, manifest["query_assets"]["sessions"])
        progress["completed"][str(epoch)] = {**report, "checkpoint_sha256": cp_sha}
        atomic_json(progress_path, progress)
        print(f"epoch {epoch}: equal_session_mean={report['equal_session_mean']:.6f} pooled={report['pooled_r2']:.6f}", flush=True)
    validate_complete_curve(progress["completed"], manifest["query_assets"]["sessions"])
    values = {e: float(progress["completed"][str(e)]["equal_session_mean"]) for e in EPOCHS}
    best = max(EPOCHS, key=lambda e: (values[e], -e))
    selected = progress["completed"][str(best)]
    verify_manifest(manifest, run_dir, cache)
    selected_state = torch.load(Path(manifest["checkpoint_bytes"][str(best)]["path"]), map_location="cpu", weights_only=False)
    _, package_model = _model_and_loader(meta, torch.device("cpu"))
    _load_ema(package_model, selected_state, frontend)
    ema_only = {name: value.detach().cpu().clone() for name, value in package_model.state_dict().items()}
    package = dest / "selected_ema.pt"
    if package.exists():
        prior = torch.load(package, map_location="cpu", weights_only=True)
        if set(prior) != set(ema_only) or any(not torch.equal(prior[k], ema_only[k]) for k in ema_only):
            raise RuntimeError("selected EMA package exists but differs from sealed selected candidate")
    else:
        torch.save(ema_only, package)
    gate_value = float(values[best])
    baseline = float(BASELINE_ANCHORS["baseline_concat_ext6_equal_session_mean"])
    audit = {
        "schema": SCHEMA + "_validation_audit", "status": "COMPLETED",
        "manifest_sha256": canonical_digest(manifest),
        "selection": {"rule": "earliest maximum finite unweighted equal_session_mean", "epoch": best,
                      "equal_session_mean": gate_value},
        "alignment": {
            "vs_baseline_concat_ext6_e9": gate_value - baseline,
            "vs_official_581973_precedent": gate_value - BASELINE_ANCHORS["pick_method_precedent_official_581973_held_out_r2_mean"],
            "vs_baseline_official_582189": gate_value - BASELINE_ANCHORS["baseline_concat_official_582189_held_out_r2_mean"],
            "baseline_ext4_best_epoch16": 0.4192012408277416, "baseline_ext4_mean": 0.3826,
            "note": "ext6 dev deltas are against dev ext6 anchors; official 581973/582189 are held-out scores of different lines and are context only",
        },
        "recommendation_gate": {"threshold": baseline + 0.03, "value": gate_value,
                                 "passes": gate_value >= baseline + 0.03},
        "artifact_sha256": {"manifest.json": sha(manifest_path),
                            "score_progress.json": sha(progress_path),
                            "selected_ema.pt": sha(package)},
        "disclosure": DISCLOSURE,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt = {"schema": SCHEMA + "_selection", "status": "COMPLETED",
               "manifest_sha256": canonical_digest(manifest), "view": VIEW,
               "selection": {"rule": "earliest maximum finite unweighted equal_session_mean", "epoch": best,
                             "equal_session_mean": values[best]},
               "selected": selected,
               "selected_ema_state": {"path": str(package), "sha256": sha(package), "raw_state_serialized": False},
               "ema_by_epoch": progress["completed"],
               "baseline_anchors": manifest["baseline_anchors"], "disclosure": DISCLOSURE,
               "evalai_opened": False, "official_test_used": False,
               "utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(dest / "score_receipt.json", receipt)
    atomic_json(dest / "selected_ema_receipt.json", receipt)
    atomic_json(dest / "validation_audit.json", audit)
    for name in ("manifest.json", "score_receipt.json", "selected_ema_receipt.json", "validation_audit.json", "score_progress.json"):
        _seal(dest / name)
    return receipt


def main() -> int:
    p = argparse.ArgumentParser(description="sealed M2 RIFT vstate4 ext6 EMA epoch picker")
    p.add_argument("--run-dir", "--run", dest="run", type=Path,
                   default=ROOT / "results/carrier_v3_m2/train_vstate4")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--query-cache", type=Path, default=QUERY_CACHE)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.cpu_threads < 1:
        p.error("thread count must be positive")
    print(json.dumps(run(a), indent=2, sort_keys=True, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
