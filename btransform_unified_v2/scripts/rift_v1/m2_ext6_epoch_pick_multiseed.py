#!/usr/bin/env python3
"""Immutable multi-seed ext6 EMA epoch picker for formal M2 RIFT runs.

This is deliberately separate from the train and ext4-score runners.  It reads
only the frozen ``query_start_trial=0`` ext6 query cache, never opens an NWB,
and never contacts EvalAI.  A run is sealed to its 24 checkpoint bytes, the
query-builder bytes, and every query/cache asset before any candidate is read.

Concat and joint B/D have different frontends.  Concat consumes the frozen E0
bank.  Joint B/D must be given an independently frozen raw-M33 directory with
``<session>/calib_activity.npy``; E0 is never substituted for its live M33
identity provider.
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
for p in (ROOT, ROOT / "src", V1 / "src", WORKSPACE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.r2 import variance_weighted_r2
from scripts.rift_v1 import m2_concat_train as concat
from tfpd_exploration.src.m2_dual_track_v1 import contracts as old_contracts
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan

SIX = (
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
)
EPOCHS = tuple(range(1, 25))
JOINT_SEEDS = (42, 43, 44)
VIEW = "EMA"
QUERY_CACHE = WORKSPACE / "tfpd_exploration/results/m2_small_s1_visible_ext6_epoch_pick_v1/official_heldout_query"
SEALED_BT_ORACLE = ROOT / "results/rift_v1/m2_ext6_query_verify_v1/receipt.json"
SEALED_BT_ORACLE_SOURCE = ROOT / "scripts/rift_v1/verify_m2_ext6_query.py"
SEALED_BT_PAYLOAD = WORKSPACE / "tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"
SCHEMA = "m2_rift_ext6_epoch_pick_multiseed_v1"


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


def _source_paths() -> tuple[Path, ...]:
    # These are the complete code paths able to shape query inputs or scores.
    import btransform_unified_v2.concat_model as concat_model
    import btransform_unified_v2.joint_m2_model as joint_model
    import btransform_unified_v2.model as rift_model
    import btransform_unified_v2.temporal as temporal
    import btransform_unified_v1.bank as bank
    import btransform_unified_v1.r2 as r2
    import tfpd_exploration.src.m2_dual_track_v1.champion as champion
    import tfpd_exploration.src.m2_hold_film_probe_v1.encoder as encoder
    return (Path(__file__), Path(concat.__file__), Path(concat_model.__file__),
            Path(joint_model.__file__), Path(rift_model.__file__), Path(temporal.__file__), Path(bank.__file__), Path(r2.__file__),
            Path(old_data.__file__), Path(old_contracts.__file__), Path(old_plan.__file__),
            Path(champion.__file__), Path(encoder.__file__))


def _assert_no_688(paths: Mapping[str, str]) -> None:
    bad = [path for path in paths if "688" in path]
    if bad:
        raise RuntimeError(f"DANDI 688 is forbidden by ext6 picker: {bad}")


def query_asset_hashes(cache: Path) -> dict[str, Any]:
    receipt = cache / "official_heldout_query_banks.json"
    doc = json.loads(receipt.read_text(encoding="utf-8"))
    if doc.get("schema") != "m2_small_s1_visible_ext6_official_heldout_query_v1":
        raise RuntimeError("unrecognized ext6 query receipt")
    if doc.get("hidden_or_test_opened") is not False or doc.get("mutated_dual_track_cache") is not False:
        raise RuntimeError("ext6 cache receipt is not read-only/non-hidden")
    if set(doc.get("sessions", {})) != set(SIX):
        raise RuntimeError("ext6 cache does not contain exactly six sessions")
    out: dict[str, Any] = {"receipt_sha256": sha(receipt), "sessions": {}}
    for session in SIX:
        d = cache / session
        files = ("mapping.json", "e0_u.pt", "T.npy", "eligible_starts.npy", "X_store.npy", "target_store.npy")
        if not d.is_dir(): raise RuntimeError(f"missing ext6 query session {session}")
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
        out["sessions"][session] = {"files": file_hashes, "window_count": int(len(starts)),
                                    "eligible_starts_sha256": hashlib.sha256(np.asarray(starts, dtype=np.int64).tobytes()).hexdigest(),
                                    "query_start_trial": 0}
    return out


def sealed_bt_oracle_binding(cache: Path, assets: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the mandatory sealed BT proof to these exact query bytes.

    This is evidence only for the independently packed ORT payload; it does
    not select an epoch.  Its receipt must nevertheless be sealed into every
    scorer manifest so a later cache or payload substitution is rejected.
    """
    receipt = json.loads(SEALED_BT_ORACLE.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "m2_ext6_sealed_bt_query_oracle_v1" or
        receipt.get("status") != "COMPLETED" or
        receipt.get("official_test_opened") is not False or receipt.get("evalai_opened") is not False or
        receipt.get("payload") != str(SEALED_BT_PAYLOAD) or receipt.get("payload_sha256") != sha(SEALED_BT_PAYLOAD) or
        receipt.get("query_cache") != str(cache) or receipt.get("query_cache_receipt_sha256") != assets["receipt_sha256"]):
        raise RuntimeError("sealed BT oracle receipt/payload/query-cache binding drift")
    expected = {"ses-2020-11-24-Run1", "ses-2020-11-24-Run2"}
    rows = receipt.get("sessions")
    if not isinstance(rows, list) or {row.get("session") for row in rows} != expected:
        raise RuntimeError("sealed BT oracle does not cover both Nov24 query sessions")
    bound: dict[str, Any] = {}
    for row in rows:
        session = str(row["session"]); directory = cache / session
        starts = np.load(directory / "eligible_starts.npy", mmap_mode="r")
        targets = np.load(directory / "target_store.npy", mmap_mode="r")
        expected_ordinals = [0, len(starts) - 1]
        proof = row.get("oracle", {})
        checks = proof.get("checks", []) if isinstance(proof, Mapping) else []
        if (row.get("selected_ordinals") != expected_ordinals or row.get("window_count") != len(starts) or
            row.get("target_sha256") != hashlib.sha256(np.asarray(targets[expected_ordinals], dtype=np.float32).tobytes()).hexdigest() or
            proof.get("count") != 2 or len(checks) != 2 or
            any(check.get("wrong_start_fails_tolerance") is not True for check in checks)):
            raise RuntimeError(f"sealed BT oracle detail drift for {session}")
        bound[session] = {"target_store_sha256": row["target_sha256"], "prediction_sha256": row.get("prediction_sha256"),
                          "selected_ordinals": expected_ordinals}
    return {"receipt": str(SEALED_BT_ORACLE), "receipt_sha256": sha(SEALED_BT_ORACLE),
            "oracle_source": str(SEALED_BT_ORACLE_SOURCE), "oracle_source_sha256": sha(SEALED_BT_ORACLE_SOURCE),
            "payload": str(SEALED_BT_PAYLOAD), "payload_sha256": sha(SEALED_BT_PAYLOAD), "sessions": bound}


def load_query_pair(session: str, cache: Path) -> tuple[old_contracts.SessionBank, TaskBank]:
    d = cache / session; mapping = json.loads((d / "mapping.json").read_text())
    e0_obj = torch.load(d / "e0_u.pt", map_location="cpu", weights_only=False)
    e0 = np.ascontiguousarray(e0_obj["E0"].detach().cpu().numpy(), dtype=np.float32)
    carrier = np.ascontiguousarray(np.load(d / "T.npy"), dtype=np.float32)
    starts = np.ascontiguousarray(np.load(d / "eligible_starts.npy"), dtype=np.int64)
    x_store, targets = old_data.read_memmap(d / "X_store.npy"), old_data.read_memmap(d / "target_store.npy")
    dual = old_contracts.SessionBank(session_id=session, support_trial_ids=tuple(mapping["support_trial_ids"]),
        raw_trial_ids=tuple(mapping["raw_trial_ids"]), X_store=x_store, target_store=targets,
        eligible_starts=starts, E0=torch.from_numpy(e0.copy()), T=torch.from_numpy(carrier.copy()),
        unit_mask=torch.ones(old_plan.CHANNELS, dtype=torch.bool), provenance={"surface":"official_heldout_query", "session_id":session})
    # The RIFT concat decoder reads only E0/carrier/mask/session_id from this bank.
    # RIFT consults this object only for frozen identity/carry metadata.  Keep
    # one representative legal shape here; query batches always come from the
    # separate immutable SessionBank above, so no large duplicate is created.
    store = np.asarray(x_store)
    x3 = np.ascontiguousarray(store[:1] if store.ndim == 3 else store[:old_plan.WINDOW][None, ...], dtype=np.float32)
    t3 = np.ascontiguousarray(np.asarray(targets)[:1], dtype=np.float32)
    bank = TaskBank(session_id=session, E0=e0, carrier=carrier, unit_mask=np.ones(old_plan.CHANNELS, dtype=np.bool_),
        X_store=x3, target_store=t3, window_ids=starts[:1],
        calibration_meta={"shape":tuple(e0.shape), "trial_count":33, "estimator":"frozen official query M33 bank", "budget":33,
                          "surface":"official_heldout_query", "cache_root":str(d), "array_sha256":array_sha256(e0),
                          "carrier_sha256":hashlib.sha256(carrier.tobytes()).hexdigest(), "query_start_trial":0})
    return dual, bank


def _valid(starts: tuple[int, ...], device: torch.device) -> torch.Tensor:
    # Query cache coordinates are padded; values themselves are never inspected.
    x = torch.as_tensor(starts, dtype=torch.long, device=device)[:, None]
    return x + torch.arange(50, device=device)[None, :] >= 49


def _model_and_loader(run_meta: Mapping[str, Any], device: torch.device, banks: Mapping[str, TaskBank], joint_m33_root: Path | None):
    cell = str(run_meta.get("cell", "")); schema = str(run_meta.get("schema", ""))
    if cell == concat.CELL and schema == "m2_rift_concat_train_v1":
        return "concat", concat._decoder(device)
    if cell == "M2-RIFT-R50-D4-JOINT-FILM-M33-V1" and schema == "m2_rift_joint_train_v2":
        from btransform_unified_v2.joint_m2_model import JointM2RiftDecoder
        if joint_m33_root is None: raise RuntimeError("joint B/D requires --joint-m33-root with frozen six-session raw M33 cache")
        model = JointM2RiftDecoder(str(run_meta["arm"]), seed=int(run_meta["seed"])).to(device)
        # The model's joint frontend must receive raw M33, never cached E0.
        for session in SIX:
            raw = joint_m33_root / session / "calib_activity.npy"
            if not raw.is_file() or np.load(raw, mmap_mode="r").shape != (33, 100, 96):
                raise RuntimeError(f"joint raw M33 missing/drifted for {session}: {raw}")
        model.install_session_memory(banks, joint_m33_root)
        return "joint", model
    raise RuntimeError(f"unsupported formal frontend: cell={cell!r} schema={schema!r}")


def _load_ema(model: torch.nn.Module, checkpoint: Mapping[str, Any], frontend: str) -> None:
    key = "raw_state_dict" if frontend == "concat" else "model"
    model.load_state_dict(checkpoint[key])
    shadow = checkpoint["ema"].get("shadow")
    if not isinstance(shadow, Mapping): raise RuntimeError("checkpoint has no EMA shadow")
    with torch.no_grad():
        named = dict(model.named_parameters())
        if set(named) != set(shadow): raise RuntimeError("EMA parameter keys drift")
        for name, value in named.items(): value.copy_(shadow[name].to(value.device, value.dtype))


def score(model: torch.nn.Module, duals: Mapping[str, Any], banks: Mapping[str, TaskBank], device: torch.device, max_batches: int | None = None) -> dict[str, Any]:
    model.eval(); rows: dict[str, Any] = {}; all_t: list[np.ndarray] = []; all_p: list[np.ndarray] = []
    for session in SIX:
        pred, target = [], []
        for index, batch in enumerate(old_data.iter_session_batches(duals[session], batch_size=32, device=device, target_space=old_plan.SCORING_TARGET_SPACE)):
            if max_batches is not None and index >= max_batches: break
            with torch.inference_mode(): raw = model(batch.X, banks[session], input_valid_mask=_valid(batch.window_ids, device))
            p = np.ascontiguousarray(raw.float().cpu().numpy() / old_plan.BEHAVIOR_SCALE, dtype=np.float32)
            t = np.ascontiguousarray(batch.last_target.float().cpu().numpy(), dtype=np.float32)
            pred.append(p); target.append(t)
        if not pred: raise RuntimeError(f"{session}: no scored query batches")
        p, t = np.concatenate(pred), np.concatenate(target)
        if not np.isfinite(p).all() or not np.isfinite(t).all():
            raise RuntimeError(f"{session}: non-finite prediction or target")
        r2 = float(variance_weighted_r2(t, p))
        if not math.isfinite(r2): raise RuntimeError(f"{session}: non-finite R2")
        rows[session] = {"r2":r2, "window_count":int(len(t)), "prediction_sha256":hashlib.sha256(p.tobytes()).hexdigest()}
        all_p.append(p); all_t.append(t)
    mean = float(np.mean([rows[s]["r2"] for s in SIX])); pooled = float(variance_weighted_r2(np.concatenate(all_t), np.concatenate(all_p)))
    if not math.isfinite(mean) or not math.isfinite(pooled): raise RuntimeError("non-finite ext6 aggregate")
    return {"view":VIEW, "partial":max_batches is not None, "per_session":rows, "equal_session_mean":mean, "pooled_r2":pooled,
            "n_windows":int(sum(x["window_count"] for x in rows.values()))}


def validate_complete_report(report: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """A previous atomic row is usable only if it is a complete ext6 result."""
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
    """Selection is legal only for an intact, complete 24 × six-session curve."""
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
        required = {"schema":"m2_rift_concat_epoch_checkpoint_v1", "cell":concat.CELL, "context_bins":50, "attention_backend":"local", "identity_interface":"concat"}
        hashes = "source_hashes"
    else:
        required = {"schema":"m2_rift_joint_checkpoint_v2", "cell":"M2-RIFT-R50-D4-JOINT-FILM-M33-V1", "context_bins":50, "attention_backend":"local"}
        hashes = "source_hashes"
    identity = {"seed": meta.get("seed")}
    if frontend == "joint": identity["arm"] = meta.get("arm")
    cache_key = "frozen_cache_hashes" if frontend == "concat" else "cache_hashes"
    if (any(state.get(k) != v for k,v in required.items()) or
        any(state.get(k) != v for k,v in identity.items()) or
        state.get(hashes) != meta.get(hashes) or state.get(cache_key) != meta.get(cache_key)):
        raise RuntimeError(f"epoch {epoch}: checkpoint contract/source drift")


def manifest_for(run: Path, cache: Path, joint_m33_root: Path | None) -> dict[str, Any]:
    meta_path, receipt_path = run / "run_meta.json", run / "train_receipt.json"
    meta = json.loads(meta_path.read_text()); receipt = json.loads(receipt_path.read_text())
    cache_key = "frozen_cache_hashes" if meta.get("schema") == "m2_rift_concat_train_v1" else "cache_hashes"
    # The historical concat receipt schema did not serialize seed; its sealed
    # run_meta and every checkpoint do.  Joint receipts carry it directly.
    required_receipt = {"cell": meta.get("cell"), "epochs":24,
                        "global_step":24 * 3165, "status":"COMPLETED"}
    if meta.get("schema") == "m2_rift_concat_train_v1":
        required_receipt["schema"] = "m2_rift_concat_train_receipt_v1"
    elif meta.get("schema") == "m2_rift_joint_train_v2":
        required_receipt.update({"schema":"m2_rift_joint_train_receipt_v2", "arm":meta.get("arm"), "seed":meta.get("seed"), "sampler_seed":42})
    else: raise RuntimeError("unsupported run metadata schema")
    if meta.get("schema") == "m2_rift_concat_train_v1" and int(meta.get("seed", -1)) != 42:
        raise RuntimeError("concat ext6 picker permits only formal seed 42")
    if meta.get("schema") == "m2_rift_joint_train_v2" and int(meta.get("seed", -1)) not in JOINT_SEEDS:
        raise RuntimeError(f"joint ext6 picker requires seed in {JOINT_SEEDS}")
    if (meta.get("status") != "FORMAL" or any(receipt.get(k) != v for k,v in required_receipt.items()) or
        int(receipt.get("epochs", -1)) != 24 or receipt.get("source_hashes") != meta.get("source_hashes") or
        receipt.get(cache_key) != meta.get(cache_key)):
        raise RuntimeError("requires completed formal run and train receipt")
    # Training code is an authority: a later edit may not be silently resealed.
    for path, digest in meta.get("source_hashes", {}).items():
        if sha(Path(path)) != digest: raise RuntimeError(f"formal training source drift: {path}")
    ckpts = {str(e): {"path":str(run/f"epoch_{e:03d}.pt"), "sha256":sha(run/f"epoch_{e:03d}.pt")} for e in EPOCHS}
    src = {str(p):sha(p) for p in _source_paths()}; _assert_no_688(src)
    joint_hashes = None
    if joint_m33_root is not None:
        joint_hashes = {s:sha(joint_m33_root/s/"calib_activity.npy") for s in SIX}
    assets = query_asset_hashes(cache)
    return {"schema":SCHEMA+"_manifest", "run":str(run), "run_meta_sha256":sha(meta_path), "train_receipt_sha256":sha(receipt_path),
            "cell":meta.get("cell"), "run_schema":meta.get("schema"), "seed":meta.get("seed"), "view":VIEW, "epochs":list(EPOCHS),
            "checkpoint_bytes":ckpts, "query_cache":str(cache), "query_assets":assets,
            "sealed_bt_oracle":sealed_bt_oracle_binding(cache, assets), "source_sha256":src,
            "joint_m33_root":str(joint_m33_root) if joint_m33_root else None, "joint_m33_sha256":joint_hashes,
            "official_test_used":False, "evalai_opened":False, "created_utc":datetime.now(timezone.utc).isoformat()}


def verify_manifest(manifest: Mapping[str, Any], run: Path, cache: Path, joint_m33_root: Path | None) -> None:
    fresh = manifest_for(run, cache, joint_m33_root)
    # Timestamp is informational and intentionally excluded from the seal.
    for key in fresh:
        if key != "created_utc" and fresh[key] != manifest.get(key): raise RuntimeError(f"sealed manifest drift: {key}")


def preflight(args: argparse.Namespace, manifest: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    # CPU 2 threads, epochs 8/9, few real query windows; no partial result is eligible to select/package.
    # Repeat this required proof check at execution, rather than trusting a
    # neighboring old receipt after the manifest was written.
    if sealed_bt_oracle_binding(args.query_cache.resolve(), manifest["query_assets"]) != manifest.get("sealed_bt_oracle"):
        raise RuntimeError("sealed BT oracle binding is absent or drifted")
    device=torch.device("cpu"); torch.set_num_threads(2)
    duals, banks = zip(*(load_query_pair(s, args.query_cache) for s in SIX)); dual_map=dict(zip(SIX,duals)); bank_map=dict(zip(SIX,banks))
    frontend, model = _model_and_loader(meta, device, bank_map, args.joint_m33_root)
    checks=[]
    for epoch in (8,9):
        cp=Path(manifest["checkpoint_bytes"][str(epoch)]["path"]); state=torch.load(cp,map_location=device,weights_only=False)
        _checkpoint_contract(state,meta,epoch,frontend); _load_ema(model,state,frontend)
        report=score(model,dual_map,bank_map,device,max_batches=args.preflight_batches)
        for s in SIX:
            starts=np.load(args.query_cache/s/"eligible_starts.npy",mmap_mode="r"); mask=_valid((int(starts[0]),int(starts[-1])),device).cpu().numpy()
            if not (mask[0,0] == (int(starts[0]) >= 49) and mask[1,-1]): raise RuntimeError("padding endpoint oracle failed")
        checks.append({"epoch":epoch,"checkpoint_sha256":sha(cp),"report":report})
    return {"schema":SCHEMA+"_preflight", "status":"COMPLETED", "device":"cpu", "cpu_threads":2,"epochs":[8,9], "partial":True,
            "selection_or_packaging_allowed":False,"source_audit":{"source_sha256":manifest["source_sha256"],"no_688":True,"nwb_opened":False},
            "sealed_bt_oracle":manifest["sealed_bt_oracle"],"checks":checks}


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir=args.run.resolve(); dest=args.dest.resolve(); cache=args.query_cache.resolve()
    candidate=manifest_for(run_dir,cache,args.joint_m33_root)
    dest.mkdir(parents=True,exist_ok=True); manifest_path=dest/"manifest.json"
    if manifest_path.exists() and not args.resume:
        raise RuntimeError("existing destination requires explicit --resume")
    if args.resume and not manifest_path.exists():
        raise RuntimeError("--resume requires an existing sealed manifest")
    if manifest_path.exists(): manifest=json.loads(manifest_path.read_text()); verify_manifest(manifest,run_dir,cache,args.joint_m33_root)
    else: manifest=candidate; atomic_json(manifest_path,manifest)
    meta=json.loads((run_dir/"run_meta.json").read_text())
    if args.preflight_only:
        out=preflight(args,manifest,meta); atomic_json(dest/"preflight_receipt.json",out); return out
    if (dest / "preflight_receipt.json").exists() or (dest / "score_receipt.json").exists():
        raise RuntimeError("preflight/completed destination cannot be promoted or repeated")
    duals,banks=zip(*(load_query_pair(s,cache) for s in SIX)); dual_map=dict(zip(SIX,duals)); bank_map=dict(zip(SIX,banks)); device=torch.device(args.device);torch.set_num_threads(args.cpu_threads)
    frontend,model=_model_and_loader(meta,device,bank_map,args.joint_m33_root)
    progress_path=dest/"score_progress.json"
    if args.resume and not progress_path.exists():
        raise RuntimeError("--resume requires existing score progress")
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {"schema":SCHEMA+"_progress","manifest_sha256":canonical_digest(manifest),"completed":{}}
    if progress.get("manifest_sha256") != canonical_digest(manifest): raise RuntimeError("progress is from another manifest")
    for epoch in EPOCHS:
        cp=Path(manifest["checkpoint_bytes"][str(epoch)]["path"]); cp_sha=sha(cp); prior=progress["completed"].get(str(epoch))
        if prior:
            if prior.get("checkpoint_sha256") != cp_sha or prior.get("partial") is not False: raise RuntimeError(f"epoch {epoch}: progress drift/partial")
            validate_complete_report(prior, manifest["query_assets"]["sessions"])
            continue
        state=torch.load(cp,map_location=device,weights_only=False); _checkpoint_contract(state,meta,epoch,frontend); _load_ema(model,state,frontend)
        report=score(model,dual_map,bank_map,device)
        expected=manifest["query_assets"]["sessions"]
        validate_complete_report(report, expected)
        progress["completed"][str(epoch)]={**report,"checkpoint_sha256":cp_sha};atomic_json(progress_path,progress)
    validate_complete_curve(progress["completed"], manifest["query_assets"]["sessions"])
    values={e:float(progress["completed"][str(e)]["equal_session_mean"]) for e in EPOCHS}
    best=max(EPOCHS,key=lambda e:(values[e],-e))
    selected=progress["completed"][str(best)]
    # Packaging is impossible until every full candidate is present and resealed.
    verify_manifest(manifest,run_dir,cache,args.joint_m33_root)
    selected_state=torch.load(Path(manifest["checkpoint_bytes"][str(best)]["path"]),map_location="cpu",weights_only=False)
    # Produce a strict-loadable family state dict after materializing EMA.  The
    # checkpoint's raw weights, optimizer, RNG, and EMA bookkeeping are absent.
    package_model_kind, package_model = _model_and_loader(meta, torch.device("cpu"), bank_map, args.joint_m33_root)
    _load_ema(package_model, selected_state, package_model_kind)
    ema_only = {name: value.detach().cpu().clone() for name, value in package_model.state_dict().items()}
    package=dest/"selected_ema.pt"
    if package.exists():
        prior = torch.load(package, map_location="cpu", weights_only=True)
        if set(prior) != set(ema_only) or any(not torch.equal(prior[k], ema_only[k]) for k in ema_only):
            raise RuntimeError("selected EMA package exists but differs from sealed selected candidate")
    else: torch.save(ema_only,package)
    receipt={"schema":SCHEMA+"_selection","status":"COMPLETED","manifest_sha256":canonical_digest(manifest),"view":VIEW,"selection":{"rule":"earliest maximum finite unweighted equal_session_mean","epoch":best,"equal_session_mean":values[best]},"selected":selected,"selected_ema_state":{"path":str(package),"sha256":sha(package),"raw_state_serialized":False},"ema_by_epoch":progress["completed"],"evalai_opened":False,"official_test_used":False,"utc":datetime.now(timezone.utc).isoformat()}
    atomic_json(dest/"score_receipt.json",receipt);atomic_json(dest/"selected_ema_receipt.json",receipt);return receipt


def main() -> int:
    p=argparse.ArgumentParser(description="sealed M2 RIFT ext6 EMA epoch picker")
    p.add_argument("--run-dir", "--run", dest="run", type=Path,required=True);p.add_argument("--dest",type=Path,required=True);p.add_argument("--query-cache",type=Path,default=QUERY_CACHE)
    p.add_argument("--joint-m33-root",type=Path);p.add_argument("--device",default="cuda:0");p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--preflight-only",action="store_true");p.add_argument("--preflight-batches",type=int,default=1);p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.cpu_threads < 1 or a.preflight_batches < 1: p.error("thread and preflight batch counts must be positive")
    print(json.dumps(run(a),indent=2,sort_keys=True));return 0
if __name__ == "__main__": raise SystemExit(main())
