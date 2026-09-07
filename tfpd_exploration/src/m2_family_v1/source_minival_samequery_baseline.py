"""Prepared, explicitly gated same-query source-minival baseline replay.

Metadata mode is read-only.  Execution is deliberately refused until a
separate review authorizes the exact e8 and actual-SPINT replay resources.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_same_query_comparator_v1 import core
from tfpd_exploration.src.m2_same_query_comparator_v1 import physical

ROOT = Path(__file__).resolve().parents[3]
PAYLOAD = ROOT / "tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"
TEACHER = ROOT / "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt"
OUT = ROOT / "tfpd_exploration/results/m2/family_v1/source_minival_e8_spint_m30_replay_v1.json"
TEACHER_SHA = "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"
PAYLOAD_SHA = "4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4"
COUNTS = (173, 129, 117, 116, 141, 141, 194)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata() -> dict[str, object]:
    rows = []
    for session in plan.HELDIN_SESSIONS:
        root = data._session_dir("source_minival", session)
        starts = np.ascontiguousarray(np.load(root / "eligible_starts.npy"), dtype=np.int64)
        target = np.ascontiguousarray(np.load(root / "target_store.npy", mmap_mode="r"), dtype=np.float32)
        if starts.size == 0 or target.shape != (starts.size, 2):
            raise RuntimeError(f"{session}: source-minival geometry drift")
        rows.append({"session": session, "window_count": int(starts.size),
                     "ordered_window_starts_sha256": core.array_sha256(starts),
                     "target_sha256": core.array_sha256(target),
                     "X_store_sha256": sha(root / "X_store.npy"),
                     "target_store_sha256": sha(root / "target_store.npy"),
                     "eligible_starts_file_sha256": sha(root / "eligible_starts.npy")})
    return {"schema": "m2_family_v1_source_minival_samequery_baseline_manifest_v1",
            "status": "METADATA_ONLY_BASELINES_NOT_YET_EXECUTED",
            "surface": "source_minival all eligible windows; seven heldin sessions",
            "aggregation_primary": "equal-session mean native R2",
            "aggregation_secondary": "pooled native R2; disclosed, not paired-trainer primary",
            "native_units": True, "total_windows": sum(row["window_count"] for row in rows),
            "rows": rows, "e8_payload_sha256": sha(PAYLOAD),
            "baselines": {"e8": "missing exact source-minival replay",
                          "actual_spint": "missing exact source-minival replay"},
            "exposure": "training-overlap diagnostic only: historical e8 and original SPINT trained on these heldin sessions; not untouched generalization",
            "forbidden": ["selection", "official", "ext4", "heldout", "training"]}


def code_hashes() -> dict[str, str]:
    files = {
        "own": Path(__file__), "data": Path(data.__file__), "plan": Path(plan.__file__),
        "core": Path(core.__file__), "physical": Path(physical.__file__),
        "e8_payload_module": ROOT / "tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/trf_falcon_decoder.py",
        "spint": ROOT / "streaming_calibration_exp/src/models/components/spint.py",
    }
    return {name: sha(path) for name, path in files.items()}


def _payload_module():
    path = ROOT / "tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/trf_falcon_decoder.py"
    spec = importlib.util.spec_from_file_location("_minival_e8", path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve(): raise RuntimeError("e8 payload module import shadow")
    return module


def load_teacher_only():
    """Pure architecture construction + strict historical teacher state load; no DM/wrapper setup."""
    if sha(TEACHER) != TEACHER_SHA:
        raise RuntimeError("historical M2 teacher checkpoint SHA drift")
    streaming = ROOT / "streaming_calibration_exp"
    if str(streaming) not in sys.path: sys.path.insert(0, str(streaming))
    from src.models.components.spint import SpintModel
    if Path(inspect.getsourcefile(SpintModel)).resolve() != (streaming / "src/models/components/spint.py").resolve():
        raise RuntimeError("SpintModel import shadow")
    model = SpintModel(512, 2, 50, num_heads=64, num_layers=1, num_id_layers=3,
                       dropout_rate=0.0, dynamic_dropout=False, tf_drop_rate=0.1)
    payload = torch.load(TEACHER, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    keys = {key[4:]: value for key, value in state.items() if key.startswith("net.")}
    if not keys or any(key.startswith("net.") and key[4:] not in keys for key in state):
        raise RuntimeError("checkpoint net-prefix state contract drift")
    missing, unexpected = model.load_state_dict(keys, strict=True)
    if missing or unexpected: raise RuntimeError("strict teacher state mismatch")
    model.eval()
    for value in model.parameters(): value.requires_grad_(False)
    if (model.model_dim, model.num_covariates, model.window_size, model.num_heads,
            model.num_layers, model.num_id_layers) != (512, 2, 50, 64, 1, 3):
        raise RuntimeError("teacher geometry drift")
    return model


def sealed_inputs():
    manifest = metadata()
    if manifest["total_windows"] != 1011 or sha(PAYLOAD) != PAYLOAD_SHA: raise RuntimeError("source-minival seal drift")
    if tuple(row["window_count"] for row in manifest["rows"]) != COUNTS: raise RuntimeError("exact source-minival count drift")
    for row in manifest["rows"]:
        root = data._session_dir("source_minival", row["session"])
        activity = np.load(root / "calib_activity.npy", mmap_mode="r")
        if activity.shape != (33, 100, 96): raise RuntimeError("stored calibration geometry drift")
        row["calib_activity_sha256"] = sha(root / "calib_activity.npy")
        row["calib_identity_slice"] = [0, 30]
        row["calib_first30_sha256"] = core.array_sha256(np.ascontiguousarray(activity[:30], dtype=np.float32))
        row["provenance_sha256"] = sha(root / "provenance.json")
        row["mapping_sha256"] = sha(root / "mapping.json")
    return manifest


def run() -> dict[str, object]:
    if os.environ.get("M2_SOURCE_MINIVAL_BASELINE_REPLAY") != "1": raise RuntimeError("explicit replay authorization required")
    if OUT.exists() or OUT.with_suffix(".npz").exists(): raise FileExistsError(OUT)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    seal = sealed_inputs(); code_before = code_hashes(); teacher_before = sha(TEACHER); payload_before = sha(PAYLOAD)
    teacher = load_teacher_only(); module = _payload_module(); payload = module.load_payload(PAYLOAD)
    if payload.get("schema_version") != module.PAYLOAD_SCHEMA or float(payload["behavior_scaling_factor"]) != 5.0: raise RuntimeError("e8 payload contract drift")
    rows=[]; ep=[]; sp=[]; ys=[]; starts_all=[]; sessions=[]
    with torch.inference_mode():
        for item in seal["rows"]:
            session=item["session"]; root=data._session_dir("source_minival",session)
            raw=np.asarray(np.load(root/"X_store.npy",mmap_mode="r"),dtype=np.float32); starts=np.load(root/"eligible_starts.npy"); target=np.asarray(np.load(root/"target_store.npy",mmap_mode="r"),dtype=np.float32)
            if (starts.size != item["window_count"] or target.shape != (starts.size,2)
                    or target.dtype != np.float32 or not np.isfinite(target).all()
                    or int(starts.min()) < 0 or int(starts.max())+50 > len(raw)): raise RuntimeError("window/target range drift")
            bits=session.removeprefix("ses-").split("-"); tag=f"{bits[-1]}_{''.join(bits[:3])}"; bank_payload=payload["bank_by_dataset_tag"][tag]
            bank=module.SessionBank(torch.as_tensor(bank_payload["E0"],dtype=torch.float32),
                                    torch.as_tensor(bank_payload["T"],dtype=torch.float32),
                                    torch.as_tensor(bank_payload["unit_mask"],dtype=torch.bool))
            if bank.E0.shape != (96,50) or bank.T.shape != (96,4) or bank.unit_mask.shape != (96,) or not bool(bank.unit_mask.any()):
                raise RuntimeError("e8 payload bank geometry/mask drift")
            activity=torch.from_numpy(np.array(np.load(root/"calib_activity.npy",mmap_mode="r")[:30],dtype=np.float32,copy=True)).unsqueeze(0)
            identity=physical._teacher_identity(torch,teacher,activity); e8=[]; old=[]
            decoder=module.build_decoder(payload["kind"]).eval(); decoder.load_state_dict({k:torch.as_tensor(v,dtype=torch.float32) for k,v in payload["state_dict"].items()},strict=True)
            e8=[]; old=[]
            for offset in range(0,len(starts),16):
                ix=starts[offset:offset+16]; x=torch.from_numpy(np.stack([raw[i:i+50] for i in ix]).astype(np.float32))
                e8.append((decoder.forward_last(x,bank)/5).numpy()); old.append((physical._manual_teacher_decode(teacher,x,identity)[:,-1,:]/5).numpy())
            e8=np.concatenate(e8).astype(np.float32); old=np.concatenate(old).astype(np.float32)
            if e8.shape != target.shape or old.shape != target.shape or not np.isfinite(e8).all() or not np.isfinite(old).all(): raise RuntimeError("native prediction shape/finite drift")
            rows.append({"session":session,"window_count":len(starts),"ordered_window_starts_sha256":core.array_sha256(starts),"target_sha256":core.array_sha256(target),"e8_prediction_sha256":core.array_sha256(e8),"spint_prediction_sha256":core.array_sha256(old),"e8_r2":core.variance_weighted_r2(target,e8),"spint_r2":core.variance_weighted_r2(target,old),"native_units":True})
            ep.append(e8);sp.append(old);ys.append(target);starts_all.append(starts);sessions += [session]*len(starts)
    e=np.concatenate(ep); s=np.concatenate(sp); y=np.concatenate(ys)
    if e.shape != (1011,2) or y.shape != (1011,2) or len(sessions) != 1011: raise RuntimeError("global native archive geometry drift")
    if sealed_inputs() != seal or code_hashes() != code_before or sha(TEACHER) != teacher_before or sha(PAYLOAD) != payload_before: raise RuntimeError("input/code authority drift after inference")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=OUT.parent,suffix=".npz",delete=False) as handle:
        temporary=Path(handle.name)
    np.savez_compressed(temporary,e8_prediction=e,spint_prediction=s,target=y,start=np.concatenate(starts_all),session=np.asarray(sessions))
    archive=OUT.with_suffix(".npz"); temporary.replace(archive)
    result={"schema":"m2_family_v1_source_minival_samequery_baseline_v1","status":"TRAINING_OVERLAP_DIAGNOSTIC_NOT_GENERALIZATION","surface":"source_minival","aggregation_primary":"equal_session_mean_r2","aggregation_secondary":"pooled_r2_not_paired_trainer_primary","rows":rows,"e8_equal_session_r2":float(np.mean([r["e8_r2"] for r in rows])),"spint_equal_session_r2":float(np.mean([r["spint_r2"] for r in rows])),"e8_pooled_r2":core.variance_weighted_r2(y,e),"spint_pooled_r2":core.variance_weighted_r2(y,s),"input_manifest_pre":seal,"input_manifest_post":sealed_inputs(),"code_sha256_pre":code_before,"code_sha256_post":code_hashes(),"e8_payload_sha256_pre":payload_before,"e8_payload_sha256_post":sha(PAYLOAD),"spint_teacher_sha256_pre":teacher_before,"spint_teacher_sha256_post":sha(TEACHER),"stream_npz_sha256":sha(archive),"parameter_updates":0}
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def smoke() -> dict[str, object]:
    """Synthetic B2 architecture smoke; intentionally opens no source query/targets."""
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    teacher=load_teacher_only(); module=_payload_module(); payload=module.load_payload(PAYLOAD)
    tags=sorted(payload["bank_by_dataset_tag"])[:2]; rows=[payload["bank_by_dataset_tag"][tag] for tag in tags]
    bank=module.SessionBank(torch.stack([torch.as_tensor(row["E0"],dtype=torch.float32) for row in rows]),
                            torch.stack([torch.as_tensor(row["T"],dtype=torch.float32) for row in rows]),
                            torch.stack([torch.as_tensor(row["unit_mask"],dtype=torch.bool) for row in rows]))
    decoder=module.build_decoder(payload["kind"]).eval(); decoder.load_state_dict({k:torch.as_tensor(v,dtype=torch.float32) for k,v in payload["state_dict"].items()},strict=True)
    x=torch.zeros((2,50,96),dtype=torch.float32); activity=torch.zeros((2,30,100,96),dtype=torch.float32)
    with torch.inference_mode():
        e8=decoder.forward_last(x,bank,bank.unit_mask); identity=torch.cat([physical._teacher_identity(torch,teacher,activity[i:i+1]) for i in range(2)])
        spint=physical._manual_teacher_decode(teacher,x,identity)[:,-1,:]
    if e8.shape != (2,2) or spint.shape != (2,2) or not bool(torch.isfinite(e8).all()) or not bool(torch.isfinite(spint).all()): raise RuntimeError("synthetic B2 smoke failed")
    return {"status":"PASS_SYNTHETIC_ZERO_B2_NO_QUERY_TARGET_SCORE","e8_shape":list(e8.shape),"spint_shape":list(spint.shape),"tags":tags}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", action="store_true"); parser.add_argument("--preflight",action="store_true"); parser.add_argument("--loader-test",action="store_true"); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--run",action="store_true")
    args = parser.parse_args()
    if args.metadata: print(json.dumps(metadata(), indent=2, sort_keys=True))
    elif args.preflight: print(json.dumps({"status":"PASS_INPUT_PREFLIGHT_NO_MODEL","manifest":sealed_inputs()},sort_keys=True))
    elif args.loader_test: print(json.dumps({"status":"PASS_PURE_MODEL_STRICT_LOAD_NO_QUERIES","teacher_sha256":TEACHER_SHA,"geometry":[512,2,50,64,1,3],"state_keys":len(load_teacher_only().state_dict())},sort_keys=True))
    elif args.smoke: print(json.dumps(smoke(),sort_keys=True))
    elif args.run: print(json.dumps(run(),sort_keys=True))
    else: parser.error("choose --metadata, --loader-test, or separately authorized --run")
