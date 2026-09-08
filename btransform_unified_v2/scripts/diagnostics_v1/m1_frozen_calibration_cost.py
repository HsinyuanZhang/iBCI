#!/usr/bin/env python3
"""Read-only CPU audit of the M1 RIFT e3 bank calibration path.

It rebuilds the two production inputs independently: rSyn3 from raw M10
support and B3 Sfix-e11 E0 from the same M10 neural trials.  It never loads a
decoder checkpoint, runs query scoring, or applies target backpropagation.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, resource, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Set process limits before numerical libraries are imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WS, V1 = ROOT.parent, ROOT.parent / "btransform_unified_v1"
for path in (ROOT / "src", V1 / "src", V1 / "scripts", WS):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
import m1_projadd_depth2_series as legacy
from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
from tfpd_exploration.src.m1_optimized_v2 import bank as source_bank
from tfpd_exploration.src.m1_optimized_v2 import plan as source_plan
from tfpd_exploration.src.two_mainlines_long_v1.decoder import m1_config as b3_config

HO = ("20121004", "20121017", "20121024")
E3_RECEIPT = ROOT / "results/rift_v1/m1_r100_recency_s42_formal_v3/score_receipt.json"
DEFAULT_OUT = ROOT / "results/diagnostics_v1/m1_frozen_calibration_cost_preflight_20260908.json"

def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()
def sha_array(a: Any) -> str: return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def utc() -> str: return datetime.now(timezone.utc).isoformat()
def rss_kib() -> int: return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
def fresh_dest(p: Path) -> None:
    if p.exists(): raise FileExistsError(f"refusing to overwrite existing receipt: {p}")
def source_hashes() -> dict[str,str]:
    paths=(Path(__file__), V1/"scripts/m1_projadd_depth2_series.py", V1/"src/btransform_unified_v1/m1_projadd.py", WS/"tfpd_exploration/src/m1_b3_allsource_v1/rsyn3_bank.py", WS/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py", Path(source_bank.__file__), source_plan.BANK_NPZ, b3_config.S_FIX_PATH, E3_RECEIPT)
    return {str(p):sha_file(p) for p in paths}
def basis() -> syn3.SourceBasis:
    z=np.load(source_plan.BANK_NPZ,allow_pickle=False)
    return syn3.SourceBasis("nnmf",np.asarray(z["scale"]),np.asarray(z["d0"]),np.asarray(z["activations"]),tuple(int(x) for x in z["nmf_order"]),str(z["reconstruction_digest"][0]),{"source":"rSyn3-refit-v1.sealed"},{})
def support_path(session: str) -> Path: return legacy._heldout_calib_path(session)

def _cold_inputs(session: str) -> dict[str, Any]:
    """Load all immutable source objects once; these loads are outside warm-bank timing."""
    p = support_path(session)
    t = time.perf_counter(); record = rsyn3_bank.load_public_calib_support(p); carrier_rawload = time.perf_counter() - t
    t = time.perf_counter(); opened = legacy.open_heldout_calib_session(session); b3_calib_rawload = time.perf_counter() - t
    t = time.perf_counter(); frozen_basis = basis(); norm = source_bank.load(); carrier_objects_coldload = time.perf_counter() - t
    t = time.perf_counter(); provider = legacy.mp.default_identity_provider(); b3_provider_coldload = time.perf_counter() - t
    return {"path": p, "record": record, "calib10": np.ascontiguousarray(opened["calib10"], dtype=np.float32), "basis": frozen_basis, "normalizer": norm, "provider": provider,
            "cold_loads": {"carrier_raw_M10_support_load_seconds": carrier_rawload, "B3_raw_M10_neural_input_load_seconds": b3_calib_rawload, "rSyn3_H_and_source_RMS_object_coldload_seconds": carrier_objects_coldload, "frozen_B3_Sfix_e11_provider_coldload_seconds": b3_provider_coldload}}

def rebuild_one(session: str, *, time_round: bool) -> dict[str,Any]:
    """One true in-memory M10-to-static-bank timing, plus separately measured cold setup."""
    inputs = _cold_inputs(session)
    # This is the only timing labelled warm/static bank wall. It starts after
    # every raw file and source object exists in memory, then ends immediately
    # after TaskBank construction. No SHA, serialization, JSON, or query work is inside.
    wall0 = time.perf_counter()
    t = time.perf_counter(); raw = rsyn3_bank._encode_record(inputs["record"], inputs["basis"]); carrier = np.ascontiguousarray(syn3.normalize_carriers(raw, inputs["normalizer"]["normalizer_mean"], inputs["normalizer"]["normalizer_scale"]), dtype=np.float32); carrier_solve = time.perf_counter() - t
    t = time.perf_counter(); e0 = np.ascontiguousarray(inputs["provider"](inputs["calib10"]), dtype=np.float32); embedding = time.perf_counter() - t
    t = time.perf_counter(); direct = legacy.make_pick_bank(session, e0, carrier); static_bank = time.perf_counter() - t
    warm_wall = time.perf_counter() - wall0
    # Post-boundary reporting work is intentionally below this line.
    t = time.perf_counter(); blob = pickle.dumps({"E0":e0,"carrier":carrier,"unit_mask":direct.unit_mask}, protocol=4); serialization = time.perf_counter() - t
    p = inputs["path"]
    return {"session":session,"timed":time_round,"support":{"path":str(p),"sha256":sha_file(p),"trial_ids":list(range(10)),"carrier_raw_shape":list(inputs["record"].rates.shape),"B3_calib10_shape":list(inputs["calib10"].shape)},"cold_setup":inputs["cold_loads"],"warm_inmemory_components":{"carrier_solve_seconds":carrier_solve,"embedding_build_seconds":embedding,"static_bank_materialization_seconds":static_bank},"warm_static_bank_wall":{"definition":"one perf_counter boundary from already-in-memory raw M10 support, B3 M10 neural activity, rSyn3 H/source-RMS objects, and frozen B3 provider through carrier + E0 + TaskBank; excludes hashing, serialization, JSON, and all file/source-object loads","wall_seconds":warm_wall,"same_perf_counter_boundary":True},"cold_to_static_bank_scope":{"definition":"raw support/B3 input loads and source-object cold loads are separately measured in cold_setup; their sum must not be reported as an end-to-end wall because they are independent timer intervals","reported_as_end_to_end":False},"serialization_outside_wall":{"seconds":serialization,"nbytes":len(blob)},"arrays":{"carrier":{"shape":list(carrier.shape),"dtype":str(carrier.dtype),"sha256":sha_array(carrier)},"E0":{"shape":list(e0.shape),"dtype":str(e0.dtype),"sha256":sha_array(e0)},"unit_mask":{"sha256":sha_array(direct.unit_mask)},"memory_nbytes":int(carrier.nbytes+e0.nbytes+direct.unit_mask.nbytes)},"query":{"query_values_read_for_carrier":False,"query_values_read_for_e0":False,"query_scoring":False,"target_backpropagation":False},"_bank":direct}

def production_equality(sessions: tuple[str,...], locally_rebuilt: dict[str,Any]) -> dict[str,Any]:
    """Compare against m1_train's actual production materializer, no decoder call."""
    import importlib.util
    spec=importlib.util.spec_from_file_location("_m1_train_for_cost",ROOT/"scripts/rift_v1/m1_train.py")
    if spec is None or spec.loader is None: raise RuntimeError("cannot import m1_train")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    prod=module._ho_material()  # Its dataset construction is only a comparison witness; no score/model invocation occurs.
    rows={}
    for s in sessions:
        ours=locally_rebuilt[s]["_bank"]; theirs=prod[s]["bank"]
        c=bool(np.array_equal(ours.carrier,theirs.carrier)); e=bool(np.array_equal(ours.E0,theirs.E0)); m=bool(np.array_equal(ours.unit_mask,theirs.unit_mask))
        # E0 can differ by platform/library float kernels. Report error explicitly;
        # byte equality remains the strict production-reproducibility test.
        maxerr=float(np.max(np.abs(ours.E0-theirs.E0)))
        rows[s]={"carrier_byte_equal":c,"E0_byte_equal":e,"unit_mask_byte_equal":m,"E0_max_abs_error":maxerr,"E0_float_tolerance":1e-6,"E0_within_tolerance":maxerr<=1e-6,"production_carrier_sha256":sha_array(theirs.carrier),"production_E0_sha256":sha_array(theirs.E0),"rebuilt_carrier_sha256":sha_array(ours.carrier),"rebuilt_E0_sha256":sha_array(ours.E0)}
        if not (c and m and maxerr<=1e-6): raise RuntimeError(f"{s}: reconstruction diverges from m1_train production bank")
    return rows

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--sessions",nargs="+",choices=HO,default=list(HO)); ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--cpus",default="12,13"); ap.add_argument("--preflight",action="store_true")
    a=ap.parse_args(); fresh_dest(a.output)
    if a.rounds < 1: raise ValueError("--rounds must be positive")
    cpus={int(x) for x in a.cpus.split(",")}; allowed={12,13}
    if not cpus or not cpus <= allowed: raise RuntimeError(f"this task may only use CPUs {sorted(allowed)}, got {sorted(cpus)}")
    os.sched_setaffinity(0,cpus); torch.set_num_threads(2); torch.set_num_interop_threads(1)
    # Rebuild every HO bank once for required production equality. Only selected
    # sessions receive repeat timing; preflight commonly selects exactly one.
    once={s:rebuild_one(s,time_round=False) for s in HO}
    equality=production_equality(HO,once)
    rounds={s:[] for s in a.sessions}
    for s in a.sessions:
        for _ in range(a.rounds): rounds[s].append(rebuild_one(s,time_round=True))
    receipt={"schema":"m1_frozen_calibration_cost_v2","status":"PREFLIGHT_COMPLETED" if a.preflight else "COMPLETED","utc":utc(),"scope":{"cell":"M1-RIFT-R100-D4-P16-RECENCY-V1","selected_weight_view":"EMA","selected_epoch":3,"target_bp":0,"query_not_used":True,"no_decoder_or_full_window_score":True,"formal_protocol":"all 3 sessions × 3 measured rounds, launched only by coordinator"},"runtime":{"device":"cpu","dtype":"float32","threads":2,"affinity":sorted(os.sched_getaffinity(0)),"nice":os.nice(0),"max_rss_kib":rss_kib()},"e3_reference":{"score_receipt":str(E3_RECEIPT),"score_receipt_sha256":sha_file(E3_RECEIPT),"checkpoint_sha256":json.loads(E3_RECEIPT.read_text())["checkpoint_sha256_by_epoch"]["3"]},"construction":{"carrier":"raw M10 support -> rSyn3 SourceBasis H -> source RMS normalizer","E0":"frozen B3 Sfix-e11 default_identity_provider on M10 neural activity","two_inputs":"carrier and E0 are rebuilt independently from the actual production inputs"},"production_binding":{"B3_checkpoint_path":str(b3_config.S_FIX_PATH),"B3_checkpoint_sha256":sha_file(b3_config.S_FIX_PATH),"B3_checkpoint_expected_sha256":b3_config.S_FIX_SHA256,"B3_provider_source":str(Path(legacy.mp.__file__).resolve()),"B3_provider_source_sha256":sha_file(Path(legacy.mp.__file__).resolve()),"rSyn3_H_and_source_RMS_npz":str(source_plan.BANK_NPZ),"rSyn3_H_and_source_RMS_npz_sha256":sha_file(source_plan.BANK_NPZ)},"source_hashes":source_hashes(),"production_bank_equality":equality,"untimed_reconstruction":{s:{k:v for k,v in row.items() if k!="_bank"} for s,row in once.items()},"timed_rounds":{s:[{k:v for k,v in row.items() if k!="_bank"} for row in rs] for s,rs in rounds.items()}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n"); print(a.output)
if __name__=="__main__": main()
