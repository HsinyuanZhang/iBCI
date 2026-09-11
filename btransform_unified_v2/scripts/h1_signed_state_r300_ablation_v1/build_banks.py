#!/usr/bin/env python3
"""Seal H1 signed-state baseline ablation banks for ACTIVITY_ONLY or NONE.

ACTIVITY_ONLY rematerializes each official C2 M3 activity identity with literal
zero T. NONE creates literal zero E0 and T and never calls C2 materialization.
No hidden test data is opened.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
if __name__ == "__main__" and ("-h" in __import__("sys").argv or "--help" in __import__("sys").argv):
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--arm", choices=("ACTIVITY_ONLY", "NONE"), required=True)
    _parser.add_argument("--dest", type=Path, required=True)
    _parser.print_help()
    raise SystemExit(0)
import numpy as np
import torch
from common import SOURCE_PAYLOAD, atomic_json, setup_imports, sha_array, sha_file
setup_imports()
from btransform_unified_v1 import adapters, h1_config
import h1_c2_cal1_b2_l200_p16 as c2

ARMS=("ACTIVITY_ONLY","NONE")
UNITS=176; E0_WIDTH=700; T_WIDTH=4

def _need(ok: bool, msg: str) -> None:
    if not ok: raise RuntimeError(msg)

def _zero_t() -> np.ndarray: return np.zeros((UNITS,T_WIDTH),np.float32)
def _zero_e0() -> np.ndarray: return np.zeros((UNITS,E0_WIDTH),np.float32)

def _act_bank(session: str) -> tuple[np.ndarray,np.ndarray,str]:
    activity, _full_t=adapters._h1_payload_arrays(session)
    _need(np.asarray(activity).shape==(3,1024,UNITS),"C2 activity must be exact M3 [3,1024,176]")
    _need(np.asarray(activity).dtype==np.float32 and np.isfinite(activity).all(),"C2 activity must be finite float32")
    t=_zero_t(); e0,hc=c2._materialize_e0(activity,t)
    _need(np.array_equal(hc,t),"ACTIVITY_ONLY C2 materializer did not retain literal zero T")
    _need(e0.shape==(UNITS,E0_WIDTH) and e0.dtype==np.float32 and np.isfinite(e0).all(),"ACTIVITY_ONLY rematerialized E0 shape/finite dtype drift")
    return np.ascontiguousarray(e0,np.float32),t,sha_array(np.ascontiguousarray(activity,np.float32))

def _none_bank() -> tuple[np.ndarray,np.ndarray]: return _zero_e0(),_zero_t()

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--arm",choices=ARMS,required=True); p.add_argument("--dest",type=Path,required=True); a=p.parse_args()
    dest=a.dest.resolve(); _need(not dest.exists(),f"refusing existing --dest: {dest}")
    _need(SOURCE_PAYLOAD.is_file(),f"source payload missing: {SOURCE_PAYLOAD}")
    payload=torch.load(SOURCE_PAYLOAD,map_location="cpu",weights_only=False); rows=payload.get("sessions",{})
    _need(isinstance(rows,dict) and len(rows)==27,"official C2 payload must contain 27 tags")
    arrays: dict[str,np.ndarray]={}; tags: dict[str,Any]={}
    for tag in sorted(rows):
        row=rows[tag]; session=str(row["session"]); trials=tuple(float(x) for x in row["calibration_trials"])
        _need(len(trials)==3 and len(set(trials))==3,f"{tag}: exact public M3 required")
        if a.arm=="ACTIVITY_ONLY": e0,t,activity_sha=_act_bank(session)
        else: e0,t=_none_bank(); activity_sha=None
        arrays[f"E0/{tag}"]=e0; arrays[f"T/{tag}"]=t
        tags[tag]={"session":session,"trials":list(trials),"E0_sha256":sha_array(e0),"T_sha256":sha_array(t),"activity_sha256":activity_sha,"E0_literal_zero":bool(np.array_equal(e0,_zero_e0())),"T_literal_zero":bool(np.array_equal(t,_zero_t()))}
    dest.mkdir(parents=True); np.savez_compressed(dest/"banks_27.npz",**arrays)
    receipt={"schema":"h1_signed_state_r300_ablation_banks_v1","status":"BUILT","arm":a.arm,"source_sessions":list(h1_config.H1_ALL_SESSIONS),"source_count":13,"tag_count":27,"bank_shape":{"E0":[UNITS,E0_WIDTH],"T":[UNITS,T_WIDTH]},"information_route":"C2 activity rematerialized with literal zero T" if a.arm=="ACTIVITY_ONLY" else "literal zero E0 and T; C2 materializer not called","official_test_used":False,"source_payload":str(SOURCE_PAYLOAD),"source_payload_sha256":sha_file(SOURCE_PAYLOAD),"banks_27":str((dest/"banks_27.npz").resolve()),"banks_27_sha256":sha_file(dest/"banks_27.npz"),"implementation_sha256":{str(Path(__file__).resolve()):sha_file(Path(__file__).resolve()),str(Path(__file__).resolve().parent/"common.py"):sha_file(Path(__file__).resolve().parent/"common.py")},"tags":tags,"utc":datetime.now(timezone.utc).isoformat()}
    atomic_json(dest/"receipt.json",receipt); print(json.dumps({"arm":a.arm,"receipt":str(dest/"receipt.json"),"tags":len(tags)},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
