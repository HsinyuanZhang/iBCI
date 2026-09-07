"""Authorized-only supervisor for CRST-B4 split-arm formal workers.

Running this module requires an external authorization path and an empty output
root.  It never restarts a failed worker and stops both children at the fixed
six-hour global deadline.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import torch
from . import familyformal_split_train as train

DEADLINE_SECONDS = 21600

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def failure(out: Path, reason: str, children: dict[str, subprocess.Popen]) -> None:
    for p in children.values():
        if p.poll() is None: p.terminate()
    for p in children.values():
        try: p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(timeout=15)
    train.atomic_json({"status":"FAILED_OR_INCOMPLETE_NO_RESTART","reason":reason,"children":{a:p.returncode for a,p in children.items()}},out / "FAILED_OR_INCOMPLETE.json")

def main(output: Path, authorization: Path) -> None:
    if output.exists() or not authorization.is_file(): raise RuntimeError("requires empty output and explicit authorization")
    # Authorization is deliberately hash-bound by a final reviewer, not inferred.
    auth=json.loads(authorization.read_text())
    closure={**train.code_closure(),"launcher":sha(Path(__file__)),"protocol":train.protocol_sha256()}
    from tfpd_exploration.src.h1_optimized_v2.cache import ROOT as H1ROOT, CACHE
    gate=H1ROOT/"family_v1/crst_b4_set_v3_localbalanced_source208_preflight_v1/report_1040.json"; resource=H1ROOT/"family_v1/crst_b4_localbalanced_splitarm_recipe_smoke_v1/report.json"; source=H1ROOT/"source_cache_authority.json"
    bindings={"source_authority_sha256":sha(source),"source_cache_sha256":sha(CACHE),"gate1040_sha256":sha(gate),"split_resource_sha256":sha(resource)}
    if auth.get("code_closure") != closure or auth.get("bindings") != bindings or auth.get("output") != str(output): raise RuntimeError("authorization does not bind exact closure/input/output")
    output.mkdir(parents=True); (output/"barrier").mkdir(); (output/"workers").mkdir(); (output/"checkpoints").mkdir()
    train.atomic_json({"authorization_sha256":sha(authorization),"code_closure":closure,"bindings":bindings,"protocol":train.PROSPECTIVE_PROTOCOL},output/"input_authority.json")
    children={}; finalizers={}; began=time.monotonic()
    for arm,gpu in train.ARM_DEVICE.items():
        env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONNOUSERSITE":"1"}
        children[arm]=subprocess.Popen(["taskset","-c","4-7",sys.executable,"-c",f"from pathlib import Path; from tfpd_exploration.src.h1_family_v1.familyformal_split_train import worker_run; worker_run(arm='{arm}', output=Path(r'{output}'), physical_gpu={gpu}, start_marker=Path(r'{output}/barrier/START'))"],env=env,cwd=Path.cwd())
    try:
        while not all((output/"barrier"/f"{a}.ready.json").is_file() for a in children):
            if any(p.poll() is not None for p in children.values()): raise RuntimeError("child failed preflight")
            if time.monotonic()-began>DEADLINE_SECONDS: raise TimeoutError("global deadline before start")
            time.sleep(.1)
        ready={a:json.loads((output/"barrier"/f"{a}.ready.json").read_text()) for a in children}
        if ready["flat"]["shared_sha256"] != ready["route"]["shared_sha256"] or ready["flat"]["identities"] != ready["route"]["identities"]: raise RuntimeError("paired preupdate identities mismatch")
        (output/"barrier"/"START").write_text("authorized synchronized start\n")
        while any(p.poll() is None for p in children.values()):
            if any(p.poll() not in (None,0) for p in children.values()): raise RuntimeError("child failed; no auto-restart")
            if time.monotonic()-began>DEADLINE_SECONDS: raise TimeoutError("global deadline")
            time.sleep(.5)
        if any(p.returncode != 0 for p in children.values()): raise RuntimeError("child failed at phase exit")
        complete={a:json.loads((output/"workers"/f"{a}_complete.json").read_text()) for a in children}
        for arm, result in complete.items():
            expected_pick = train.earliest_argmax([(row["epoch"], row["selection"]["r2_concat_float64"]) for row in result["epochs"]])
            if expected_pick != (result["selected_epoch"], result["selected_ema_r2_float64"]): raise RuntimeError("all-epoch selection mismatch")
            if result["identities"] != ready[arm]["identities"] or result["shared_sha256"] != ready[arm]["shared_sha256"]: raise RuntimeError("worker identity drift")
            for row in result["epochs"]:
                if row["selection"]["n_bins"] != 2908 or sha(Path(row["checkpoint"])) != row["checkpoint_sha256"]: raise RuntimeError("epoch score/checkpoint authority drift")
        # Freeze only after both arms' all-12 primary EMA receipts exist.
        freeze={a:{"epoch":complete[a]["selected_epoch"],"ema_r2_float64":complete[a]["selected_ema_r2_float64"],"checkpoint":complete[a]["epochs"][complete[a]["selected_epoch"]-1]["checkpoint"],"checkpoint_sha256":complete[a]["epochs"][complete[a]["selected_epoch"]-1]["checkpoint_sha256"],"tie_break":"earliest"} for a in children}
        endpoints = {arm: {"epoch": 12, "ema_r2_float64": result["epochs"][-1]["selection"]["r2_concat_float64"], "checkpoint": result["epochs"][-1]["checkpoint"], "checkpoint_sha256": result["epochs"][-1]["checkpoint_sha256"]} for arm, result in complete.items()}
        train.atomic_json({"schema":"h1_crst_b4_splitarm_selection_freeze_v1","selected":freeze,"endpoints":endpoints},output/"selection_freeze.json")
        for arm,gpu in train.ARM_DEVICE.items():
            env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONNOUSERSITE":"1"}
            finalizers[arm]=subprocess.Popen(["taskset","-c","4-7",sys.executable,"-c",f"from pathlib import Path; from tfpd_exploration.src.h1_family_v1.familyformal_split_train import finalizer_run; finalizer_run(arm='{arm}', output=Path(r'{output}'), physical_gpu={gpu})"],env=env,cwd=Path.cwd())
        while any(p.poll() is None for p in finalizers.values()):
            if any(p.poll() not in (None,0) for p in finalizers.values()): raise RuntimeError("strict finalizer failed")
            if time.monotonic()-began>DEADLINE_SECONDS: raise TimeoutError("global deadline finalizers")
            time.sleep(.5)
        if any(p.returncode != 0 for p in finalizers.values()): raise RuntimeError("finalizer failed at phase exit")
        finals={a:json.loads((output/"workers"/f"{a}_final.json").read_text()) for a in finalizers}
        train.require_frozen_inputs(output)
        train.atomic_json({"status":"COMPLETE","selection_freeze":freeze,"complete":finals},output/"final.json")
    except BaseException as exc:
        failure(output,repr(exc),{**children,**finalizers}); raise

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--authorization",type=Path,required=True);a=p.parse_args();main(a.output,a.authorization)
