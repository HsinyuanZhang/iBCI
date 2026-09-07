"""Gated, frozen-weight SPINT M30 replay on comparator's local within-post30 surface.

This is deliberately separate from paired training.  It never queries EvalAI
and never writes a submission artifact.  It is the concrete route from the
known SPINT teacher/config/carrier to a same-surface baseline receipt.
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any
import numpy as np

from tfpd_exploration.src.m2_same_query_comparator_v1 import core, physical, plan

OUT = Path(__file__).resolve().parents[3] / "tfpd_exploration/results/m2/family_v1/spint_m30_within_post30_replay_v1.json"
ENV = "M2_FAMILY_V1_SPINT_REPLAY"

def formal_replay_cli() -> str:
    return (f"PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID "
            f"CUBLAS_WORKSPACE_CONFIG=:4096:8 {ENV}=1 "
            "/home/xinyuan/miniconda3/envs/spint/bin/python -m tfpd_exploration.src.m2_family_v1.spint_replay --run")

def run() -> dict[str, Any]:
    if os.environ.get(ENV) != "1": raise RuntimeError(f"REFUSED: {ENV}=1 required")
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite immutable replay {OUT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0","1"}: raise RuntimeError("one leased GPU required")
    import torch
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic replay environment missing")
    torch.cuda.set_device(0); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model, data_module, _task, meta=load_frozen_model_and_data()
    core.require(meta["teacher_checkpoint_sha256"] == plan.SPINT_CHECKPOINT_SHA256, "SPINT checkpoint drift")
    core.require(meta["checkpoint_sha256"] == plan.T4_CHECKPOINT_SHA256, "carrier config checkpoint drift")
    core.require(meta["normalization_sha256"] == plan.NORMALIZATION_SHA256, "carrier normalizer drift")
    teacher=model.teacher.to("cuda:0").eval(); student=model.student.to("cuda:0").eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    cell=next(c for c in core.CELL_SPECS if c.name == "spint_chronological_m30")
    dataset=data_module.train_dataset; rows=[]
    for session in sorted(dataset.calib_trialized_neural_features):
        starts,target=physical._query_arrays(dataset,session,"within_post30")
        rows.append(physical._network_row(torch=torch,cell=cell,student=student,teacher=teacher,dataset=dataset,session=session,surface="within_post30",starts=starts,target=target,device=torch.device("cuda:0"),batch_size=1024))
    per={r["session"]:float(r["r2"]) for r in rows}
    payload={"schema":"m2_family_v1_spint_m30_within_post30_replay_v1","status":"IMMUTABLE_LOCAL_BASELINE","surface":"within_post30","cell":cell.name,"spint_checkpoint_sha256":meta["teacher_checkpoint_sha256"],"t4_checkpoint_sha256":meta["checkpoint_sha256"],"normalization_sha256":meta["normalization_sha256"],"rows":rows,"summary":core.summarize_sessions(per),"parameter_updates":0,"official_access":False}
    body=(json.dumps(payload,indent=2,sort_keys=True)+"\n").encode(); OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_bytes(body); OUT.with_suffix(OUT.suffix+".sha256").write_text(hashlib.sha256(body).hexdigest()+"  "+OUT.name+"\n"); OUT.chmod(0o444)
    return payload

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--run",action="store_true"); a=p.parse_args()
    print(json.dumps(run() if a.run else {"command":formal_replay_cli()},sort_keys=True))
