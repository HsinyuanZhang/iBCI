#!/usr/bin/env python3
"""Fail-closed aggregation of independently written M1 DLA arm results."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("full", "rate_only", "label_shuffle", "rate_residualized_condition_only")
BASE = ROOT / "sua_exploration" / "results" / "m1_decoder_latent_alignment_oracle_v2" / "selection_runs"

def sha256(p: Path) -> str:
 d=hashlib.sha256();
 with p.open("rb") as h:
  for b in iter(lambda:h.read(1<<20),b""): d.update(b)
 return d.hexdigest()

def main() -> None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,default=BASE);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if a.out.exists(): raise FileExistsError(f"refusing to overwrite {a.out}")
 rows={}
 for arm in ARMS:
  f=a.base/f"arm_{arm}"/"result.json"
  if not f.is_file(): raise FileNotFoundError(f"missing immutable arm result {f}")
  x=json.loads(f.read_text());
  if set(x.get("arms",{}))!={arm} or x.get("heldout_access") or x.get("EvalAI_access") or not x.get("no_report_access"):
   raise ValueError(f"{arm}: invalid scope/result arm")
  rows[arm]=(x,f)
 reference=rows["full"][0]
 for arm,(x,_) in rows.items():
  if x["manifest"]!=reference["manifest"] or x["source_audits"]!=reference["source_audits"]:
   raise ValueError(f"{arm}: manifest/session audit mismatch")
  if x["arms"][arm]["outer_selection"]["identity_r2"]!=reference["arms"]["full"]["outer_selection"]["identity_r2"]:
   raise ValueError(f"{arm}: F0 outer baseline mismatch")
 f0=reference["arms"]["full"]["outer_selection"]["identity_r2"]
 d={arm:x["arms"][arm]["outer_selection"]["delta_r2"] for arm,(x,_) in rows.items()}
 ci=reference["arms"]["full"]["outer_selection"]["bootstrap"]["ci95"];mde=reference["arms"]["full"]["outer_selection"]["bootstrap"]["two_sided_mde_r2"]
 gates={"full_minus_identity_ge_0p015":d["full"]>=.015,"full_minus_rate_ge_0p010":d["full"]-d["rate_only"]>=.01,"full_minus_label_shuffle_ge_0p010":d["full"]-d["label_shuffle"]>=.01,"full_ci_excludes_zero":ci[0]>0,"full_mde_le_0p015":mde<=.015}
 out={"schema_version":"m1_dla_selection_arm_aggregate_v1","manifest":reference["manifest"],"source_audits":reference["source_audits"],"f0_outer_identity_r2":f0,"delta_r2_by_arm":d,"full_bootstrap":reference["arms"]["full"]["outer_selection"]["bootstrap"],"gates":gates,"input_hashes":{arm:sha256(pth) for arm,(_,pth) in rows.items()},"scope":{"no_report_access":True,"heldout_access":False,"EvalAI_access":False}}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(a.out)
if __name__=="__main__":main()
