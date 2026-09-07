#!/usr/bin/env python3
"""Final GPU-forward-only scorer preflight; writes evidence, never opens M1 data."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"sua_exploration/results/m1_decoder_latent_alignment_oracle_v2"
FILES={"runner":ROOT/"sua_exploration/scripts/m1_decoder_latent_selection_runner.py","stage":ROOT/"sua_exploration/scripts/m1_decoder_latent_selection_stage.py","aggregator":ROOT/"sua_exploration/scripts/aggregate_m1_decoder_latent_selection_arms.py","tests":ROOT/"sua_exploration/tests/test_m1_decoder_latent_selection_stage.py","previous":OUT/"selection_scorer_prelaunch_v2.json"}
def sha(p):
 d=hashlib.sha256()
 with p.open("rb") as h:
  for b in iter(lambda:h.read(1<<20),b""):d.update(b)
 return d.hexdigest()
def main():
 out=OUT/"selection_scorer_prelaunch_v3.json"
 if out.exists():raise FileExistsError(out)
 s=FILES["runner"].read_text()
 must=["--device","--decode-batch-size","torch.no_grad","for start in range(0, neural.shape[0], batch_size)","--smoke-mini-batches",'"optimizer_steps": 0','"backward_calls": 0','"trainable_parameters": 0',"if name == OUTER_LEFT_OUT","query_end_trial=210"]
 if any(x not in s for x in must):raise ValueError("missing GPU-forward-only/scoping guard")
 p={"schema_version":"m1_dla_selection_scorer_prelaunch_v3","created_utc":datetime.now(timezone.utc).isoformat(),"status":"gpu_forward_scorer_implemented_not_executed_pending_root_launch","execution":{"devices":["cuda:0","cuda:1","cpu"],"forward_only":True,"optimizer_steps":0,"backward_calls":0,"trainable_parameters":0,"decode_batching":True,"smoke_mode":True,"per_arm_outputs":True,"fail_closed_aggregate":True},"scope":{"heldin_only":True,"sources":["ses-20120924","ses-20120926","ses-20120927","ses-20120928"],"outer_leftout":"ses-20120926","support":[0,10],"query":[10,210],"report":False,"heldout":False,"EvalAI":False},"hashes":{k:{"path":str(v.relative_to(ROOT)),"sha256":sha(v)} for k,v in FILES.items()},"not_executed":["NWB_or_query_loading","smoke","full_scoring","report","heldout","EvalAI","optimizer","backward"]}
 out.write_text(json.dumps(p,indent=2,sort_keys=True)+"\n");(OUT/"selection_scorer_prelaunch_v3.sha256").write_text(f"{sha(out)}  selection_scorer_prelaunch_v3.json\n");print(out)
if __name__=="__main__":main()
