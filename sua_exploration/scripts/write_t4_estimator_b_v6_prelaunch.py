#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'sua_exploration'))
from scripts.audit_t4_estimator_b_v6 import prelaunch,strict
OUT=ROOT/'sua_exploration/results/t4_estimator_b_v6_prelaunch/receipt.json'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 r=prelaunch();r['writer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(strict(r),indent=2,sort_keys=True)+'\n');print(hashlib.sha256(OUT.read_bytes()).hexdigest())
if __name__=='__main__':main()
