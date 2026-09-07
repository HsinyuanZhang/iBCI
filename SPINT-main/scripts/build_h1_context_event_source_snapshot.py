#!/usr/bin/env python3
"""Build/verify the isolated ``ser_context_q4`` snapshot without Torch before SVD.

Build mode intentionally imports only NumPy/h5py-level CPU design modules at
process start.  The source-encoding SVD is completed and bound to the fixed
CPU receipt before the script imports any ``src.data`` module (which imports
Lightning/Torch and can perturb the numerical backend process state).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event

FIXED_SCREEN=REPO/"sua_exploration/results/h1_event_carrier_design_screen_v1/source_screen.json"
FIXED_SCREEN_SHA="74bbc01490432794546e7ca2fd4242fbed6f2a7ebdd65786f56035eaa49bfeb3"
FIXED_MAP_SHA="50c0c55969e6898e00846302f97a6637ac78b0af6715f33de428a9f3d845e525"
FIXED_ARRAYS={"active_mask":"1b109aa95e5bee6519b035de92dbdddad5b8660ab014f429105a934b4b101223","feature_mean":"cd50c588dc3992f1c07ba65a56553261641d7dcb794d4a6b331e7c8fe2f1b521","feature_scale":"def0b8f639903128e42d775fb635f440d1259aaf9e0be2911030b25f332de6dc","projection":"5f9a3f485ee4be33d915f4db587c67b3e8b88eabca76be34ef41a0fbbc6625df","latent_scale":"164472716c91002aa80cca18ecb79f44e53f5b384757d720596d524695ef8618"}
def _need(x,msg):
 if not x:raise ValueError(msg)
def _pure_cpu_fold0_map(data_dir:Path):
 """Must run before importing Lightning/Torch or src.data."""
 _need(event.sha256_file(FIXED_SCREEN)==FIXED_SCREEN_SHA,"fixed CPU screen receipt SHA drift")
 indexed=event.index_heldin_calib(data_dir); names=tuple(n for n in event.H1_HELDIN_SESSIONS if event.session_date(n)!="19250101")
 sessions={n:design.load_context_session(event.load_event_session(indexed[n])) for n in names}
 candidate=next(x for x in design.CANDIDATES if x.name=="ser_context_q4")
 mapping=design.fit_latent_map(sessions,outer_date="19250101",candidate=candidate); fixed=json.loads(FIXED_SCREEN.read_text())["basis_by_candidate_and_outer_date"]["ser_context_q4"]["19250101"]
 _need(mapping.map_sha256==FIXED_MAP_SHA and mapping.manifest()==fixed and mapping.manifest()["array_sha256"]==FIXED_ARRAYS,"pure CPU pre-Torch SVD differs from fixed receipt")
 return mapping
def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--verify-only",type=Path);p.add_argument("--data-dir",type=Path,default=ROOT/"data/000954");p.add_argument("--cache-dir",type=Path,default=ROOT/"pilot_artifacts/h1_context_event_carrier/shared_source_cache");p.add_argument("--snapshot",type=Path);p.add_argument("--receipt",type=Path);p.add_argument("--expected-manifest-sha256");a=p.parse_args()
 if a.verify_only:
  if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
  from src.data.h1_context_event_source_snapshot import load_snapshot
  x=load_snapshot(a.verify_only);result={"status":"PASS","receipt":str(x["receipt_path"]),"snapshot":str(x["snapshot_path"]),"manifest_sha256":x["metadata"]["manifest_sha256"],"context_map_sha256":x["latent_map"].map_sha256}
 else:
  _need(a.snapshot and a.receipt and a.expected_manifest_sha256 and len(a.expected_manifest_sha256)==64,"build needs --snapshot --receipt --expected-manifest-sha256")
  mapping=_pure_cpu_fold0_map(a.data_dir.resolve())
  # Only now is it legal to import the training stack; rebuild all cache rows
  # from the frozen mapping rather than refitting the map.
  if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
  from types import SimpleNamespace
  from src.data.h1_context_event_carrier import H1ContextSourceDataset,H1M4EBPairedBatchSampler,build_context_manifest,build_context_source_assets
  from src.data.h1_context_event_source_snapshot import write_snapshot
  records,_sessions,frozen,cache,normalizer=build_context_source_assets(a.data_dir.resolve(),frozen_map=mapping);dataset=H1ContextSourceDataset(records,cache,normalizer);sampler=H1M4EBPairedBatchSampler(dataset,batch_size=32,seed=42,max_epochs=50,cache_dir=None);manifest=build_context_manifest(records=records,latent_map=frozen,cache=cache,normalizer=normalizer,dataset=dataset,sampler=sampler);_need(event.canonical_sha256(manifest)==a.expected_manifest_sha256,"fixed-map rebuilt manifest differs from expected SHA")
  source=SimpleNamespace(_setup_done=True,latent_map=frozen,normalizer=normalizer,carrier_cache=cache,pilot_manifest=lambda:dict(manifest))
  result=write_snapshot(snapshot_path=a.snapshot,receipt_path=a.receipt,source_module=source,expected_manifest_sha256=a.expected_manifest_sha256,builder_path=Path(__file__))
 print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
