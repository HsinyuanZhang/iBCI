"""Supplement the immutable carrier NPZ with unambiguous support-row provenance.

The original NPZ's `support_row_ids` was a unit-axis placeholder and its
`support_emg_trial_ids` label was ambiguous for full-source EMG.  This writes
new, separately hashed evidence; it never mutates the carrier bytes/receipt.
"""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
import numpy as np
from pynwb import NWBHDF5IO
from . import bank, plan
from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data, plan as parent_plan
OUT_NPZ=plan.RESULT_ROOT/"rSyn3-refit-v1.source-only.provenance-supplement.npz"
OUT_JSON=plan.RESULT_ROOT/"rSyn3-refit-v1.source-only.provenance-supplement.receipt.json"
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def _ad(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def run():
 loaded=bank.load()
 if OUT_NPZ.exists() or OUT_JSON.exists():
  if not OUT_NPZ.exists() or not OUT_JSON.exists(): raise RuntimeError("partial provenance supplement")
  receipt=json.loads(OUT_JSON.read_text())
  if _sha(OUT_NPZ)!=receipt["supplement_npz_sha256"]: raise RuntimeError("supplement drift")
  return receipt
 root=parent_data.repo_root(); payload={}; rows={}
 for name in plan.SOURCE_SESSIONS:
  path=parent_data.require_source_path(root/parent_plan.SOURCE_RELATIVE[name]); rec=fold_data.load_fold_session(path,role="source")
  emg_rows=np.flatnonzero(np.asarray(rec.emg_trial_ids)<plan.SUPPORT_TRIALS).astype(np.int64)
  rate_rows=np.flatnonzero(np.asarray(rec.rate_trial_ids)<plan.SUPPORT_TRIALS).astype(np.int64)
  # Unit IDs are read in roster order, the same order used for spike/rate columns.
  with NWBHDF5IO(str(path),"r",load_namespaces=True) as io: unit_ids=np.asarray(io.read().units.id[:])
  if len(unit_ids)!=rec.rates.shape[1] or len(unit_ids)!=plan.N_UNITS: raise RuntimeError(f"unit roster mismatch {name}")
  payload[f"support_emg_bin_row_indices/{name}"]=emg_rows; payload[f"support_rate_bin_row_indices/{name}"]=rate_rows
  payload[f"support_emg_trial_ids/{name}"]=np.asarray(rec.emg_trial_ids)[emg_rows]; payload[f"support_rate_trial_ids/{name}"]=np.asarray(rec.rate_trial_ids)[rate_rows]
  payload[f"nwb_unit_ids_in_rate_column_order/{name}"]=unit_ids; payload[f"rate_column_indices/{name}"]=np.arange(len(unit_ids),dtype=np.int64)
  rows[name]={"source_file_sha256":parent_data.file_sha256(path),"support_emg_bin_count":int(len(emg_rows)),"support_rate_bin_count":int(len(rate_rows)),"support_emg_bin_row_indices_sha256":_ad(emg_rows),"support_rate_bin_row_indices_sha256":_ad(rate_rows),"support_emg_trial_ids_sha256":_ad(payload[f"support_emg_trial_ids/{name}"]),"support_rate_trial_ids_sha256":_ad(payload[f"support_rate_trial_ids/{name}"]),"nwb_unit_ids_sha256":_ad(unit_ids),"rate_column_indices_sha256":_ad(payload[f"rate_column_indices/{name}"])}
 np.savez_compressed(OUT_NPZ,**payload)
 receipt={"schema":"m1_optimized_v2_carrier_provenance_supplement_v1","carrier_revision":plan.CARRIER_REVISION,"carrier_npz_sha256":loaded["receipt"]["digests"]["npz"],"supplement_npz_sha256":_sha(OUT_NPZ),"immutable_original_npz_mutated":False,"legacy_npz_field_disclosure":{"support_row_ids":"ambiguous unit-axis placeholder; superseded by supplement support_*_bin_row_indices","support_emg_trial_ids":"full-source EMG trial IDs; superseded by selected M10 support_emg_trial_ids","unit_roster":"arange(64) positional placeholder; superseded by nwb_unit_ids_in_rate_column_order"},"selection":"exact masks used by _mask_budget: trial_id < 10","source_sessions":list(plan.SOURCE_SESSIONS),"rows":rows,"outer_path_resolved":False,"outer_query_opened":False,"created":datetime.now(timezone.utc).isoformat()}
 OUT_JSON.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n"); return receipt
if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
