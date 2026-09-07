"""CPU gate for carrier persistence, source-only loading, and repaired B3 initialization."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from . import bank, plan
from .data import build_source_only_datamodule, materialize_source_banks

def run():
    loaded=bank.load(); dm=build_source_only_datamodule(loaded); banks=materialize_source_banks()
    manifest=dm.get_split_manifest()
    gate={"schema":"m1_optimized_v2_cpu_preflight_v1","status":"PASS_CPU_GATE__GPU_LEARNABILITY_PENDING_COMMON_TEMPORAL","carrier_receipt":loaded["receipt"],"source_only_manifest":manifest,"bank_shapes":{n:{"e0":list(x.E0.shape),"carrier":list(x.T.shape)} for n,x in banks.items()},"checks":{"new_named_carrier":loaded["receipt"]["carrier_revision"]==plan.CARRIER_REVISION,"old_stage0_not_claimed":loaded["receipt"]["not_old_stage0"],"outer_unread":not manifest["target_path_resolved_during_fit"] and not manifest["target_query_values_read"],"no_validation_dataset":dm.val_heldin_dataset is None,"b3_identity_and_carrier_widths":all(tuple(x.E0.shape)==(64,100) and tuple(x.T.shape)==(64,4) for x in banks.values()),"sfix_decoder_weights_copied":False},"gpu_authorized_by_this_receipt":False,"updated":datetime.now(timezone.utc).isoformat()}
    required=("new_named_carrier","old_stage0_not_claimed","outer_unread","no_validation_dataset","b3_identity_and_carrier_widths")
    if not all(gate["checks"][key] for key in required) or gate["checks"]["sfix_decoder_weights_copied"]: raise RuntimeError("CPU M1 gate failed")
    out=plan.RESULT_ROOT/"cpu_source_isolation_b3_gate.json"; out.write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n"); return gate
if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
