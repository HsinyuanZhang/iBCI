"""Static/payload audit entrypoint for chronological H1 leave-last-two results."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for item in (str(HERE),str(ROOT/"src"),str(ROOT.parent/"btransform_unified_v1"/"src"),str(ROOT.parent)):
    if item not in sys.path:sys.path.insert(0,item)
sys.path.remove(str(HERE));sys.path.insert(0,str(HERE))
import h1_prepare as prep
import h1_train as train

def require(v,msg):
    if not v:raise RuntimeError(msg)
def read(path):return json.loads(path.read_text())

def audit_prepared(prepared:Path,arm:str)->dict:
    contract=prep.split_contract(); manifests={}
    for surface,expected in (("source",contract["source_sessions"]),("target",contract["target_sessions"])):
        manifest=read(prepared/surface/"manifest.json");manifests[surface]=manifest
        require(manifest.get("schema")==prep.PREPARE_SCHEMA and prep.matches_split_contract(manifest),f"{surface}: split contract mismatch")
        require(sorted(manifest["records"])==sorted(expected),f"{surface}: roster mismatch")
        authority=manifest["source_authority"]
        require(tuple(authority.get("source_sessions",()))==tuple(contract["source_sessions"]) and authority.get("target_records_opened")==0,f"{surface}: source authority leak")
        files=manifest["source_authority_files"]
        require(files.get("json_sha256")==train.sha_file(prepared/"source"/"source_hc_plan.json") and files.get("arrays_sha256")==train.sha_file(prepared/"source"/"source_hc_plan_arrays.npz"),f"{surface}: frozen authority file hashes")
        for session,row in manifest["records"].items():
            train.audit_npz(prepared/surface/f"{session}.npz",row,arm)
            if surface=="source":
                val=row["validation"];train.audit_npz(prepared/surface/f"{session}.val.npz",val,arm)
                require(not(set(map(float,row["query_trials"]))&set(map(float,val["query_trials"]))),f"{session}: source train/val overlap")
    require(not(set(manifests["source"]["records"])&set(manifests["target"]["records"])),"source/target session overlap")
    manifests["selection_seals"]=prep.validate_selection_seals(prepared)
    return manifests

def audit_run(prepared:Path,dest:Path,arm:str,manifests:dict)->dict:
    contract=prep.split_contract();meta=read(dest/"run_meta.json");receipt=read(dest/"train_receipt.json");selection=read(dest/"selection.json")
    meta_without_recipe={k:v for k,v in meta.items() if k!="recipe_binding"};recipe=meta.get("recipe_binding")
    require(meta_without_recipe.get("split")==contract and meta_without_recipe.get("arm")==arm and receipt.get("meta")==meta_without_recipe,f"{arm}: run metadata binding")
    require(isinstance(recipe,str) and receipt.get("recipe_binding")==recipe and train.bind_hash(meta_without_recipe)==recipe,f"{arm}: train recipe binding")
    require(meta.get("source_manifest_sha256")==train.sha_file(prepared/"source"/"manifest.json"),f"{arm}: source manifest hash")
    preflight_path=dest/"preflight.json";preflight=read(preflight_path)
    require(preflight.get("schema")=="h1_chronological_last2_preflight_v1" and preflight.get("status")=="PASSED" and preflight.get("split")==contract and preflight.get("arm")==arm and preflight.get("source_only") is True and preflight.get("target_records_opened")==0 and preflight.get("source_manifest_sha256")==train.sha_file(prepared/"source"/"manifest.json"),f"{arm}: current source-only preflight proof")
    source_code=meta_without_recipe.get("source_code_sha256",{})
    require(bool(source_code) and all(Path(path).is_file() and train.sha_file(Path(path))==digest for path,digest in source_code.items()),f"{arm}: source code hashes")
    source_files=receipt.get("source_files",{})
    require(set(source_files)==set(prep.SOURCE_SESSIONS),f"{arm}: source file roster")
    for session,files in source_files.items():
        require(files=={"train":train.sha_file(prepared/"source"/f"{session}.npz"),"validation":train.sha_file(prepared/"source"/f"{session}.val.npz")},f"{arm}: source file hashes {session}")
    selected=receipt["selected"];require(selection==selected and receipt["selection"].startswith("earliest maximum source validation equal_date_mean"),f"{arm}: selection receipt")
    seal=manifests["selection_seals"][arm];receipt_path=(dest/"train_receipt.json").resolve()
    require(Path(seal["train_receipt"]).resolve()==receipt_path and seal["train_receipt_sha256"]==train.sha_file(receipt_path) and seal["selected_epoch"]==selected["epoch"] and seal["selected"]==selected,f"{arm}: selection seal/run receipt binding")
    curve=receipt["curve"];require(len(curve)==32 and [r["epoch"] for r in curve]==list(range(1,33)),f"{arm}: epoch curve")
    for row in curve:
        val=row["source_val_ema"];per=val["per_session"]
        require(np.isfinite(row["train_mse"]) and np.isfinite(row["elapsed_seconds"]) and isinstance(row.get("sampler_endpoint_keep_sha256"),str) and len(row["sampler_endpoint_keep_sha256"])==64,f"{arm}: nonfinite/incomplete curve row")
        require(set(per)==set(prep.SOURCE_SESSIONS),f"{arm}: source validation roster")
        require(all(np.isfinite(item["r2"]) and item["windows"]>0 for item in per.values()),f"{arm}: nonfinite source validation")
        independent={d:float(np.mean([per[s]["r2"] for s in prep.H1_SESSIONS_BY_DATE[d]])) for d in prep.SOURCE_DATES}
        require(val["per_date"]==independent and np.isfinite(val["equal_date_mean"]) and abs(val["equal_date_mean"]-float(np.mean(list(independent.values()))))<=1e-12,f"{arm}: date-equal source metric")
    best=max(curve,key=lambda r:(r["source_val_ema"]["equal_date_mean"],-r["epoch"]));require(best==selected,f"{arm}: earliest maximum selection")
    require(set(receipt["checkpoints"])=={f"ema_epoch_{epoch:03d}" for epoch in range(1,33)},f"{arm}: checkpoint roster")
    for name,digest in receipt["checkpoints"].items():require(train.sha_file(dest/f"{name}.pt")==digest,f"{arm}: checkpoint checksum {name}")
    score=read(dest/"target_score.json");require(score.get("split")==contract and score.get("arm")==arm and score["selected_source_epoch"]==selected["epoch"],f"{arm}: score split/epoch")
    require(score["target_manifest_sha256"]==train.sha_file(prepared/"target"/"manifest.json"),f"{arm}: target manifest hash")
    payloads={}
    for label in ("selected_ema","fixed_e32_ema"):
        checkpoint=score[label];epoch=int(checkpoint["epoch"]);cp=dest/f"ema_epoch_{epoch:03d}.pt"
        require(epoch==(selected["epoch"] if label=="selected_ema" else 32),f"{arm}: score label epoch {label}")
        require(checkpoint["checkpoint_sha256"]==train.sha_file(cp)==receipt["checkpoints"].get(f"ema_epoch_{epoch:03d}"),f"{arm}: score checkpoint binding {label}")
        target=score[label]["target"];per=target["per_session"];require(set(per)==set(prep.TARGET_SESSIONS),f"{arm}: target roster {label}")
        independent={d:float(np.mean([per[s]["r2"] for s in prep.H1_SESSIONS_BY_DATE[d]])) for d in prep.TARGET_DATES}
        require(target["per_date"]==independent and abs(target["equal_date_mean"]-float(np.mean(list(independent.values()))))<=1e-12,f"{arm}: target date metric {label}")
        for s,row in per.items():
            artifact=Path(row["artifact"]);require(artifact.is_file() and train.sha_file(artifact)==row["artifact_sha256"],f"{arm}: target artifact {label}/{s}")
            with np.load(artifact,allow_pickle=False) as z:
                require(set(z.files)=={"prediction","target","endpoints"},f"{arm}: target artifact schema {label}/{s}")
                require(train.sha_array(z["prediction"])==row["prediction_sha256"] and train.sha_array(z["target"])==row["y_sha256"] and train.sha_array(z["endpoints"])==row["endpoint_sha256"],f"{arm}: target artifact payload {label}/{s}")
                with np.load(prepared/"target"/f"{s}.npz",allow_pickle=False) as source:
                    require(np.array_equal(z["target"],source["y"]) and np.array_equal(z["endpoints"],source["ends"]),f"{arm}: target labels/endpoints differ from prepared {label}/{s}")
                require(abs(float(train.variance_weighted_r2(z["target"],z["prediction"]))-row["r2"])<=1e-12,f"{arm}: target R2 recomputation {label}/{s}")
                payloads[(label,s)]=(row["y_sha256"],row["endpoint_sha256"])
    require(all(payloads[("selected_ema",s)]==payloads[("fixed_e32_ema",s)] for s in prep.TARGET_SESSIONS),f"{arm}: selected/e32 labels or coordinates differ")
    return {"arm":arm,"selected_epoch":selected["epoch"],"selected_source_equal_date_mean":selected["source_val_ema"]["equal_date_mean"],"target_selected_equal_date_mean":score["selected_ema"]["target"]["equal_date_mean"],"target_fixed_e32_equal_date_mean":score["fixed_e32_ema"]["target"]["equal_date_mean"],"target_payloads":payloads,"sampler_curve":[(row["epoch"],row["step"],row["sampler_endpoint_keep_sha256"]) for row in curve],"source_files":source_files,"initial_parameter_sha256":meta_without_recipe["initial_parameter_sha256"],"preflight_sha256":train.sha_file(preflight_path)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--run-root",type=Path,required=True);p.add_argument("--arms",nargs="+",choices=("Z_NONE","B_ACTIVITY_ONLY","D_JOINT"),required=True);p.add_argument("--dest",type=Path,required=True);a=p.parse_args()
    required_arms=("Z_NONE","B_ACTIVITY_ONLY","D_JOINT")
    require(len(a.arms)==len(required_arms) and set(a.arms)==set(required_arms),"audit requires exactly Z_NONE, B_ACTIVITY_ONLY, D_JOINT")
    manifests=audit_prepared(a.prepared.resolve(),"D_JOINT");runs=[audit_run(a.prepared.resolve(),a.run_root.resolve()/arm,arm,manifests) for arm in a.arms]
    reference=runs[0]["target_payloads"]
    require(all(run["target_payloads"]==reference for run in runs[1:]),"Z/B/D target labels or coordinates differ")
    sampler_reference=runs[0]["sampler_curve"];source_files_reference=runs[0]["source_files"]
    require(all(run["sampler_curve"]==sampler_reference for run in runs[1:]),"Z/B/D sampler/step curve differs")
    require(all(run["source_files"]==source_files_reference for run in runs[1:]),"Z/B/D source file bindings differ")
    by_arm={run["arm"]:run for run in runs}
    require(by_arm["B_ACTIVITY_ONLY"]["initial_parameter_sha256"]==by_arm["D_JOINT"]["initial_parameter_sha256"],"B/D initial parameter hash differs")
    for run in runs: run.pop("target_payloads");run.pop("sampler_curve");run.pop("source_files");run.pop("initial_parameter_sha256");run["preflight_sha256"]=run.pop("preflight_sha256")
    out={"schema":"h1_chronological_last2_audit_v1","status":"PASSED","split":prep.split_contract(),"runs":runs,"prepared_manifest_sha256":{s:train.sha_file(a.prepared.resolve()/s/"manifest.json") for s in ("source","target")},"source_code_sha256":{str(Path(__file__).resolve()):train.sha_file(Path(__file__).resolve()),str(HERE/"h1_prepare.py"):train.sha_file(HERE/"h1_prepare.py"),str(HERE/"h1_train.py"):train.sha_file(HERE/"h1_train.py")}}
    train.json_atomic(a.dest.resolve(),out);print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__":main()
