"""Independent seal and one-shot final gate for the Flat DANDI follow-up."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .. import data, protocol
from ..common import atomic_json, digest, fresh_directory, load_records, record_binding, score_predictions, sha256, verify_stats, source_hashes
from ..final_access import FinalAccess
from ..finalize import _add_artifact, _finite_number, _prepared_binding, _read_json, _same_state, _torch_payload, _validate_dev_metrics, _encoder_entries
from .contract import FLAT_RECIPE, FLAT_SCHEMA, flat_config_metadata, flat_training_source_hashes
from .model import validate_flat_model

FLAT_SEAL_SCHEMA = "dandi688_v2_flat_followup_selection_seal"
FLAT_SEAL_STATUS = "FROZEN_FORMAL_FLAT_SELECTION"
FLAT_FINAL_SCHEMA = "dandi688_v2_flat_followup_final_score"


def _canonical(payload: Mapping[str, Any]) -> str:
    import hashlib
    body = dict(payload); body.pop("sha256", None)
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _cells(plan: Mapping[str, Any]) -> tuple[str, ...]:
    cells = plan.get("flat_final_cells", plan.get("final_cells", plan.get("cells")))
    if not isinstance(cells, list) or len(cells) not in (6, 10) or len(set(cells)) != len(cells) or not all(isinstance(x, str) for x in cells):
        raise ValueError("Flat plan must declare exactly six or ten unique flat_final_cells")
    expected6 = {f"{arm}_flat_{rep}" for arm in ("full", "activity", "raw_set") for rep in ("sua", "pmua")}
    expected10 = expected6 | {f"full_flat_{rep}_s{seed}" for rep in ("sua", "pmua") for seed in (43, 44)}
    if set(cells) not in (expected6, expected10):
        raise ValueError("Flat plan cell roster is not the canonical 6/10 matrix")
    return tuple(cells)


def _spec(cell: str) -> tuple[str, str, int]:
    seed = 42
    base = cell
    if cell.endswith("_s43") or cell.endswith("_s44"):
        base, tail = cell.rsplit("_s", 1); seed = int(tail)
    for arm in ("full", "activity", "raw_set"):
        for rep in ("sua", "pmua"):
            if base == f"{arm}_flat_{rep}": return arm, rep, seed
    raise ValueError(f"invalid Flat cell: {cell}")


def _validate_flat_sources(receipt: Mapping[str, Any]) -> None:
    recorded = receipt.get("code_hashes")
    current = flat_training_source_hashes()
    if not isinstance(recorded, dict) or recorded != current:
        raise ValueError("Flat training execution-source hashes differ")


def _entry(cell: str, selection_path: Path, artifacts: dict[str, str], *, prepared_cache: Path,
           encoders: Mapping[str, Path]) -> dict[str, Any]:
    arm, rep, seed = _spec(cell); selection_path = Path(selection_path).resolve(); sel = _read_json(selection_path)
    if sel.get("schema") != FLAT_SCHEMA + "_selection" or sel.get("status") != "FORMAL" or sel.get("final_sessions_opened") != 0:
        raise ValueError(f"{cell} is not a formal Flat selection")
    ckpt = Path(sel.get("checkpoint", "")).resolve()
    if not ckpt.is_file() or sel.get("checkpoint_sha256") != sha256(ckpt): raise ValueError(f"{cell} checkpoint hash mismatch")
    root = selection_path.parent; receipt = _read_json(root / "receipt.json"); run_protocol = _read_json(root / "protocol.json"); stats = _read_json(root / "source_stats.json")
    if any(x.get("status") != "FORMAL" or x.get("final_sessions_opened") != 0 for x in (receipt, run_protocol)):
        raise ValueError(f"{cell} receipt/protocol is not formal no-final")
    if receipt.get("schema") != FLAT_SCHEMA + "_training" or run_protocol.get("schema") != FLAT_SCHEMA + "_training":
        raise ValueError(f"{cell} uses non-Flat schema")
    if receipt.get("recipe") != FLAT_RECIPE or receipt.get("flat_recipe") != FLAT_RECIPE or receipt.get("flat_config") != flat_config_metadata():
        raise ValueError(f"{cell} Flat recipe/config mismatch")
    _validate_flat_sources(receipt)
    core=("schema","status","stage","arm","representation","seed","protocol","recipe","flat_recipe","flat_config","zero_slope_proof","source_sessions","source_binding","source_stats_sha256","code_hashes","encoder_checkpoint_sha256","base_encoder_binding","final_sessions_opened")
    if any(run_protocol.get(k) != receipt.get(k) for k in core): raise ValueError(f"{cell} protocol/receipt core metadata mismatch")
    source, dev = load_records(prepared_cache, rep, "train"), load_records(prepared_cache, rep, "dev")
    verify_stats(stats, source)
    if (receipt.get("protocol") != protocol.protocol_dict() or receipt.get("representation") != rep or receipt.get("arm") != arm
        or receipt.get("stage") != "train" or receipt.get("seed") != seed or receipt.get("completed") is not True
        or receipt.get("source_sessions") != list(protocol.TRAIN_SESSIONS) or receipt.get("source_binding") != record_binding(source)
        or receipt.get("source_stats_sha256") != stats.get("sha256") or receipt.get("global_step") != 75960
        or receipt.get("actual_budget") != {"segments":24,"updates_per_segment":3165,"batch":32}):
        raise ValueError(f"{cell} training provenance/budget mismatch")
    curve = receipt.get("segments")
    if not isinstance(curve, list) or len(curve) != 24: raise ValueError(f"{cell} lacks 24 development checkpoints")
    scores=[]; found=None
    for ix,row in enumerate(curve):
        if not isinstance(row,dict) or row.get("segment") != ix+1 or row.get("global_step") != (ix+1)*3165 or not _finite_number(row.get("mean_loss")):
            raise ValueError(f"{cell} malformed training curve")
        curve_ckpt=root / str(row.get("checkpoint", ""))
        if curve_ckpt != root / f"segment_{ix+1:02d}.pt" or not curve_ckpt.is_file() or row.get("checkpoint_sha256") != sha256(curve_ckpt):
            raise ValueError(f"{cell} checkpoint curve hash mismatch at segment {ix+1}")
        scores.append(_validate_dev_metrics(row.get("development"), dev, label=f"{cell} segment {ix+1}"))
        if row.get("checkpoint") == ckpt.name and row.get("checkpoint_sha256") == sha256(ckpt): found=ix
    if found is None or ckpt != root / f"segment_{found+1:02d}.pt" or sel.get("rule") != "earliest_max_equal_session_dev_r2" or sel.get("dev_sessions") != list(protocol.DEV_SESSIONS) or found != scores.index(max(scores)) or sel.get("mean_dev_r2") != scores[found]:
        raise ValueError(f"{cell} selection is not earliest complete-dev maximum")
    payload=_torch_payload(ckpt,label=cell)
    if (payload.get("schema") != FLAT_SCHEMA+"_checkpoint" or payload.get("status") != "FORMAL" or payload.get("stage") != "train" or payload.get("protocol") != protocol.protocol_dict() or payload.get("source_stats") != stats or payload.get("recipe") != FLAT_RECIPE or payload.get("flat_recipe") != FLAT_RECIPE
        or payload.get("flat_config") != flat_config_metadata() or payload.get("zero_slope_proof",{}).get("effective_bias") != "allzero"
        or payload.get("arm") != arm or payload.get("representation") != rep or payload.get("seed") != seed or payload.get("global_step") != (found+1)*3165):
        raise ValueError(f"{cell} checkpoint is learned/non-flat or mismatched")
    from .training import load_trained_model
    model, _ = load_trained_model(ckpt, device="cpu")
    validate_flat_model(model, require_frozen_full=arm == "full")
    if arm == "full":
        enc=Path(encoders[rep]).resolve(); base=_torch_payload(enc,label=f"{rep} base encoder")
        state=payload.get("model_state",{}); base_state=base.get("encoder_state")
        flat_state={k.removeprefix("encoder."):v for k,v in state.items() if k.startswith("encoder.")}
        if not isinstance(base_state,dict) or not flat_state: raise ValueError(f"{cell} lacks Full encoder state")
        _same_state(flat_state,base_state,label=f"{cell} frozen encoder")
        if payload.get("encoder_checkpoint_sha256") != sha256(enc): raise ValueError(f"{cell} encoder checkpoint provenance mismatch")
        expected_binding={"path":str(enc),"checkpoint_sha256":sha256(enc),"receipt_sha256":sha256(enc.with_suffix(".json")),"representation":rep,"schema":_read_json(enc.with_suffix(".json")).get("schema"),"source_stats_sha256":stats.get("sha256"),"global_step":75960}
        if receipt.get("base_encoder_binding") != expected_binding or sel.get("base_encoder_binding") != expected_binding or payload.get("base_encoder_binding") != expected_binding: raise ValueError(f"{cell} base encoder binding mismatch")
        _add_artifact(artifacts,enc)
    paths=[selection_path,ckpt,root/"receipt.json",root/"protocol.json",root/"source_stats.json"]
    return {"status":"SELECTED","kind":"flat_neural","arm":arm,"representation":rep,"seed":seed,"checkpoint_sha256":sha256(ckpt),"selection_path":_add_artifact(artifacts,selection_path),"artifact_paths":[_add_artifact(artifacts,p) for p in paths]}


def seal_selection(dest: Path, *, prepared_receipt: Path, plan_path: Path, selections: Mapping[str, Path], base_final_receipt: Path, encoder_checkpoints: Mapping[str, Path]) -> dict[str, Any]:
    from .run_campaign import _load_plan, _verify_plan_dependencies
    plan_path=Path(plan_path).resolve(); plan=_load_plan(plan_path.parent); _verify_plan_dependencies(plan); cells=_cells(plan)
    if (Path(prepared_receipt).resolve() != Path(plan["prepared_receipt"]["path"]).resolve() or sha256(Path(prepared_receipt)) != plan["prepared_receipt"]["sha256"]
        or Path(base_final_receipt).resolve() != Path(plan["base_final_receipt"]["path"]).resolve() or sha256(Path(base_final_receipt)) != plan["base_final_receipt"]["sha256"]
        or any(Path(encoder_checkpoints[k]).resolve() != Path(plan["encoder_checkpoints"][k]["path"]).resolve() or sha256(Path(encoder_checkpoints[k])) != plan["encoder_checkpoints"][k]["sha256"] for k in ("sua","pmua"))):
        raise ValueError("seal arguments differ from frozen Flat plan bindings")
    if plan.get("base_results_already_observed") is not True: raise ValueError("Flat follow-up plan must record observed base results")
    if set(selections) != set(cells) or set(encoder_checkpoints) != {"sua","pmua"}: raise ValueError("Flat selection/encoder roster mismatch")
    base_final_receipt=Path(base_final_receipt).resolve(); base=_read_json(base_final_receipt)
    if base.get("status") != "FINAL_SCORED" or len(base.get("results",{})) != 19 or base.get("final_sessions_opened") != 6: raise ValueError("base19 final receipt is not complete")
    artifacts={}; prepared=_prepared_binding(Path(prepared_receipt),artifacts)
    checked_encoders=_encoder_entries(encoder_checkpoints,artifacts,prepared_cache=Path(prepared_receipt).parent)
    task_paths={task["cell"]:(plan_path.parent / task["run"] / "selection.json").resolve() for task in plan["training_tasks"]}
    if any(Path(selections[cell]).resolve() != task_paths[cell] for cell in cells): raise ValueError("Flat selections do not use frozen plan run paths")
    entries={cell:_entry(cell,Path(selections[cell]),artifacts,prepared_cache=Path(prepared_receipt).parent,encoders=encoder_checkpoints) for cell in cells}
    _add_artifact(artifacts,plan_path); _add_artifact(artifacts,base_final_receipt)
    # Training receipts bind their training sources; the seal additionally freezes
    # every extension scorer/gate byte used after the follow-up final access.
    for source in Path(__file__).resolve().parent.glob("*.py"):
        _add_artifact(artifacts, source)
    # Freeze all base execution sources, including data/model/scoring helpers.
    workspace=protocol.WORKSPACE_ROOT
    for relative, expected in source_hashes().items():
        source=workspace / relative
        if sha256(source) != expected: raise ValueError("base execution source drift")
        _add_artifact(artifacts, source)
    payload={"schema":FLAT_SEAL_SCHEMA,"status":FLAT_SEAL_STATUS,"path":str(Path(dest).resolve()/"selection_seal.json"),"protocol":protocol.protocol_dict(),"plan_path":str(plan_path),"plan_sha256":sha256(plan_path),"base_final_receipt":str(base_final_receipt),"base_final_receipt_sha256":sha256(base_final_receipt),"base_results_already_observed":True,"prepared":prepared,"encoders":{k:{x:y for x,y in v.items() if x!="encoder_state"} for k,v in checked_encoders.items()},"required_cells":list(cells),"cell_selections":entries,"artifacts":dict(sorted(artifacts.items())),"final_sessions_opened":0}
    payload["sha256"]=_canonical(payload); destination=fresh_directory(dest); atomic_json(destination/"selection_seal.json",payload); return payload


class FlatFinalAccess(FinalAccess):
    @classmethod
    def from_manifest(cls,path:Path):
        path=Path(path).resolve()
        if not path.is_file(): raise PermissionError("Flat final seal is missing")
        p=_read_json(path)
        from .run_campaign import _load_plan, _verify_plan_dependencies
        plan_path=Path(p.get("plan_path", "")).resolve()
        try:
            plan=_load_plan(plan_path.parent); _verify_plan_dependencies(plan)
        except Exception as error:
            raise PermissionError("Flat final seal plan/dependency validation failed") from error
        if p.get("schema")!=FLAT_SEAL_SCHEMA or p.get("status")!=FLAT_SEAL_STATUS or p.get("protocol")!=protocol.protocol_dict() or p.get("sha256")!=_canonical(p): raise PermissionError("invalid Flat final seal")
        artifacts=p.get("artifacts"); cells=p.get("required_cells"); entries=p.get("cell_selections")
        if p.get("plan_sha256") != sha256(plan_path) or p.get("base_final_receipt") != plan["base_final_receipt"]["path"] or p.get("base_final_receipt_sha256") != plan["base_final_receipt"]["sha256"] or p.get("prepared",{}).get("path") != plan["prepared_receipt"]["path"] or p.get("prepared",{}).get("sha256") != plan["prepared_receipt"]["sha256"]: raise PermissionError("Flat seal plan binding mismatch")
        if any(p.get("encoders",{}).get(rep,{}).get("path") != plan["encoder_checkpoints"][rep]["path"] or p["encoders"][rep].get("sha256") != plan["encoder_checkpoints"][rep]["sha256"] for rep in ("sua","pmua")): raise PermissionError("Flat seal encoder binding mismatch")
        if not isinstance(artifacts,dict) or not artifacts or tuple(cells or ()) != _cells(_read_json(Path(p["plan_path"]))) or not isinstance(entries,dict) or set(entries)!=set(cells) or p.get("final_sessions_opened") != 0: raise PermissionError("Flat seal roster mismatch")
        for raw,d in artifacts.items():
            if not Path(raw).is_absolute() or not isinstance(d,str) or len(d)!=64 or not Path(raw).is_file() or sha256(Path(raw))!=d: raise PermissionError("Flat seal artifact changed")
        for cell,entry in entries.items():
            paths=entry.get("artifact_paths", [])
            checkpoints=[x for x in paths if str(x).endswith(".pt")]
            if entry.get("status") != "SELECTED" or not isinstance(entry.get("checkpoint_sha256"),str) or len(entry["checkpoint_sha256"]) != 64 or not isinstance(paths,list) or not paths or any(x not in artifacts for x in paths) or len(checkpoints) != 1 or artifacts[checkpoints[0]] != entry["checkpoint_sha256"]: raise PermissionError(f"Flat selected entry invalid: {cell}")
            arm,rep,seed=_spec(cell)
            if (entry.get("arm"),entry.get("representation"),entry.get("seed")) != (arm,rep,seed): raise PermissionError(f"Flat selected entry metadata mismatch: {cell}")
        required=[p.get("plan_path"),p.get("base_final_receipt"),p.get("prepared",{}).get("path")]+[x.get("path") for x in p.get("encoders",{}).values() if isinstance(x,dict)]
        if len(required)!=5 or any(x not in artifacts for x in required): raise PermissionError("Flat seal missing bound plan/base/prepared/encoder artifact")
        return cls(path,sha256(path),tuple(sorted(artifacts.items())))
    def validate(self):
        fresh=self.from_manifest(self.manifest_path)
        if fresh.manifest_sha256!=self.manifest_sha256 or fresh.artifact_hashes!=self.artifact_hashes: raise PermissionError("Flat seal changed")
    def authorize(self,session_id:str):
        self.validate()
        if session_id not in protocol.FINAL_SESSIONS: raise PermissionError("Flat gate only authorizes final roster")


def score_final(seal_path:Path,dest:Path,*,raw_root:Path=protocol.DEFAULT_RAW_ROOT,device:str="cpu",score_cell:Callable|None=None)->dict[str,Any]:
    cap=FlatFinalAccess.from_manifest(seal_path); cap.validate(); seal=_read_json(cap.manifest_path)
    if seal.get("final_sessions_opened")!=0: raise PermissionError("Flat seal already records final access")
    dest=fresh_directory(dest)
    claim=cap.manifest_path.parent/".final_score_claim"
    try: claim.open("x").write(cap.manifest_sha256)
    except FileExistsError as e: raise PermissionError("Flat final claim already exists") from e
    atomic_json(dest/"final_score_started.json",{"status":"FINAL_SCORE_STARTED","final_sessions_opened":0,"seal_sha256":cap.manifest_sha256})
    records={}; log=[]
    for sid in protocol.FINAL_SESSIONS:
        cap.authorize(sid); records[sid]=data.load_pair(sid,purpose="final",final_access=cap,raw_root=raw_root,access_log=log)
    if [x["session_id"] for x in log]!=list(protocol.FINAL_SESSIONS): raise RuntimeError("Flat final roster not opened exactly once")
    prepared_cache=Path(seal["prepared"]["path"]).parent
    for rep in ("sua","pmua"):
        canonical=load_records(prepared_cache,rep,"train")[0].metadata["canonical_electrode_keys"]
        if any(records[s][rep].metadata["canonical_electrode_keys"] != canonical for s in protocol.FINAL_SESSIONS): raise RuntimeError("Flat final canonical electrode table mismatch")
    # The follow-up must score precisely the truth/query rows used by completed base19.
    base=_read_json(Path(seal["base_final_receipt"]))
    for rep in ("sua","pmua"):
        base_row=base["results"][f"full_{rep}"]
        base_sessions=base_row.get("predictions",{}).get("sessions",{})
        base_counts={x.get("session_id"):x.get("n_queries") for x in base_row.get("result",{}).get("metrics",{}).get("sessions",[]) if isinstance(x,dict)}
        for sid in protocol.FINAL_SESSIONS:
            rec=records[sid][rep]; expected=base_sessions.get(sid,{})
            actual=rec.metadata["array_sha256"]
            if (expected.get("query_indices_sha256") != actual.get("query_indices") or expected.get("velocity_sha256") != actual.get("velocity")
                or base_counts.get(sid) != len(rec.query_indices)):
                raise RuntimeError(f"Flat/base final query-truth binding mismatch: {sid}.{rep}")
    from .training import load_trained_model
    from ..training import predict_network
    rows={}
    for cell,entry in seal["cell_selections"].items():
        cap.validate()
        if score_cell: result=dict(score_cell(cell,records,entry))
        else:
            model,stats=load_trained_model(next(Path(x) for x in entry["artifact_paths"] if x.endswith(".pt")),device=device); validate_flat_model(model,require_frozen_full=entry["arm"]=="full")
            vals=[]; preds={}
            for sid in protocol.FINAL_SESSIONS:
                rec=records[sid][entry["representation"]]; pred=predict_network(model,rec,stats); preds[sid]=pred; vals.append(score_predictions(rec,pred))
            from ..common import aggregate_scores
            result={"metrics":aggregate_scores(vals),"_predictions":preds,"selected":True}
        preds=result.pop("_predictions",None)
        if result.get("selected") is not True or result.get("status", "SCORED") != "SCORED" or not isinstance(preds, Mapping) or set(preds) != set(protocol.FINAL_SESSIONS): raise RuntimeError("Flat scorer rejected/incomplete predictions")
        rep=entry["representation"]
        metrics=result.get("metrics",{}); sessions=metrics.get("sessions") if isinstance(metrics,dict) else None
        if metrics.get("n_sessions") != 6 or not isinstance(sessions,list) or [x.get("session_id") for x in sessions if isinstance(x,dict)] != list(protocol.FINAL_SESSIONS) or not _finite_number(metrics.get("mean_r2")) or any(not isinstance(x,dict) or not _finite_number(x.get("r2")) or x.get("n_queries") != len(records[x.get("session_id")][rep].query_indices) for x in sessions): raise RuntimeError("Flat final metrics incomplete")
        if abs(metrics["mean_r2"] - sum(x["r2"] for x in sessions)/6) > 1e-12: raise RuntimeError("Flat final mean mismatch")
        import numpy as np
        for sid,pred in preds.items():
            truth=np.asarray(records[sid][rep].velocity[records[sid][rep].query_indices])
            value=np.asarray(pred)
            if value.shape != truth.shape or value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all(): raise RuntimeError("Flat prediction shape/finite mismatch")
        info=None
        if preds is not None:
            out=dest/f"{cell}.final_predictions.npz"; np.savez_compressed(out,**preds)
            info={"path":str(out),"sha256":sha256(out),"sessions":{s:{"prediction_sha256":digest(v),"query_indices_sha256":records[s][entry["representation"]].metadata["array_sha256"]["query_indices"],"velocity_sha256":records[s][entry["representation"]].metadata["array_sha256"]["velocity"]} for s,v in preds.items()}}
        rows[cell]={"status":result.get("status","SCORED"),"result":result,"predictions":info}
    if set(rows)!=set(seal["required_cells"]): raise RuntimeError("Flat final coverage mismatch")
    receipt={"schema":FLAT_FINAL_SCHEMA,"status":"FINAL_SCORED","seal_path":str(cap.manifest_path),"seal_sha256":cap.manifest_sha256,"roster":list(protocol.FINAL_SESSIONS),"results":rows,"access_log":log,"final_sessions_opened":6}; atomic_json(dest/"final_score_receipt.json",receipt); return receipt
