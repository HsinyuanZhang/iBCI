"""Read-only merged report for base19 and independently sealed Flat follow-up."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping
import numpy as np
from ..common import sha256, digest
from ..report import _metrics, _read_json, _verify_predictions, _validate as _validate_base
from .finalize import FLAT_FINAL_SCHEMA, FlatFinalAccess, _spec
from .. import protocol

def _values(receipt_path:Path, *, flat:bool):
    r=_read_json(receipt_path)
    if r.get("status")!="FINAL_SCORED" or (flat and r.get("schema")!=FLAT_FINAL_SCHEMA): raise ValueError("invalid final receipt")
    roster=tuple(r.get("roster",()))
    if roster!=tuple(protocol.FINAL_SESSIONS): raise ValueError("final roster mismatch")
    seal=Path(r.get("seal_path",""));
    if not seal.is_file() or r.get("seal_sha256")!=sha256(seal): raise ValueError("seal hash mismatch")
    if flat: FlatFinalAccess.from_manifest(seal).validate()
    values={}
    for cell,row in r.get("results",{}).items():
        if row.get("status")!="SCORED": raise ValueError(f"{cell} not scored")
        values[cell]=_metrics(row,roster,cell); _verify_predictions(row,roster,cell,complete=True)
        info=row["predictions"]; metric={x["session_id"]:x for x in row["result"]["metrics"]["sessions"]}
        with np.load(info["path"],allow_pickle=False) as archive:
            for day in roster:
                array=np.asarray(archive[day]); bound=info["sessions"][day]; item=metric[day]
                if array.ndim!=2 or array.shape[1]!=2 or array.shape[0] != item.get("n_queries") or not np.isfinite(array).all(): raise ValueError(f"{cell}.{day} prediction shape/finite mismatch")
                if bound.get("prediction_sha256") != digest(array) or bound.get("query_indices_sha256") != item.get("query_indices_sha256") or bound.get("velocity_sha256") != item.get("velocity_sha256"):
                    raise ValueError(f"{cell}.{day} prediction/metric binding mismatch")
    return r,roster,values

def _write(p:Path,h:list[str],rows:list[list[Any]]):
    with p.open("w",newline="",encoding="utf-8") as f: csv.writer(f).writerows([h,*rows])

def build_report(base_final_receipt:Path,flat_final_receipt:Path,dest:Path)->dict[str,Path]:
    base,_,roster,bv=_validate_base(Path(base_final_receipt))
    flat,flat_roster,fv=_values(Path(flat_final_receipt),flat=True)
    if tuple(flat_roster) != tuple(roster): raise ValueError("base/Flat final roster mismatch")
    flat_seal=_read_json(Path(flat["seal_path"]))
    if Path(flat_seal.get("base_final_receipt", "")).resolve() != Path(base_final_receipt).resolve() or set(fv) != set(flat_seal.get("required_cells", [])):
        raise ValueError("Flat receipt does not exactly match sealed base/Flat roster")
    if base.get("final_sessions_opened")!=6 or flat.get("final_sessions_opened")!=6: raise ValueError("incomplete final access")
    for cell in fv:
        arm,rep,seed=_spec(cell); learned=f"{arm}_{rep}" + ("" if seed==42 else f"_s{seed}")
        if learned not in bv: raise ValueError(f"missing matched learned cell {learned}")
        for day in roster:
            left=base["results"][learned]["predictions"]["sessions"][day]; right=flat["results"][cell]["predictions"]["sessions"][day]
            if (left.get("query_indices_sha256"),left.get("velocity_sha256")) != (right.get("query_indices_sha256"),right.get("velocity_sha256")):
                raise ValueError(f"learned/Flat truth-query digest mismatch: {cell}.{day}")
    dest=Path(dest).resolve()
    if dest.exists() and any(dest.iterdir()): raise FileExistsError("report destination must be fresh")
    dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    neural={"full_sua","full_pmua","activity_sua","activity_pmua","raw_set_sua","raw_set_pmua","full_sua_s43","full_pmua_s43","full_sua_s44","full_pmua_s44"}
    for variant,values in (("base19",bv),("flat",fv)):
        for c,x in values.items():
            label="learned_slope_neural" if variant=="base19" and c in neural else ("existing_control" if variant=="base19" else "flat_neural")
            rows.append([label,c,f"{mean(x.values()):.12g}",*(f"{x[d]:.12g}" for d in roster)])
    combined=dest/"combined_results.csv"; _write(combined,["variant","cell","mean_r2",*roster],rows)
    diffs=[]
    for fc,x in fv.items():
        arm,rep,seed=_spec(fc)
        learned=f"{arm}_{rep}" + ("" if seed==42 else f"_s{seed}")
        if learned not in bv: raise ValueError(f"missing matched learned cell {learned}")
        for d in roster: diffs.append([arm,rep,seed,d,f"{bv[learned][d]-x[d]:.12g}"])
        diffs.append([arm,rep,seed,"equal_session_mean",f"{mean(bv[learned].values())-mean(x.values()):.12g}"])
    delta=dest/"learned_minus_flat_paired_differences.csv"; _write(delta,["arm","representation","seed","session","learned_minus_flat_r2"],diffs)
    summary={"roster":list(roster),"full":{}}
    for rep in ("sua","pmua"):
        for variant,values,prefix in (("learned",bv,"full_"),("flat",fv,"full_flat_")):
            cells=[f"{prefix}{rep}",f"{prefix}{rep}_s43",f"{prefix}{rep}_s44"]
            present=[c for c in cells if c in values]
            scores=[mean(values[c].values()) for c in present]
            summary["full"].setdefault(rep,{})[variant]={"available_seeds":[42 if c==cells[0] else int(c[-2:]) for c in present],"seed_mean_r2":scores,"three_seed_mean_r2":mean(scores) if len(scores)==3 else None,"three_seed_sample_sd_r2":stdev(scores) if len(scores)==3 else None,"note":None if len(scores)==3 else "Flat plan contains seed42 only; no three-seed summary was fabricated."}
    js=dest/"full_variant_seed_summary.json"; js.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    summary_rows=[]
    for rep in ("sua","pmua"):
        for variant in ("learned","flat"):
            item=summary["full"][rep][variant]
            value="NA (seed42 only)" if item["three_seed_mean_r2"] is None else f"{item['three_seed_mean_r2']:.12g} ± {item['three_seed_sample_sd_r2']:.12g}"
            summary_rows.append([rep,variant,",".join(map(str,item["available_seeds"])),value])
    paired=[]
    for rep in ("sua","pmua"):
        items=[]
        for seed in (42,43,44):
            fc=f"full_flat_{rep}"+("" if seed==42 else f"_s{seed}"); lc=f"full_{rep}"+("" if seed==42 else f"_s{seed}")
            if fc in fv: items.append(mean(bv[lc].values())-mean(fv[fc].values()))
        paired.append([rep, "NA (seed42 only)" if len(items)!=3 else f"{mean(items):.12g} ± {stdev(items):.12g}"])
    table="\n".join("| "+" | ".join(map(str,row))+" |" for row in rows)
    seed_table="\n".join("| "+" | ".join(row)+" |" for row in summary_rows); pair_table="\n".join("| "+" | ".join(row)+" |" for row in paired)
    md=dest/"report.md"; md.write_text("# DANDI688 learned and Flat follow-up\n\nBase19 is read from its completed receipt; Flat is a separate follow-up gate after base results were observed.\n\n## All base19 and Flat cells\n\n| category | cell | mean R² | "+" | ".join(roster)+" |\n|---|---|---:|"+"|".join("---:" for _ in roster)+"|\n"+table+"\n\n## Full seed summaries\n\n| representation | variant | seeds | three-seed mean ± sample SD |\n|---|---|---|---:|\n"+seed_table+"\n\n## Matched Full learned−Flat summary\n\n| representation | three-seed paired mean ± sample SD |\n|---|---:|\n"+pair_table+"\n\n[CSV](combined_results.csv) · [matched learned−Flat differences](learned_minus_flat_paired_differences.csv) · [Full seed summary](full_variant_seed_summary.json). Existing CPU/static rows are reference controls, not learned-slope neural rows.\n")
    return {"report":md,"combined_results":combined,"paired_differences":delta,"full_seed_summary":js}
