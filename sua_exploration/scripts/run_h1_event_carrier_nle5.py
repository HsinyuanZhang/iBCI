#!/usr/bin/env python3
"""Publish the predeclared, CPU-only H1 NLE5 sparse-event screen once."""
from __future__ import annotations

# Set before importing numpy through the experiment modules.
import os
for _thread_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sua_exploration.mc_maze import h1_event_carrier_nle5 as nle5  # noqa:E402
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1  # noqa:E402
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as hse5  # noqa:E402

DEFAULT_DATA = ROOT / "SPINT-main/data/000954"
DEFAULT_HSE5_RECEIPT = ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json"
DEFAULT_DIR = ROOT / "sua_exploration/results/h1_event_carrier_nle5"


def _need(ok: bool, message: str) -> None:
    if not ok: raise ValueError(message)


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4<<20), b""): h.update(chunk)
    return h.hexdigest()


def _write_once(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    path=path.resolve(); side=path.with_suffix(path.suffix+".sha256")
    _need(not path.exists() and not side.exists(), f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload=(json.dumps(body, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    fd,tmpname=tempfile.mkstemp(prefix="."+path.name+".", suffix=".tmp", dir=path.parent); tmp=Path(tmpname)
    try:
        with os.fdopen(fd,"wb") as f: f.write(payload); f.flush(); os.fsync(f.fileno())
        os.chmod(tmp,0o444); os.link(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()
    digest=_sha(path); fd=os.open(side,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(fd,"w",encoding="ascii") as f: f.write(f"{digest}  {path.name}\n"); f.flush(); os.fsync(f.fileno())
    _need(stat.S_IMODE(path.stat().st_mode)==stat.S_IMODE(side.stat().st_mode)==0o444, "immutable mode drift")
    return path,digest


def _predeclaration() -> dict[str, Any]:
    return {"schema":"h1_event_carrier_nle5_predeclaration_v1", "protocol":nle5.PROTOCOL,
            "written_before_nwb_open":True, "candidate":"single fixed source-only nonlinear label embedding",
            "frozen_model":{"input":["tag","start_time","endpoint_delta","duration"],"rff_width":nle5.RFF_WIDTH,"rff_seed":nle5.RFF_SEED,
            "source_ridge_lambda":nle5.SOURCE_RIDGE_LAMBDA,"source_output_svd_rank":4,"target_ridge_lambda":nle5.TARGET_RIDGE_LAMBDA,"carrier_dim":5},
            "selection":"no candidate or hyperparameter grid; no result-dependent expansion", "budgets":[3,4],
            "outer_protocol":"source-date LODO: outer date excluded from source map; target support only refit; later target labels score only",
            "required_controls":["endpoint_label_shuffle","tag_shuffle","orthogonal_coordinate_rotation_invariance","row_attachment_shuffle","intercept_only"],
            "material_gate":{"each_budget":{"mean_nle5_minus_hse5_at_least":.02,"median_at_least":.01,"positive_sessions_at_least":"10/13","leave_largest_absolute_delta_out_mean_positive":True,"correct_minus_label_tag_intercept_positive":True}},
            "prohibitions":["minival","held-out","formal","EvalAI","dense_per_bin_velocity","GPU","target_backward"],
            "thread_limits":{k:"1" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")}}


def _reproduce(body: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    _need(reference.get("schema")==hse5.SCHEMA, "HSE5 reference schema mismatch")
    diffs=[]; fields={"median_r2_correct":"median_r2_correct","median_delta_label_shuffle":"median_delta_shuffle","median_delta_intercept":"median_delta_intercept"}
    for budget in nle5.SUPPORT_BUDGETS:
        for name in v1.H1_HELDIN_SESSIONS:
            for got,want in fields.items(): diffs.append(abs(float(body["budgets"][f"M{budget}"]["hse5_baseline_sessions"][name][got])-float(reference["budgets"][f"M{budget}"]["sessions"][name]["forward"][want])))
    _need(max(diffs)<=1e-10, f"H-SE5 exact reproduction failed: {max(diffs)}")
    return {"passed":True,"comparisons":len(diffs),"absolute_tolerance":1e-10,"maximum_absolute_difference":max(diffs)}


def _baseline_only(sessions: Mapping[str, Any]) -> dict[str, Any]:
    """Run the mandatory exact H-SE5 reproduction before NLE5 source fitting."""
    bases={d:hse5.fit_source_all_event_basis(sessions,outer_date=d) for d in v1.H1_DATES}
    return {f"M{budget}": {name:nle5.hse5_baseline_session(sessions[name],bases[v1.session_date(name)],budget=budget)
                             for name in v1.H1_HELDIN_SESSIONS} for budget in nle5.SUPPORT_BUDGETS}


def run(data_root: Path, hse5_receipt: Path, predeclared_sha: str) -> dict[str, Any]:
    started=time.monotonic(); reference_path=hse5_receipt.resolve(); reference=json.loads(reference_path.read_text())
    paths=v1.index_heldin_calib(data_root.resolve())
    sessions={n:v1.load_event_session(paths[n]) for n in v1.H1_HELDIN_SESSIONS}
    baseline=_baseline_only(sessions)
    baseline_body={"budgets":{k:{"hse5_baseline_sessions":v} for k,v in baseline.items()}}
    reproduction={**_reproduce(baseline_body,reference),"reference_path":str(reference_path),"reference_sha256":_sha(reference_path)}
    try:
        body=nle5.run_screen(sessions)
    except v1.SparseEventEndpointError as error:
        # An absent tag in a source LODO fold makes its supervised semantics unidentifiable.
        coverage={date:{tag:int(sum(e.tag==tag for name in nle5.source_names_for_outer(date) for e in sessions[name].events))
                        for tag in v1.MOVEMENT_TAGS} for date in v1.H1_DATES}
        if "source tag support deficient" not in str(error): raise
        body={"schema":nle5.SCHEMA,"protocol":nle5.PROTOCOL,"candidate_matrix_predeclared_before_data_run":True,
              "status":"STOP_CPU_NLE5_UNIDENTIFIED_SOURCE_TAG_SUPPORT","gpu_authorized_by_this_screen":False,
              "failure":{"reason":str(error),"source_tag_counts_by_outer_date":coverage,
                         "fail_closed":"No source-only learned semantic coefficient exists for an absent tag; no imputation or outer data was used."},
              "frozen_constants":{"carrier_dim":5,"embedding_dim":4,"support_budgets":[3,4],"rff_width":16,"rff_seed":190271,"single_model_no_outer_expanded_grid":True},
              "scope":{"public_held_in_calibration_nwbs_opened":13,"minival_nwbs_opened":0,"held_out_nwbs_opened":0,"formal_test_labels_opened":0,"dense_velocity_opened":False,"within_event_position_trajectory_opened":False,"target_session_optimizer_steps":0,"target_session_backward_steps":0,"decoder_constructed":False,"trainer_constructed":False,"cuda_used":False},
              "budgets":{},"interpretation":{"source_only_nonlinear_map":True,"outer_date_excluded_from_map_training":True,"stopped_before_target_refit":True}}
    body.update({"runtime_seconds":time.monotonic()-started,"predeclaration_sha256":predeclared_sha,
        "baseline_reproduction":reproduction,
        "source_binding":{"sessions":list(v1.H1_HELDIN_SESSIONS),"dates":list(v1.H1_DATES),"files":[{"session":n,"path":str(paths[n]),"sha256":sessions[n].input_sha256} for n in v1.H1_HELDIN_SESSIONS]},
        "implementation_binding":{"module_path":str(Path(nle5.__file__).resolve()),"module_sha256":_sha(Path(nle5.__file__).resolve()),"runner_path":str(Path(__file__).resolve()),"runner_sha256":_sha(Path(__file__).resolve()),"event_parser_path":str(Path(v1.__file__).resolve()),"event_parser_sha256":_sha(Path(v1.__file__).resolve()),"hse5_path":str(Path(hse5.__file__).resolve()),"hse5_sha256":_sha(Path(hse5.__file__).resolve())},
        "receipt_integrity":{"publication_mode":"write-once-immutable-0444","nice_requested":True,"thread_limits":{k:os.environ[k] for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")}}})
    return body


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-root",type=Path,default=DEFAULT_DATA); p.add_argument("--hse5-receipt",type=Path,default=DEFAULT_HSE5_RECEIPT); p.add_argument("--output-dir",type=Path,default=DEFAULT_DIR); a=p.parse_args()
    try: os.nice(19)
    except OSError: pass
    out=a.output_dir.resolve(); pre_path=out/"predeclaration_v1.json"
    if pre_path.exists():
        pre=pre_path.resolve(); side=pre.with_suffix(pre.suffix+".sha256"); sha=_sha(pre)
        _need(side.is_file() and stat.S_IMODE(pre.stat().st_mode)==stat.S_IMODE(side.stat().st_mode)==0o444 and side.read_text().strip()==f"{sha}  {pre.name}","existing predeclaration integrity")
    else: pre,sha=_write_once(pre_path,_predeclaration())
    body=run(a.data_root,a.hse5_receipt,sha); result,digest=_write_once(out/"source_screen_v1.json",body)
    report={"status":body["status"],"predeclaration":str(pre),"predeclaration_sha256":sha,"receipt":str(result),"receipt_sha256":digest}
    if body["budgets"]: report.update({"M3":body["budgets"]["M3"]["aggregate"],"M4":body["budgets"]["M4"]["aggregate"]})
    else: report["failure"]=body["failure"]
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if body["status"]=="PASS_CPU_NLE5_MATERIAL" else 2

if __name__=="__main__": raise SystemExit(main())
