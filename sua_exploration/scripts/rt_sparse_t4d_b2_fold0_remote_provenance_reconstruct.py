#!/usr/bin/env python3
"""Create a read-only cryptographic reconstruction receipt for legacy RT B2 fold 0.

This never loads a B2 model.  It checks the remote preserved checkpoint,
selection/config/split/outer receipts, teacher metadata, and CPU-only query
windows against the local imported artifact bundle and Stage-2 T4d closure.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
COMPANION = ROOT / "sua_exploration/scripts/rt_sparse_t4d_vs_b2_d1024_companion.py"
HOST = "xinyuan@100.103.97.12"
B2_ROOT = "/home/xinyuan/Work_host/SPINT/rt_clean_nested_5070ti_stage/streaming_calibration_exp"
STAGE_ROOT = "/home/xinyuan/Work_host/SPINT/rt_sparse_endpoint_stage2_5070_v1_20260810/streaming_calibration_exp"
STAGE_RESULTS = "/home/xinyuan/Work_host/SPINT/rt_sparse_endpoint_stage2_5070_v1_20260810/results/matrix_v1"
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/rt_sparse_t4d_b2_companion_v1/RT_B2_FOLD0_REMOTE_PROVENANCE_RECONSTRUCTION_v1.json"


class ReconstructionError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ReconstructionError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(path: Path):
    spec = importlib.util.spec_from_file_location("fold0_companion", path)
    _need(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ssh_python(program: str, host: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", host,
         "/home/xinyuan/miniconda3/envs/spint/bin/python", "-"],
        input=program, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    _need(completed.returncode == 0, f"remote CPU query replay failed: {completed.stderr.strip()}")
    value = json.loads(completed.stdout)
    _need(isinstance(value, dict), "remote query replay root is not an object")
    return value


def _query_program(source_root: str, *, t4d_closure: str | None = None) -> str:
    """Exact current bytes at ``source_root``; only data and sampler are built."""

    closure_lines = []
    if t4d_closure:
        closure_lines = [
            f"closure=json.load(open({t4d_closure!r}))",
            "recorded=closure['outer_receipt']['matched_query_window_identity'][session]",
            "out['recorded_t4d_all_eligible_digests']={k:recorded[k] for k in DIGESTS}",
            "out['recorded_t4d_evaluated_windows']=closure['outer_receipt']['query_windows_evaluated']",
        ]
    lines = [
        "import hashlib,json,os,sys",
        "from pathlib import Path",
        "from collections import OrderedDict",
        "import numpy as np",
        f"sys.path.insert(0,{source_root!r})",
        "from src.data.rt_k4_loader import load_rt_session",
        "from src.data.falcon_datamodule import FalconDataset,SessionBatchSampler",
        "DIGESTS=('ordered_window_start_sha256','ordered_target_covariate_evalmask_sha256','ordered_query_identity_sha256')",
        "def sha(path):",
        " h=hashlib.sha256()",
        " with open(path,'rb') as f:",
        "  for b in iter(lambda:f.read(1<<20),b''):h.update(b)",
        " return h.hexdigest()",
        "def digest(values):",
        " h=hashlib.sha256()",
        " for value in values:",
        "  a=np.ascontiguousarray(value);h.update(str(a.dtype).encode());h.update(repr(a.shape).encode());h.update(a.tobytes())",
        " return h.hexdigest()",
        "session='ses-RT-20131009'",
        "target=Path('/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C/sub-C_ses-RT-20131009_behavior+ecephys.nwb')",
        "raw=load_rt_session(target)",
        "ds=FalconDataset(sessions_dict=OrderedDict([(session,raw)]),calib_sessions_dict=OrderedDict([(session,raw)]),split='rt_nested_outer_target_one_shot',window_size=50,calibration_n_trials=24,max_trial_length=100,use_calib_intertrials=True,remove_calib_still_times=False,interpolate_trials=True,interpolate_trials_kind='cubic',pad_value=-1.0,side_feature_group='zero4',side_feature_shuffle_seed=42,query_start_trial=24)",
        "audit=ds.query_window_audit[session]",
        "sampler=SessionBatchSampler(ds,32,shuffle=False)",
        "all_indices=list(range(len(ds)))",
        "indices=[i for batch in sampler for i in batch]",
        "all_starts=np.asarray([ds.window_indices[i][1] for i in all_indices],dtype=np.int64)",
        "starts=np.asarray([ds.window_indices[i][1] for i in indices],dtype=np.int64)",
        "all_target_indices=all_starts+ds.window_size-1",
        "target_indices=starts+ds.window_size-1",
        "all_rows=np.asarray(ds.covariate_data[session][all_target_indices],dtype=np.float32)",
        "rows=np.asarray(ds.covariate_data[session][target_indices],dtype=np.float32)",
        "all_mask=np.asarray(ds.eval_mask[session][all_target_indices],dtype=bool)",
        "mask=np.asarray(ds.eval_mask[session][target_indices],dtype=bool)",
        "source_files=['src/data/falcon_datamodule.py','src/data/rt_k4_loader.py','src/data/rt_nested_loso_datamodule.py','src/rt_clean_nested_loso_eval.py']",
        f"out={{'source_root':{source_root!r},'source_sha256':{{p:sha(os.path.join({source_root!r},p)) for p in source_files}},'target_nwb_sha256':sha(target),'all_eligible_windows':audit['eligible_windows'],'evaluated_windows':len(indices),'batch_size':32,'batch_count':len(sampler),'all_eligible_digests':{{'ordered_window_start_sha256':digest((all_starts,)),'ordered_target_covariate_evalmask_sha256':digest((all_target_indices,all_rows,all_mask)),'ordered_query_identity_sha256':digest((all_starts,all_target_indices,all_rows,all_mask))}},'evaluated_digests':{{'ordered_window_start_sha256':digest((starts,)),'ordered_target_covariate_evalmask_sha256':digest((target_indices,rows,mask)),'ordered_query_identity_sha256':digest((starts,target_indices,rows,mask))}}}}",
        *closure_lines,
        "print(json.dumps(out,sort_keys=True))",
    ]
    return "\n".join(lines) + "\n"


def _remote_hashes(paths: Mapping[str, str], host: str) -> dict[str, Any]:
    program = "\n".join([
        "import hashlib,json", f"paths={dict(paths)!r}",
        "def sha(p):", " h=hashlib.sha256()", " with open(p,'rb') as f:",
        "  for b in iter(lambda:f.read(1<<20),b''):h.update(b)", " return h.hexdigest()",
        "print(json.dumps({'sha256':{k:sha(v) for k,v in paths.items()},'teacher':json.load(open(paths['teacher_metadata']))},sort_keys=True))",
    ])
    return _ssh_python(program, host)


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    _need(not path.exists(), f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(prefix=".fold0-reconstruction-", dir=path.parent))
    temporary = temporary_directory / path.name
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
        if temporary_directory.exists():
            temporary_directory.rmdir()


def reconstruct(*, host: str = HOST, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    companion = _module(COMPANION)
    row = companion._legacy_b2_rows()[0]
    local = {name: companion._bound_legacy_file(row, name) for name in ("outer", "selection", "split", "config")}
    selection, outer = companion._json(local["selection"]), companion._json(local["outer"])
    fit = B2_ROOT + "/outputs/rt_stage_r_b2_remote/gpu_runs_zero4_v2/b2_d1024_zero4/fold_00/seed_42/fit"
    remote_paths = {
        "outer": fit + "/outer_target_eval.json", "selection": fit + "/rt_nested_selection_receipt.json",
        "split": fit + "/split_manifest.json", "config": fit + "/.hydra/config.yaml",
        "checkpoint": fit + "/checkpoints/best_ckpt/epoch_008.ckpt",
        "teacher_metadata": B2_ROOT + "/outputs/rt_stage_r_b2_remote/gpu_runs_zero4_v2/_artifacts/rt_clean_nested_loso_m24_b2_d1024_zero4_f0_s42_20260808_004655/teacher_metadata.json",
    }
    remote = _remote_hashes(remote_paths, host)
    for name, path in local.items():
        _need(remote["sha256"][name] == _sha(path), f"fold0 remote/local {name} SHA mismatch")
    _need(remote["sha256"]["checkpoint"] == selection.get("best_model_sha256"), "fold0 selected checkpoint SHA mismatch")
    _need(outer.get("checkpoint_sha256") == remote["sha256"]["checkpoint"], "fold0 outer/checkpoint chain mismatch")
    _need(remote["teacher"].get("teacher_checkpoint_sha256") == "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec", "fold0 teacher SHA mismatch")
    old = _ssh_python(_query_program(B2_ROOT), host)
    stage = _ssh_python(_query_program(STAGE_ROOT, t4d_closure=STAGE_RESULTS + "/cells/f00_rt_sparse_endpoint_t4d.json"), host)
    _need(old["evaluated_windows"] == outer.get("query_windows_evaluated"), "fold0 old replay count mismatch")
    _need(stage["evaluated_windows"] == stage.get("recorded_t4d_evaluated_windows"), "fold0 Stage2 replay count mismatch")
    _need(old["all_eligible_digests"] == stage["recorded_t4d_all_eligible_digests"], "fold0 all-eligible digest mismatch")
    _need(old["evaluated_digests"] == stage["evaluated_digests"], "fold0 actual evaluated digest mismatch")
    receipt = {
        "schema": "rt_sparse_t4d_b2_fold0_remote_provenance_reconstruction_v1",
        "status": "PASS_CRYPTOGRAPHIC_REMOTE_RECONSTRUCTION_LIMITED_HISTORICAL_SOURCE_ATTESTATION",
        "fold": 0, "seed": 42, "model_loaded": False, "gpu_opened": False,
        "nwb_opened_cpu_query_replay_only": True, "optimizer_constructed": False, "backward_called": False,
        "local_imported_artifacts": {name: {"path": str(path), "sha256": _sha(path)} for name, path in local.items()},
        "remote_artifact_sha256": remote["sha256"], "remote_teacher_metadata": remote["teacher"],
        "historical_b2_query_replay": old, "stage2_query_replay": stage,
        "query_identity_verdict": "PASS: all-eligible and actual ordered complete-batch digests match",
        "source_provenance_limit": "Imported fold0 had no contemporaneous full source-tree manifest. This receipt cryptographically binds currently preserved remote bytes and proves query replay; it does not claim a missing historical source manifest existed.",
    }
    _write_immutable(output, receipt)
    return {"status": receipt["status"], "receipt_path": str(output), "receipt_sha256": _sha(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(reconstruct(host=args.host, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ReconstructionError as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(2) from error
