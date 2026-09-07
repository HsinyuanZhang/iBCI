#!/usr/bin/env python3
"""Full-length batch-7 two-wave test-phase simulation for the TTA decoder.

Mirrors the official test phase shape: 13 sessions in two waves (7 held-in +
6 held-out), batch 7 with zero-padding to the wave max length, reset once per
wave, predict per bin.  Measures wall time and crashes; verifies per-session
R2 against the V4 score rows at the official query windows.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE, ROOT / "streaming_calibration_exp",
             ROOT / "tfpd_exploration/submissions/evalai_m2_apfg_static_v1"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import src.models.streaming_calibration_module  # noqa: F401
sys.path.append(str(ROOT / "SPINT-main"))

import numpy as np
import torch

from falcon_challenge.config import FalconConfig, FalconTask
from cenat_chunk_decoder import CenatChunkTTADecoder
from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data
from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import (
    select_common_post30_window_starts, variance_weighted_r2)

torch.set_num_threads(min(8, torch.get_num_threads()))
payload_path = HERE / "artifacts/t4_m2_seed42_cenat_chunk100e_tta.pkl"
task_config = FalconConfig(task=FalconTask.m2)
decoder = CenatChunkTTADecoder(task_config=task_config, model_path=str(payload_path), batch_size=7)
_model, data_module, _tc, _meta = load_frozen_model_and_data()

v4 = json.loads((ROOT / "tfpd_exploration/results/m2_ajpf_c_v4/score.json").read_text(encoding="utf-8"))
v4_rows = {(r["surface"], r["session"]): float(r["r2"]) for r in v4["rows"] if r["arm"] == "Ce-NAT"}

session_to_tag = {}
from sua_exploration.evalai_t4_m2.export_t4_payload import calibration_file_map
session_to_tag.update(calibration_file_map(ROOT / "SPINT-main/data/000953", task_config))

surfaces = (("external_official_query", "heldout", data_module.val_heldout_dataset),
            ("within_post30", "train", data_module.train_dataset))
report = {"waves": []}
overall_start = time.monotonic()
for surface_name, role, dataset in surfaces:
    sessions = sorted(dataset.calib_trialized_neural_features)
    waves = [sessions]  # one wave per surface (external=6, within=7; official test mixes 13 in 7+6)
    for w_idx, wave in enumerate(waves):
        tags = [Path(f"sub-MonkeyN-{role}_{s}_behavior+ecephys.nwb") for s in wave]
        streams, starts_by, targets_by = {}, {}, {}
        for s in wave:
            streams[s] = np.ascontiguousarray(np.asarray(dataset.neural_data[s], dtype=np.float32)[49:])
            all_starts = np.asarray([x for name, x in dataset.window_indices if name == s], dtype=np.int64)
            if surface_name == "within_post30":
                all_starts = select_common_post30_window_starts(all_starts, dataset.trial_start_indices[s])
            starts_by[s] = all_starts
            targets_by[s] = np.asarray(dataset.covariate_data[s], dtype=np.float32)
        max_len = max(int(streams[s].shape[0]) for s in wave)
        decoder.reset(dataset_tags=tags)
        started = time.monotonic()
        padded_pred = np.zeros((max_len, len(wave), 2), dtype=np.float32)
        for t in range(max_len):
            batch = np.zeros((len(wave), 96), dtype=np.float32)
            for slot, s in enumerate(wave):
                if t < streams[s].shape[0]:
                    batch[slot] = streams[s][t]
            padded_pred[t] = decoder.predict(batch)
        wall = time.monotonic() - started
        wave_report = {"surface": surface_name, "wave": w_idx, "sessions": list(wave),
                       "max_len_bins": max_len, "wall_seconds": round(wall, 1),
                       "per_session": {}}
        for slot, s in enumerate(wave):
            n = int(streams[s].shape[0])
            starts = starts_by[s]
            # Padded start s decodes the last-50 real bins ending at real bin s
            # (online_predictions[s]); the 49-bin zero pre-history lives in
            # reset, not in padded_pred.  Do NOT subtract 49 here.
            pred_at = padded_pred[starts, slot]
            targets = np.ascontiguousarray(targets_by[s][starts + 49], dtype=np.float32)
            r2 = float(variance_weighted_r2(targets, pred_at))
            v4_r2 = v4_rows[(surface_name, s)]
            wave_report["per_session"][s] = {
                "r2_batch7_full_stream": r2, "r2_v4": v4_r2,
                "abs_diff": abs(r2 - v4_r2), "windows": int(starts.size)}
            print(json.dumps({"session": s, "r2": r2, "v4": v4_r2,
                              "diff": abs(r2 - v4_r2)}), flush=True)
        report["waves"].append(wave_report)
        print(json.dumps({"wave_done": w_idx, "surface": surface_name, "wall": wall}), flush=True)

report["total_wall_seconds"] = time.monotonic() - overall_start
worst = max(w["per_session"][s]["abs_diff"] for w in report["waves"] for s in w["per_session"])
report["worst_abs_diff"] = worst
out = HERE / "artifacts/test_phase_simulation.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "DONE", "total_wall": report["total_wall_seconds"],
                  "worst_abs_diff": worst}, sort_keys=True))
