"""Parameterized independent M3 support/refit/application audit.

The independently implemented NumPy OLS law is reused from the prior auditor,
not from either producer. Neural predictions used as fit inputs P require a
separate model/runtime replay; this audit does not claim to recompute them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit_h1_m3 import AUTH, apply, fit, sha
from .audit_predictions import audit_h1
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import _index_split, load_session_arrays


def run(receipt_path, expected_sha256, target):
    receipt_path, target = Path(receipt_path), Path(target)
    if target.exists():
        raise FileExistsError(target)
    if sha(receipt_path) != expected_sha256:
        raise ValueError("explicit frozen receipt digest mismatch")
    receipt = json.loads(receipt_path.read_text())
    allowed = {"v4full24_canonical_m3_mat7_readout_v1": "v4full24_canonical_m3_mat7_maps_v1",
               "v6_canonical_m3_mat7_readout_v1": "v6_canonical_m3_mat7_maps_v1",
               "v7_canonical_m3_mat7_readout_v1": "v7_canonical_m3_mat7_maps_v1"}
    if receipt.get("schema") not in allowed or receipt.get("status") != "FITTED_FIXED_CONTRACT":
        raise ValueError("unsupported fixed model-specific M3 receipt")
    v6 = receipt["schema"] == "v6_canonical_m3_mat7_readout_v1"
    if ((receipt.get("family"), receipt.get("ridge"), receipt.get("scale_floor")) != ("MAT7", 0.0, 1e-6) or
            (not v6 and receipt.get("fit_dtype") != "float64")):
        raise ValueError("fixed MAT7 numerical law drift")
    if sha(AUTH) != receipt["canonical_authority_sha256"]:
        raise ValueError("canonical support authority digest mismatch")
    canonical = json.loads(AUTH.read_text())
    official = {row["session"]: row for row in canonical["sessions"] if row["scope"] == "held-in-calib"}
    if len(official) != 13 or set(official) != set(receipt["sessions"]):
        raise ValueError("canonical source session roster mismatch")
    maps_path = Path(receipt["maps"])
    if sha(maps_path) != receipt["maps_sha256"]:
        raise ValueError("maps file digest mismatch")
    payload = torch.load(maps_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != allowed[receipt["schema"]] or set(payload["maps"]) != set(official):
        raise ValueError("model-specific maps schema/roster mismatch")
    if receipt['schema'] == 'v7_canonical_m3_mat7_readout_v1':
        contracts = {'full_v4': 'v4_causal_full_window_no_query_temporal_contract',
                     't_v6': 'v6_recency_query_temporal_contract4'}
        arm = receipt.get('arm')
        if (arm not in contracts or payload.get('arm') != arm or
                receipt.get('operator_contract') != contracts[arm]):
            raise ValueError('V7 arm/operator/map binding mismatch')
    persisted, refitted, sessions = payload["maps"], {}, {}
    if any(np.asarray(mapping[key]).dtype != np.float64 for mapping in persisted.values() for key in
           ("p_mean", "p_scale", "y_mean", "y_scale", "weight", "intercept")):
        raise ValueError("stored fitted map arrays are not float64")
    paths, total = _index_split("held-in-calib"), 0
    for session, c in sorted(official.items()):
        row = receipt["sessions"][session]
        support = Path(row["support_npz"])
        if sha(support) != row["support_npz_sha256"]:
            raise ValueError("support archive digest mismatch")
        with np.load(support, allow_pickle=False) as z:
            if set(z.files) != {"endpoint", "trial_id", "prediction_native_velocity", "target_native_velocity"}:
                raise ValueError("support archive schema mismatch")
            ends, trial, p, y = [np.asarray(z[key]) for key in
                                 ("endpoint", "trial_id", "prediction_native_velocity", "target_native_velocity")]
        rec = load_session_arrays(paths[session], session, skip_first3=True)
        raw_sha = sha(paths[session])
        if raw_sha != c["nwb_sha256"] or raw_sha != row["raw_nwb_sha256"]:
            raise ValueError("actual raw source NWB digest mismatch")
        valid = np.asarray(rec.eval_mask, dtype=bool)
        first = tuple(np.unique(np.asarray(rec.trial_num)[valid])[:3].astype(float))
        wanted = np.flatnonzero(valid & np.isin(rec.trial_num, np.asarray(first)))
        if (first != tuple(c["calibration_trials"]) or len(ends) != c["calibration_bins"] or
                not np.array_equal(ends, wanted) or not np.array_equal(trial, np.asarray(rec.trial_num)[wanted]) or
                not np.array_equal(y, np.asarray(rec.velocity)[wanted])):
            raise ValueError("canonical M3 endpoint/trial/native-target mismatch")
        mapping = fit(p, y)
        differences = {key: float(np.max(np.abs(mapping[key] - np.asarray(persisted[session][key]))))
                       for key in ("p_mean", "p_scale", "y_mean", "y_scale", "weight", "intercept")}
        if max(differences.values()) > 1e-10:
            raise ValueError("independent float64 OLS differs from saved map")
        refitted[session] = mapping
        sessions[session] = {"support_bins": len(ends), "first_three_valid_trialnums": first,
                             "support_sha256": sha(support), "raw_nwb_sha256": raw_sha,
                             "map_max_abs_differences": differences}
        total += len(ends)
    if total != 30879 or total != receipt["support_total_bins"]:
        raise ValueError("canonical M3 total support count mismatch")
    exports = {}
    for label, row in receipt["applied_exports"].items():
        source, corrected = Path(row["source_npz"]), Path(row["npz"])
        if sha(source) != row["source_sha256"] or sha(corrected) != row["sha256"]:
            raise ValueError("source/corrected prediction archive digest mismatch")
        with np.load(source, allow_pickle=False) as original, np.load(corrected, allow_pickle=False) as final:
            if set(original.files) != set(final.files) or any(
                    not np.array_equal(original[key], final[key]) for key in original.files if key != "prediction_native_velocity"):
                raise ValueError("non-prediction field changed during calibration")
            if set(np.unique(original["session_id"])) != set(refitted):
                raise ValueError("corrected archive session coverage mismatch")
            manual = np.empty_like(original["prediction_native_velocity"])
            for session, mapping in refitted.items():
                select = original["session_id"] == session
                manual[select] = apply(mapping, original["prediction_native_velocity"][select]).astype(manual.dtype)
            error = float(np.max(np.abs(manual - final["prediction_native_velocity"])))
            if error > 1e-10:
                raise ValueError("independently refitted map application mismatch")
        surface = "selection" if label.endswith("selection") else "complete"
        exports[label] = {"source_sha256": sha(source), "corrected_sha256": sha(corrected),
                          "independent_map_max_abs_error": error, "score": audit_h1(corrected, surface=surface)}
    result = {"schema": "h1_m3_independent_support_refit_audit_v2", "status": "PASS",
              "receipt": str(receipt_path), "receipt_sha256": expected_sha256,
              "scope": "raw canonical support/independent float64 OLS/full corrected exports; not neural P reproduction",
              "support_bins": total, "sessions": sessions, "exports": exports,
              "code_sha256": {str(Path(__file__)): sha(__file__),
                              str(Path(__file__).with_name("audit_h1_m3.py")): sha(Path(__file__).with_name("audit_h1_m3.py"))}}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.receipt, args.expected_sha256, args.output)
    print(json.dumps({"status": result["status"], "support_bins": result["support_bins"], "output": str(args.output)}))
