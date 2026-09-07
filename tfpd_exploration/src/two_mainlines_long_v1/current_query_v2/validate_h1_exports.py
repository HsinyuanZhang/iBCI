"""One-bin H1 replay of both frozen surfaces for a selected/endpoint EMA model."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch

from .streaming import CurrentQueryStream, ExactFullWindowStream
from .validate_m1_exports import _r2, _sha, _verify_file


def _mask_sha(mask):
    a = mask.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def run(manifest_path, arm, pick, output, device="cpu", exact_backend="baseline", *,
        query_backend="baseline", readout_receipt=None):
    from tfpd_exploration.src.h1_optimized_v2.cache import CACHE, ROOT, validate_authority
    from tfpd_exploration.src.decoder_validation_v2.audit_predictions import audit_h1
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

    manifest_path, output = Path(manifest_path), Path(output)
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text())
    flat_selection = False
    export_arm = arm
    if manifest['schema'] == 'h1_v7_dropout30_independent_native_score_export_v1':
        from tfpd_exploration.src.h1_optimized_v7.model import make_matched_pair
        revision = 'v7_dropout30_same_v6_architecture'
        if set(manifest['arms']) != {'full_v4', 't_v6'}:
            raise ValueError('V7 paired export roster drift')
        export_arm = {'full': 'full_v4', 't': 't_v6'}[arm]
    elif manifest["schema"] == "h1_v6_independent_native_score_export_v1":
        from tfpd_exploration.src.h1_optimized_v6.model import make_matched_pair
        revision = "v6_recency"
        flat_selection = True
        if arm != "t":
            raise ValueError("V6 query-only export has no newly trained FULL arm")
    elif manifest["schema"] == "h1_v5_independent_native_score_export_v1":
        from tfpd_exploration.src.h1_optimized_v5.model import make_matched_pair
        revision = "v5_logage"
        flat_selection = True
        if arm != "t":
            raise ValueError("V5 query-only export has no newly trained FULL arm")
    elif manifest["schema"] == "h1_v4_continuation_independent_native_score_export_v1":
        from tfpd_exploration.src.h1_optimized_v4.model import make_matched_pair
        revision = "v4_signed_continue24_deterministic"
        flat_selection = True
        if arm != "full":
            raise ValueError("V4 continuation export has no newly trained query arm")
    elif manifest["schema"] == "h1_v4_independent_native_score_export_v2":
        from tfpd_exploration.src.h1_optimized_v4.model import make_matched_pair
        revision = "v4_signed"
    else:
        raise ValueError("audited V4/V5/V6/V7 formal export manifest required")
    attempt = manifest_path.parent.parent
    for name, digest in manifest["committed_input_bindings"].items():
        _verify_file((ROOT if name == "source_cache_authority.json" else attempt) / name, digest)
    frozen = json.loads((attempt / "selection_freeze.json").read_text())
    record = manifest["arms"][export_arm][pick]
    if revision == 'v7_dropout30_same_v6_architecture':
        contract = ('v4_causal_full_window_no_query_temporal_contract' if arm == 'full'
                    else 'v6_recency_query_temporal_contract4')
        if (record.get('operator_contract') != contract or record.get('frontend_contract_version') != 4 or
                (arm == 'full' and 'temporal_contract_version' in record) or
                (arm == 't' and record.get('temporal_contract_version') != 4)):
            raise ValueError('V7 actual arm operator metadata mismatch')
    ckpt = _verify_file(record["checkpoint"], record["checkpoint_sha256"])
    selected = frozen["selected"] if flat_selection else frozen["selected"][export_arm]
    if pick == "selected" and ckpt.resolve() != Path(selected["checkpoint"]).resolve():
        raise ValueError("selected artifact differs from frozen selection")
    state_path = _verify_file(record["plain_ema_model_state"], record["plain_ema_model_state_sha256"])
    references, audits = {}, {}
    readout, calibration = None, None
    if readout_receipt is not None:
        if torch.device(device).type != "cpu":
            raise ValueError("verified native M3 wrapper currently requires CPU-bound model/banks")
        from tfpd_exploration.src.h1_runtime_v3 import load_frozen_readout
        calibration = json.loads(Path(readout_receipt).read_text())
        # Selected and endpoint labels can serialize the identical EMA tensor
        # state to different bytes. Only consider a manifest-bound alias from
        # the SAME checkpoint; the M3 decorator additionally compares every
        # live model tensor to this exact bound state before the first call.
        if calibration['schema'] == 'v7_canonical_m3_mat7_readout_v1' and calibration.get('arm') != export_arm:
            raise ValueError('V7 readout must bind the exact exported arm')
        plain_key = ("plain_ema_sha256" if calibration["schema"] in
                     {"v6_canonical_m3_mat7_readout_v1", "v7_canonical_m3_mat7_readout_v1"}
                     else "frozen_plain_ema_state_sha256")
        bound_state = state_path
        if _sha(bound_state) != calibration.get(plain_key):
            aliases = [candidate for candidate in manifest["arms"][export_arm].values()
                       if candidate["checkpoint_sha256"] == record["checkpoint_sha256"] and
                       candidate["plain_ema_model_state_sha256"] == calibration.get(plain_key)]
            if len(aliases) != 1:
                raise ValueError("M3 plain state is not a uniquely bound serialization of this checkpoint")
            bound_state = _verify_file(aliases[0]["plain_ema_model_state"], aliases[0]["plain_ema_model_state_sha256"])
        readout = load_frozen_readout(readout_receipt, checkpoint_path=ckpt, plain_ema_state_path=bound_state)
    for surface in ("selection", "complete"):
        row = record["surfaces"][surface]
        path = _verify_file(row["npz"], row["sha256"])
        if calibration is not None:
            corrected = calibration["applied_exports"][f"{pick}_{surface}"]
            if Path(corrected["source_npz"]).resolve() != path.resolve() or corrected["source_sha256"] != row["sha256"]:
                raise ValueError("M3 transformed reference does not bind this exact operator export")
            path = _verify_file(corrected["npz"], corrected["sha256"])
        audits[surface] = audit_h1(path, surface=surface)
        with np.load(path, allow_pickle=False) as z:
            references[surface] = {key: z[key].copy() for key in z.files}
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    validate_authority(cache, manifest["input_authority"])
    model = make_matched_pair()[0 if arm == "full" else 1]
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False), strict=True)
    if int(model.frontend_contract_version) != 4:
        raise ValueError("wrong frontend contract in state")
    if revision == "v5_logage" and int(model.temporal.temporal_contract_version) != 3:
        raise ValueError("wrong log-age temporal contract in state")
    if (revision == "v6_recency" or (revision == 'v7_dropout30_same_v6_architecture' and arm == 't')) and int(model.temporal.temporal_contract_version) != 4:
        raise ValueError("wrong recency-prior temporal contract in state")
    model.to(device).eval()
    if exact_backend != "baseline" and arm != "full":
        raise ValueError("exact backend options apply only to a FULL operator")
    if query_backend != "baseline" and arm != "t":
        raise ValueError("query backend options apply only to a query operator")
    if exact_backend == "static_qonly":
        from tfpd_exploration.src.h1_exact_cpu_v1.static_frontend_qonly import StaticCarrierQOnlyExactFullWindowStream
        engine_cls = StaticCarrierQOnlyExactFullWindowStream
    elif exact_backend == "baseline":
        if query_backend == "static_signed":
            from tfpd_exploration.src.h1_runtime_v3 import StaticSignedQueryStream
            engine_cls = StaticSignedQueryStream
        elif query_backend == "baseline":
            engine_cls = ExactFullWindowStream if arm == "full" else CurrentQueryStream
        else:
            raise ValueError("unknown query backend")
    else:
        raise ValueError("unknown exact backend")
    ref = references["complete"]
    offline, target = ref["prediction_native_velocity"], ref["target_native_velocity"]
    prediction = np.empty_like(offline)
    per_session = {}
    with torch.no_grad():
        for name, row in cache["minival"].items():
            if _mask_sha(row["bank"]["unit_mask"]) != manifest["per_session_bank_unit_mask_sha256"]["minival"][name]:
                raise ValueError("bank unit-mask hash drift")
            ids = np.flatnonzero(ref["session_id"] == name)
            ends = ref["bin_timestep"][ids].astype(np.int64)
            if not np.array_equal(ends, np.flatnonzero(row["eval_mask"])):
                raise ValueError("complete endpoint order drift")
            bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
            stream = engine_cls(model, bank, task="h1", session_id=name, unit_ids=range(176))
            if readout is not None:
                from tfpd_exploration.src.h1_runtime_v3 import CompactM3NativeReadoutStream
                stream = CompactM3NativeReadoutStream(stream, readout, session_id=name)
            count = 0
            for step in range(int(ends[-1]) + 1):
                observation = np.asarray(row["neural"][step:step + 1], dtype=np.float32)
                if count < len(ids) and step == ends[count]:
                    value = stream.predict(observation)[0]
                    index = ids[count]
                    bound = 1e-5 + 1e-5 * np.abs(offline[index])
                    if not np.all(np.abs(value - offline[index]) <= bound):
                        raise AssertionError(f"H1 {arm}/{pick} parity {name}:{step}: {np.max(np.abs(value-offline[index]))}")
                    prediction[index] = value
                    count += 1
                else:
                    stream.observe(observation)
            if count != len(ids):
                raise AssertionError("not all complete bins scored")
            per_session[name] = {"n": count, "observed_bins": int(ends[-1]) + 1,
                                 "r2_offline": _r2(offline[ids], target[ids]), "r2_stream": _r2(prediction[ids], target[ids]),
                                 "max_abs_error": float(np.max(np.abs(prediction[ids] - offline[ids])))}
            print(json.dumps({"session": name, **per_session[name]}), flush=True)
    keys = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(ref["session_id"], ref["bin_timestep"], strict=True))}
    surface_results = {}
    for surface, original in references.items():
        indices = np.asarray([keys[(str(s), int(t))] for s, t in zip(original["session_id"], original["bin_timestep"], strict=True)])
        p, y, original_p = prediction[indices], original["target_native_velocity"], original["prediction_native_velocity"]
        if not np.all(np.abs(p - original_p) <= 1e-5 + 1e-5 * np.abs(original_p)):
            raise AssertionError(f"{surface} parity failed")
        delta = _r2(p, y) - _r2(original_p, y)
        session_delta = {name: _r2(p[original["session_id"] == name], y[original["session_id"] == name]) -
                         _r2(original_p[original["session_id"] == name], y[original["session_id"] == name]) for name in per_session}
        if abs(delta) > 1e-5 or max(map(abs, session_delta.values())) > 1e-5:
            raise AssertionError("H1 frozen pooled/per-session R2 tolerance exceeded")
        surface_results[surface] = {"n": len(y), "r2_stream": _r2(p, y), "r2_offline": _r2(original_p, y),
                                    "r2_delta": delta, "per_session_r2_delta": session_delta,
                                    "equal_session_r2_delta": float(np.mean(list(session_delta.values())))}
    export = dict(ref); export["prediction_native_velocity"] = prediction
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), **export)
    code_paths = {Path(__file__), Path(__file__).with_name("streaming.py"), Path(__file__).with_name("core.py")}
    code_paths.update(Path(inspect.getfile(type(module))) for module in (model, model.frontend, model.temporal))
    # Include inherited backend sources as well as its final override.
    code_paths.update(Path(inspect.getfile(cls)) for cls in engine_cls.__mro__ if cls.__module__.startswith("tfpd_exploration."))
    if readout is not None:
        code_paths.update(Path(inspect.getfile(cls)) for cls in type(stream).__mro__ if cls.__module__.startswith("tfpd_exploration."))
    result = {"schema": "h1_frozen_ema_complete_api_replay_v2", "operator_revision": revision, "status": "PASS", "arm": arm, "pick": pick,
              "operator": "E_exact_full" if arm == "full" else "T_cached_query", "exact_backend": exact_backend,
              "query_backend": query_backend, "surfaces": surface_results,
              "per_session_complete": per_session, "independent_reference_audits": audits,
              "manifest": str(manifest_path), "manifest_sha256": _sha(manifest_path), "reference": record,
              "stream_export": str(output.with_suffix(".npz")), "stream_export_sha256": _sha(output.with_suffix(".npz")),
              "device": device, "torch_threads": torch.get_num_threads(), "affinity": sorted(os.sched_getaffinity(0)),
              "all_unscored_bins_consumed": True, "trial_boundary_resets": False, "not_a_latency_benchmark": True,
              "code_sha256": {str(p): _sha(p) for p in sorted(code_paths)}}
    if readout is not None:
        result["fixed_m3_readout"] = {"receipt": str(readout_receipt), "receipt_sha256": _sha(readout_receipt),
            "maps_sha256": calibration["maps_sha256"], "schema": calibration["schema"],
            "bound_plain_state": str(readout.plain_ema_state_path),
            "bound_plain_state_sha256": readout.plain_ema_state_sha256,
            "live_model_tensor_equality_to_bound_state_checked_before_each_session": True,
            "applied_reference_exports": {surface: calibration["applied_exports"][f"{pick}_{surface}"]
                                          for surface in ("selection", "complete")},
            "last_session_binding_memory": stream.memory_breakdown}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("full", "t"), required=True)
    parser.add_argument("--pick", choices=("selected", "epoch12", "epoch24"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exact-backend", choices=("baseline", "static_qonly"), default="baseline")
    parser.add_argument("--query-backend", choices=("baseline", "static_signed"), default="baseline")
    parser.add_argument("--readout-receipt", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    result = run(args.manifest, args.arm, args.pick, args.output, args.device, args.exact_backend,
                 query_backend=args.query_backend, readout_receipt=args.readout_receipt)
    print(json.dumps({"status": result["status"], "surfaces": result["surfaces"], "output": str(args.output)}), flush=True)
