"""Selected actual P1 versus as-shipped original M1, same-image public timing.

The full completed P1 proof is a prerequisite. This benchmark uses only raw
source neural and already proved frozen banks, never target values. B4 lanes
cover all three available source sessions with one explicitly repeated bank.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

from .complete_m1_family_source import artifact_audit, array_sha, atomic_json, require_same_files, sha, source_audit

IMAGE = "sha256:f5af9eb29b7f86616d898261070193b1b0777db75567848c62d7f888ce3d76cd"
WRAPPER = Path("/third_party/falcon_challenge/spint_decoder.py")
PAYLOAD = Path("/data/decoder.pkl")
EXPECTED = {
    WRAPPER: "3aacad1c22ea2b41ec776627de938c7973417361cf6d01da1e0daf15be0c7144",
    PAYLOAD: "052e9eab7be2af8bacd5298e348d414cce60dad767d6a0880b58dd39b27279f6",
    Path("/src/models/components/spint.py"): "855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519",
    Path("/decode.py"): "3d17b8097a2804f99380f922500a4a41c7f6c4e5a81a6504a010c6a55ce4260e",
}
NAMES = ("ORIGINALdefault", "FLAT", "ROUTE")
LANES = ("ses-20120926", "ses-20120927", "ses-20120928", "ses-20120926")
OFFSETS = (0, 0, 0, 4096)


def require_proof(path, expected_sha, audit):
    if len(expected_sha) != 64 or sha(path) != expected_sha:
        raise RuntimeError("full completed P1 proof SHA drift")
    proof = json.loads(path.read_text())
    if (proof.get("status") != "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY" or proof.get("outer_query_opened") is not False
            or proof.get("parameter_updates") != 0 or proof.get("batch") != 1
            or proof["pre_artifact"] != audit or proof["post_artifact"] != audit
            or proof["pre_source"] != proof["post_source"]):
        raise RuntimeError("selected P1 complete proof authority mismatch")
    for arm in ("flat", "route"):
        row = proof["arms"][arm]
        if (row["selected"] != audit["selected"][arm] or row["scored_count"] != 31252
                or row["public_calls"] != 112985 or row["initial_current_predictions"] != 3
                or not 0 <= row["max_abs_error"] <= 1e-5):
            raise RuntimeError("selected full chronological proof incomplete")
    require_same_files(proof["pre_source"]["files"])
    return proof


def load_source_lanes(source, count):
    import torch
    from tfpd_exploration.src.m1_runtime_v3.runtime import BankBatch
    raw_rows, banks, rosters = [], [], []
    with np.load(source["cache"], allow_pickle=False) as cache, np.load(source["provenance"], allow_pickle=False) as provenance:
        for name, offset in zip(LANES, OFFSETS, strict=True):
            raw = cache[f"raw_neural/{name}"]
            if (raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[1] != 64
                    or array_sha(raw) != source["cache_receipt"]["arrays"][name]["raw_neural_sha256"]
                    or not np.all(raw[:99] == 0)):
                raise RuntimeError("source raw startup/cache identity drift")
            value = np.array(raw[99+offset:99+offset+count], dtype=np.float32, copy=True)
            if value.shape != (count, 64) or not np.isfinite(value).all():
                raise RuntimeError("insufficient/nonfinite timing neural values")
            raw_rows.append(value)
            banks.append(tuple(cache[f"bank_{key}/{name}"].copy() for key in ("e0", "t", "unit_mask")))
            roster = provenance[f"nwb_unit_ids_in_rate_column_order/{name}"]
            if (roster.shape != (64,) or len(set(roster.tolist())) != 64
                    or array_sha(roster) != source["provenance_receipt"]["rows"][name]["nwb_unit_ids_sha256"]):
                raise RuntimeError("physical unit roster authority drift")
            rosters.append(tuple(roster.tolist()))
    e0, t, mask = [torch.from_numpy(np.stack([row[i] for row in banks])) for i in range(3)]
    bank = BankBatch(e0, t, mask, LANES, tuple(rosters))
    bank.validate(4, 64)
    if e0.shape != (4, 64, 100) or t.shape != (4, 64, 4):
        raise RuntimeError("actual P1 bank geometry drift")
    return bank, np.ascontiguousarray(np.stack(raw_rows, axis=1))


def assert_public(name, value):
    if (not isinstance(value, np.ndarray) or value.shape != (4, 16) or value.dtype != np.float32
            or not value.flags.owndata or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise RuntimeError(name + ": owning native FP32 [4,16] output required")


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or not (array > 0).all():
        raise RuntimeError("invalid timing samples")
    return {"mean_ms": float(array.mean()), "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)), "p99_ms": float(np.percentile(array, 99)), "max_ms": float(array.max())}


def original_oracle(engine):
    import torch
    raw = torch.from_numpy(engine.observation_buffer.copy().transpose(1, 0, 2))
    with torch.no_grad():
        values = [engine.local_clf(raw[i:i+1], calib_trialized_neural_features=calib.unsqueeze(0).to(raw))[:, -1]
                  for i, calib in enumerate(engine.local_calib_trial_features)]
    return (torch.cat(values).numpy() / engine.behavior_scaling_factor).astype(np.float32, copy=True)


def family_oracle(engine):
    import torch
    with torch.no_grad():
        return engine.model.forward_last(engine.raw, engine.bank).numpy().copy()


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if (os.environ.get("M1_FAMILY_SPINT_COMPARISON_GO") != "1" or not args.container_reference
            or args.image_digest != IMAGE or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")):
        raise RuntimeError("explicit same-image CPU-only comparison gate required")
    if args.calls not in (32, 2048) or args.warmup != 128 or args.threads not in (1, 2):
        raise RuntimeError("only a 32-call contract smoke or a 2048-call long run; warmup128 and T1/T2")
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError("frozen original image file SHA drift: " + str(path))
    # Bind image/helper code before importing model or source machinery, not
    # merely after those imports have already executed.
    immutable_paths = [*EXPECTED, Path(__file__), args.proof,
                       Path(__file__).with_name("complete_m1_family_source.py")]
    immutable_paths += sorted(Path("/src/models").rglob("*.py"))
    immutable = {str(path): sha(path) for path in immutable_paths}
    audit = artifact_audit(args.run_root)
    proof = require_proof(args.proof, args.proof_sha256, audit)
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("CPU only")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # Import image-owned SPINT before trainer helpers can extend sys.path.
    import_started = time.perf_counter_ns()
    import src.models.components.spint as original_module
    if Path(original_module.__file__).resolve() != Path("/src/models/components/spint.py"):
        raise RuntimeError("not the image-owned SPINT implementation")
    spec = importlib.util.spec_from_file_location("_original_m1_frozen_wrapper", WRAPPER)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    import_ms = (time.perf_counter_ns()-import_started)/1e6
    source = source_audit(audit)
    if source != proof["pre_source"]:
        raise RuntimeError("current source differs from completed proof")
    bank, values = load_source_lanes(source, 1+args.warmup+args.calls)
    from tfpd_exploration.src.m1_family_v1.family_train_v2 import _p1_models
    from .m1_lifted import LiftedFiveTokenCurrentQueryStream
    from .m1_route_lifted import RouteLiftedFiveTokenCurrentQueryStream
    from falcon_challenge.config import FalconConfig, FalconTask
    engines, constructor, reset = {}, {}, {}
    begun = time.perf_counter_ns()
    engine = wrapper.SpintDecoder(FalconConfig(task=FalconTask.m1), str(PAYLOAD), batch_size=4)
    constructor["ORIGINALdefault"] = (time.perf_counter_ns()-begun)/1e6
    begun = time.perf_counter_ns()
    engine.reset([Path("sub-MonkeyL-held-in-calib_"+name+"_behavior+ecephys") for name in LANES])
    reset["ORIGINALdefault"] = (time.perf_counter_ns()-begun)/1e6
    if (engine.device.type != "cpu" or engine.smooth_calibration or engine.window_size != 100
            or engine.behavior_scaling_factor != 1.0 or any(m.training for m in engine.local_clf.modules())
            or len(engine.local_calib_trial_features) != 4):
        raise RuntimeError("original payload public constructor/reset contract drift")
    engines["ORIGINALdefault"] = engine
    for arm, index, stream_type in (("FLAT", 0, LiftedFiveTokenCurrentQueryStream),
                                    ("ROUTE", 1, RouteLiftedFiveTokenCurrentQueryStream)):
        begun = time.perf_counter_ns()
        pair = _p1_models(torch.device("cpu"))
        model = pair[index]
        del pair
        state = torch.load(args.run_root / "finalized_p1" / f"{arm.lower()}_selected_ema_state.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        engine = stream_type(model, bank)
        constructor[arm] = (time.perf_counter_ns()-begun)/1e6
        begun = time.perf_counter_ns()
        engine.refresh_state()
        reset[arm] = (time.perf_counter_ns()-begun)/1e6
        engines[arm] = engine
    times, first, errors = {name: [] for name in NAMES}, {}, {name: 0. for name in NAMES}
    raw_oracle = np.zeros((4, 100, 64), dtype=np.float32)
    indices = {0, 99, 100, len(values)-1}
    for index, raw in enumerate(values):
        raw_oracle[:, :-1] = raw_oracle[:, 1:]
        raw_oracle[:, -1] = raw
        shift = index % len(NAMES)
        outputs = {}
        for name in NAMES[shift:] + NAMES[:shift]:
            begun = time.perf_counter_ns()
            output = engines[name].predict(raw)
            elapsed = (time.perf_counter_ns()-begun)/1e6
            assert_public(name, output)
            outputs[name] = output
            if index == 0:
                first[name] = elapsed
            elif index >= 1+args.warmup:
                times[name].append(elapsed)
        for name in NAMES:
            engine = engines[name]
            observed = engine.observation_buffer.transpose(1, 0, 2) if name == "ORIGINALdefault" else engine.raw.numpy()
            if not np.array_equal(observed, raw_oracle):
                raise RuntimeError(name + ": independent W100 raw history mismatch")
            if index in indices:
                direct = original_oracle(engine) if name == "ORIGINALdefault" else family_oracle(engine)
                assert_public(name+" native oracle", direct)
                np.testing.assert_allclose(outputs[name], direct, atol=1e-5, rtol=1e-5)
                errors[name] = max(errors[name], float(np.abs(outputs[name]-direct).max()))
    if any(len(row) != args.calls for row in times.values()):
        raise RuntimeError("exact timed-call cardinality mismatch")
    post_audit, post_source = artifact_audit(args.run_root), source_audit(audit)
    if post_audit != audit or post_source != source:
        raise RuntimeError("fresh post-comparison full authority mismatch")
    require_same_files(immutable)
    require_same_files(source["files"])
    measured = {name: stats(row) for name, row in times.items()}
    result = {"schema": "m1_actual_p1_vs_original_public_b4_source_neural_v1",
              "status": "PASS_LONG_PUBLIC_TIMING_ONLY" if args.calls == 2048 else "PASS_CONTRACT_SMOKE_NOT_LATENCY_ACCEPTANCE",
              "batch": 4, "threads": args.threads, "interop_threads": 1, "calls": args.calls, "warmup": args.warmup,
              "lane_sessions": list(LANES), "lane_real_source_offsets": list(OFFSETS),
              "distinct_source_banks": 3, "fourth_lane_repeats_first_bank_with_separate_neural_segment": True,
              "first_call_separate_from_warmup_and_steady": True, "rotating_three_engine_order": True,
              "historical_spint_image": IMAGE, "immutable_pre": immutable, "immutable_post": require_same_files(immutable),
              "pre_artifact": audit, "post_artifact": post_audit, "pre_source": source, "post_source": post_source,
              "proof_sha256": args.proof_sha256, "wrapper_and_module_import_ms": import_ms,
              "constructor_ms": constructor, "separate_reset_ms": reset, "first_public_call_ms": first,
              "constructor_scope": "includes CPU model deserialize/load; candidate additionally constructs paired shells and performs initial zero refresh; explicit second reset timed separately",
              "raw_steady_public_ms": times, "steady_public_call_ms": measured,
              "p95_ratio_to_original_default": {name: measured[name]["p95_ms"]/measured["ORIGINALdefault"]["p95_ms"] for name in ("FLAT", "ROUTE")},
              "oracle_indices": sorted(indices), "max_public_vs_full_native_abs_error": errors,
              "not_official_latency": True, "host_not_exclusive": True, "quality_or_selection_claim": False,
              "targets_deserialized": False, "nwb_deserialized": False, "source_nwb_hash_bytes_for_authority_only": True,
              "parameter_updates": 0, "affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__}
    atomic_json(result, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output), "p95": {name: row["p95_ms"] for name, row in measured.items()},
                      "p95_ratio_to_original_default": result["p95_ratio_to_original_default"], "oracle_error": errors}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--container-reference", action="store_true")
    parser.add_argument("--image-digest", default=IMAGE)
    main(parser.parse_args())
