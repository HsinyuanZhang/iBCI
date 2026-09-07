"""Fixed H1 endpoint-12 RAW native forwards with frozen selected-EMA controls.

No optimizer, RNG restoration, parameter updates, model selection or promotion.
All evaluated identities are the previously frozen source208 and complete20325.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np

ARMS = ("flat", "route")
COUNT, LIMIT, MEMORY = 20325, 2400, 22 << 30

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError("JSON authority must be an object")
    return value

def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError("explicit lowercase SHA-256 required")
    return value

def _bind(files, path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != _hash(expected):
        raise RuntimeError("bound input SHA drift: " + str(path))
    if files.setdefault(str(path), digest) != digest:
        raise RuntimeError("conflicting input binding")
    return path

def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)

def atomic_npz(arrays, path):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _collect(formal, proof, proof_sha, source_ref, source_ref_sha):
    # Snapshot helper bytes before their imports and any model/cache load.
    here = Path(__file__).resolve()
    files = {}
    for path in (here, here.with_name("complete_h1_family_source.py"),
                 here.with_name("h1_replay_contract.py"),
                 here.with_name("diagnose_h1_selected_source208.py"),
                 here.with_name("h1_family_spint_comparison.py"),
                 here.parents[1] / "h1_family_v1/cold_phase_summary.py"):
        _bind(files, path)
    _bind(files, proof, _hash(proof_sha))
    _bind(files, source_ref, _hash(source_ref_sha))
    from . import diagnose_h1_selected_source208 as selected
    from .h1_family_spint_comparison import require_proof
    pre = selected.preflight_audit(formal)
    source = selected.code_source_audit(formal, pre)
    artifact = pre["artifact"]
    require_proof(proof, proof_sha, artifact, source["complete_source"])
    reference = read(source_ref)
    if (reference.get("schema") != "h1_selected_ema_exact_source208_descriptive_v1"
            or reference.get("status") != "COMPLETE_READ_ONLY_DESCRIPTIVE"
            or reference.get("preflight") != pre or reference.get("postflight") != pre
            or reference.get("source") != source or reference.get("post_source") != source
            or reference.get("parameter_updates") != 0 or reference.get("selection_changed") is not False
            or reference.get("minival_forward") is not False):
        raise RuntimeError("selected source208 authority drift")
    for closure in (artifact["files"], pre["files"], source["complete_source"]["files"]):
        for path, digest in closure.items():
            _bind(files, path, _hash(digest))
    checkpoints, baselines, ready = {}, {}, {}
    for arm in ARMS:
        checkpoint = formal / "checkpoints" / f"{arm}_epoch_012.pt"
        ledger = formal / "workers" / f"{arm}_epoch_012_metrics.json"
        row = read(ledger)
        if row.get("epoch") != 12 or row.get("checkpoint") != str(checkpoint):
            raise RuntimeError("fixed endpoint12 ledger identity drift")
        _bind(files, checkpoint, _hash(row.get("checkpoint_sha256")))
        _bind(files, ledger, artifact["files"][str(ledger)])
        checkpoints[arm] = {"path": str(checkpoint), "sha256": files[str(checkpoint)],
                            "ledger_path": str(ledger), "ledger_sha256": files[str(ledger)]}
        ready[arm] = read(formal / "barrier" / f"{arm}.ready.json")
        item = reference["archives"][arm]
        source_path = _bind(files, item["path"], _hash(item.get("sha256")))
        complete_path = formal / "exports" / f"{arm}_selected_complete_native_float64.npz"
        _bind(files, complete_path, artifact["files"][str(complete_path)])
        baselines[arm] = {"source208": str(source_path), "complete": str(complete_path)}
    if {path: sha(path) for path in files} != files:
        raise RuntimeError("input mutation during preflight")
    return {"preflight": pre, "source": source, "files": files,
            "checkpoints": checkpoints, "ready": ready, "baselines": baselines,
            "proof": str(proof), "proof_sha256": proof_sha,
            "source208_reference": str(source_ref), "source208_reference_sha256": source_ref_sha}

def preflight(formal, out, proof, proof_sha, source_ref, source_ref_sha):
    if out.exists():
        raise FileExistsError(out)
    if (os.environ.get("H1_ENDPOINT_RAW_GO") != "1"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or any(not Path(p).is_absolute() for p in (formal, out, proof, source_ref))):
        raise RuntimeError("explicit GO/absolute paths/visible cuda:0 required")
    _hash(proof_sha); _hash(source_ref_sha)
    return _collect(formal, proof, proof_sha, source_ref, source_ref_sha)

def validate_payload(payload, arm, binding):
    import torch
    if (payload.get("schema") != "h1_crst_b4_splitarm_end_epoch_checkpoint_v1"
            or payload.get("arm") != arm or payload.get("epoch") != 12
            or payload.get("global_step") != 8772 or payload.get("next_batch_index") != 731
            or payload.get("ema", {}).get("n_updates") != 8772
            or payload.get("ema", {}).get("decay") != .9995):
        raise RuntimeError("endpoint RAW checkpoint schema/identity drift")
    frozen = binding["source"]["complete_source"]["frozen"]
    protocol_digest = hashlib.sha256((json.dumps(frozen["protocol"], sort_keys=True,
                                                separators=(",", ":")) + "\n").encode()).hexdigest()
    ready = binding["ready"][arm]
    expected = {"protocol_sha256": protocol_digest,
                "source_authority_sha256": frozen["bindings"]["source_authority_sha256"],
                "code_sha256": frozen["code_closure"],
                "sampler_identity_sha256": ready["identities"]["12"]["sampler_sha256"],
                "dropout_identity_sha256": ready["identities"]["12"]["keep_sha256"],
                "shared_init_sha256": ready["shared_sha256"]}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("endpoint RAW checkpoint source/code/paired identity drift")
    state = payload.get("model")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(x, torch.Tensor) or x.dtype != torch.float32
                   or not bool(torch.isfinite(x).all()) for x in state.values())):
        raise RuntimeError("endpoint RAW state FP32/finite drift")
    return state

class RawScore:
    def score_with_ema(self, model, callback):
        return callback(model)

class GuardedModel:
    """Inference-only proxy; no optimizer/EMA substitution or model method edits."""
    def __init__(self, model, check):
        self.model, self.check = model, check
        self.calls, self.rows = 0, 0
    def eval(self):
        self.model.eval()
        return self
    def forward_last(self, x, bank):
        self.check(self.calls, self.rows)
        value = self.model.forward_last(x, bank)
        self.calls += 1; self.rows += len(x)
        self.check(self.calls, self.rows)
        return value

def check_archive(arrays, baseline, *, source208):
    n, coordinate = (208, "start") if source208 else (COUNT, "end")
    if (set(arrays) != {"prediction", "target", "session_id", coordinate}
            or arrays["prediction"].shape != (n, 7) or arrays["target"].shape != (n, 7)
            or arrays["prediction"].dtype != np.float64 or arrays["target"].dtype != np.float64
            or arrays["session_id"].shape != (n,) or arrays["session_id"].dtype.kind != "U"
            or arrays[coordinate].shape != (n,) or arrays[coordinate].dtype != np.int64
            or not np.isfinite(arrays["prediction"]).all() or not np.isfinite(arrays["target"]).all()):
        raise RuntimeError("RAW archive shape/dtype/finite drift")
    if any(not np.array_equal(arrays[key], baseline[key]) for key in ("target", "session_id", coordinate)):
        raise RuntimeError("RAW/selected EMA exact query metadata drift")

def run(formal, out, proof, proof_sha, source_ref, source_ref_sha):
    binding = preflight(formal, out, proof, proof_sha, source_ref, source_ref_sha)
    import torch
    if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError("visible cuda:0 unavailable")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    from tfpd_exploration.src.h1_optimized_v2.cache import CACHE, ROOT, validate_authority
    from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    from tfpd_exploration.src.h1_family_v1.familyformal_split_train import score_complete_cached_ema
    from tfpd_exploration.src.h1_family_v1 import cold_phase_summary as summary
    from . import diagnose_h1_selected_source208 as selected
    from .complete_h1_family_source import check_metadata
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    authority = read(ROOT / "source_cache_authority.json")
    validate_authority(cache, authority)
    fixed = selected._require_ids()
    selected.validate_manifest_against_cache(cache, authority, fixed)
    baselines = {}
    for arm in ARMS:
        baselines[arm] = {}
        for mode in ("source208", "complete"):
            path = binding["baselines"][arm][mode]
            if sha(path) != binding["files"][path]:
                raise RuntimeError("baseline changed before archive load")
            baselines[arm][mode] = summary.load(path, source208=mode == "source208")
        check_metadata(baselines[arm]["complete"], cache)
    for mode, coordinate in (("source208", "start"), ("complete", "end")):
        if any(not np.array_equal(baselines["flat"][mode][key], baselines["route"][mode][key])
               for key in ("target", "session_id", coordinate)):
            raise RuntimeError("selected FLAT/ROUTE archive metadata drift")
    out.mkdir(parents=True)
    atomic_json({"status": "RUNNING_FIXED_ENDPOINT12_RAW", "bindings": binding}, out / "input_authority.json")
    dev, results = torch.device("cuda:0"), {}
    for arm, index in (("flat", 0), ("route", 1)):
        checkpoint = binding["checkpoints"][arm]
        if sha(checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError("RAW checkpoint changed before deserialization")
        payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
        state = validate_payload(payload, arm, binding)
        expected_state = selected.state_digest(state)
        pair = make_v2_unscaled_dot_localbalanced_pair(seed=42)
        model = pair[index].to(dev); del pair
        model.load_state_dict(state, strict=True); model.eval()
        if selected.state_digest(model.state_dict()) != expected_state:
            raise RuntimeError("strict RAW load changed tensor contents")
        archives, scores = {}, {}
        for mode in ("source208", "complete"):
            last_announced = -1
            def guard(calls, rows):
                nonlocal last_announced
                torch.cuda.synchronize()
                elapsed, peak = time.monotonic() - started, torch.cuda.max_memory_allocated()
                if elapsed > LIMIT or peak > MEMORY:
                    raise RuntimeError("endpoint diagnostic resource bound exceeded")
                if (calls == 0 or calls % 128 == 0) and calls != last_announced:
                    atomic_json({"status": "SCORING", "arm": arm, "surface": mode,
                                 "forward_calls": calls, "rows": rows, "elapsed_seconds": elapsed,
                                 "peak_memory_bytes": peak}, out / "live.json")
                    last_announced = calls
            guarded = GuardedModel(model, guard)
            if mode == "source208":
                reported, arrays = selected.evaluate_source208(
                    guarded, cache, fixed, dev,
                    lambda row, d: H1Bank(*[row["bank"][k].to(d) for k in ("E0", "T", "unit_mask")]))
            else:
                reported = score_complete_cached_ema(model=guarded, ema=RawScore(), cache=cache, device=dev)
                arrays = {key: reported.pop("_" + key) for key in ("prediction", "target", "session_id", "end")}
            check_archive(arrays, baselines[arm][mode], source208=mode == "source208")
            measured = summary.metric(arrays)
            if mode == "source208":
                summary._check_e1(reported, measured)
                measured.update(selected._stats(arrays["prediction"], arrays["target"]))
                base_metric = summary.metric(baselines[arm][mode])
                base_metric.update(selected._stats(baselines[arm][mode]["prediction"], baselines[arm][mode]["target"]))
                analysis = {"raw": measured, "selected_ema": base_metric,
                            "pooled_raw_minus_selected_ema": measured["r2_concat_float64"] - base_metric["r2_concat_float64"]}
            else:
                summary._check_e2(reported, measured)
                raw_analysis, base_analysis = summary.analysis(arrays), summary.analysis(baselines[arm][mode])
                analysis = {"raw": raw_analysis, "selected_ema": base_analysis,
                            "raw_minus_selected_ema": summary._delta(raw_analysis, base_analysis)}
            if selected.state_digest(model.state_dict()) != expected_state:
                raise RuntimeError("RAW state changed during diagnostic scoring")
            path = out / f"{arm}_endpoint12_raw_{mode}_float64.npz"
            atomic_npz(arrays, path)
            archives[mode] = {"path": str(path), "sha256": sha(path)}
            scores[mode] = {"reported_native": reported, "analysis": analysis,
                            "forward_calls": guarded.calls, "rows": guarded.rows}
        results[arm] = {"checkpoint": checkpoint, "raw_state_digest": expected_state,
                        "scores": scores, "archives": archives}
        del model, state, payload, guarded
    post = _collect(formal, proof, proof_sha, source_ref, source_ref_sha)
    if post != binding:
        raise RuntimeError("fresh post RAW diagnostic authority drift")
    for result in results.values():
        for archive in result["archives"].values():
            if sha(archive["path"]) != archive["sha256"]:
                raise RuntimeError("RAW output archive changed before receipt")
    elapsed, peak = time.monotonic() - started, torch.cuda.max_memory_allocated()
    if elapsed > LIMIT or peak > MEMORY:
        raise RuntimeError("endpoint diagnostic resource bound exceeded at completion")
    receipt = {"schema": "h1_formal_endpoint12_raw_native_forward_diagnostic_v3",
               "status": "COMPLETE_FIXED_RAW_DIAGNOSTIC_NO_SELECTION", "pre": binding, "post": post,
               "arms": results, "elapsed_seconds": elapsed, "peak_memory_bytes": peak,
               "parameter_updates": 0, "selection_changed": False, "no_selection_or_promotion": True,
               "not_official_or_external": True,
               "interpretation": "fixed original endpoint12 RAW versus original selected EMA; diagnostic only, original EMA selection is unchanged"}
    atomic_json(receipt, out / "receipt.json")
    atomic_json({"status": receipt["status"], "elapsed_seconds": elapsed}, out / "live.json")
    return receipt

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--source208-reference", type=Path, required=True)
    parser.add_argument("--source208-reference-sha256", required=True)
    args = parser.parse_args(argv)
    return run(args.formal, args.output, args.proof, args.proof_sha256,
               args.source208_reference, args.source208_reference_sha256)

if __name__ == "__main__":
    main()
