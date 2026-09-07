"""Review-gated as-shipped original-M1 score on the finalized P1 surface.

The released image's payload is deliberately distinct from the local
source-fold teacher.  This module does no model/source work at import time and
refuses to run before both the P1 finalizer and its complete selected-stream
proof are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .complete_m1_family_source import (
    COUNT, PUBLIC_CALLS, SESSIONS, archive_audit, artifact_audit, array_sha,
    metric, require_same_files, sha, source_audit, typed_array_sha,
)
from .m1_family_spint_comparison import EXPECTED, IMAGE, PAYLOAD, WRAPPER

W, UNITS, OUTPUTS, SCALE = 100, 64, 16, 1.0
GO = "ORIGINAL_M1_FROZEN_GO"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tag_for(session: str) -> Path:
    if session not in SESSIONS:
        raise RuntimeError("exact source-session tag required")
    return Path(f"sub-MonkeyL-held-in-calib_{session}_behavior+ecephys")


def assert_public(value: object) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.shape != (1, OUTPUTS)
            or value.dtype != np.float32 or not value.flags.owndata
            or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise RuntimeError("released original public output must be owning FP32 [1,16]")
    return value


def assert_input(value: object) -> np.ndarray:
    """Validate a raw B1 bin before calling a stateful public API.

    This ordering matters: an invalid raw bin must not be allowed to advance
    the released decoder's history, even in a diagnostic replay.
    """
    if (not isinstance(value, np.ndarray) or value.shape != (1, UNITS)
            or value.dtype != np.float32 or not value.flags.c_contiguous
            or not np.isfinite(value).all()):
        raise RuntimeError("released original raw input must be finite C-order FP32 [1,64]")
    return value


def assert_constructor_classifier_cpu(engine) -> None:
    """Constructor-time CPU check; the released wrapper creates `.device` in reset.

    The frozen image wrapper deserializes `clf` in ``__init__`` and only sets
    ``self.device`` in ``reset``.  Parameters and buffers are therefore the
    valid pre-reset placement authority.
    """
    clf = getattr(engine, "clf", None)
    if clf is None or not callable(getattr(clf, "parameters", None)) or not callable(getattr(clf, "buffers", None)):
        raise RuntimeError("as-shipped original M1 constructor classifier contract drift")
    values = [*clf.parameters(), *clf.buffers()]
    for value in values:
        device = getattr(value, "device", None)
        if getattr(device, "type", None) != "cpu":
            raise RuntimeError("as-shipped original M1 constructor has non-CPU classifier state")


def _raw(engine) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(engine.observation_buffer).transpose(1, 0, 2))


def _native(engine, history: np.ndarray) -> np.ndarray:
    """Independent image-model forward; excluded from the public-call count."""
    import torch
    if (engine.device.type != "cpu" or engine.smooth_calibration
            or len(engine.local_calib_trial_features) != 1
            or tuple(engine.local_calib_trial_features[0].shape) != (10, 1024, UNITS)):
        raise RuntimeError("original post-reset CPU/unsmoothed/B1 contract drift")
    if (not isinstance(history, np.ndarray) or history.shape != (1, W, UNITS)
            or history.dtype != np.float32 or not history.flags.c_contiguous
            or not np.isfinite(history).all()):
        raise RuntimeError("native original history must be finite C-order FP32 [1,100,64]")
    raw = torch.from_numpy(history)
    with torch.no_grad():
        value = engine.local_clf(
            raw,
            calib_trialized_neural_features=engine.local_calib_trial_features[0].unsqueeze(0).to(raw),
        )[:, -1]
    value = np.ascontiguousarray(value.numpy() / engine.behavior_scaling_factor, dtype=np.float32)
    return assert_public(value)


def _require_gate(out: Path, *, container_reference: bool, image_digest: str, threads: int) -> None:
    if out.exists():
        raise FileExistsError(out)
    if (os.environ.get(GO) != "1" or not container_reference or image_digest != IMAGE
            or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1") or threads not in (1, 2)):
        raise RuntimeError("explicit original-M1 image / CPU T1-or-T2 review gate required")


def require_proof(path: Path, expected_sha: str, audit: dict) -> dict:
    if len(expected_sha) != 64 or sha(path) != expected_sha:
        raise RuntimeError("complete selected-stream proof SHA drift")
    proof = json.loads(path.read_text())
    if (proof.get("status") != "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY"
            or proof.get("outer_query_opened") is not False
            or proof.get("parameter_updates") != 0
            or proof.get("batch") != 1
            or proof.get("pre_artifact") != audit or proof.get("post_artifact") != audit
            or proof.get("pre_source") != proof.get("post_source")):
        raise RuntimeError("complete selected-stream proof authority mismatch")
    for arm in ("flat", "route"):
        row = proof.get("arms", {}).get(arm, {})
        if (row.get("selected") != audit["selected"][arm] or row.get("scored_count") != COUNT
                or row.get("public_calls") != sum(PUBLIC_CALLS)
                or row.get("initial_current_predictions") != len(SESSIONS)
                or not np.isfinite(row.get("max_abs_error", np.nan))
                or not 0 <= float(row["max_abs_error"]) <= 1e-5):
            raise RuntimeError("complete selected-stream proof is incomplete")
    require_same_files(proof["pre_source"]["files"])
    return proof


def image_and_helper_closure(proof: Path) -> dict[str, str]:
    """Every image Python source and scorer dependency is frozen pre/post."""
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError("frozen original image file SHA drift: " + str(path))
    helpers = (Path(__file__), Path(__file__).with_name("complete_m1_family_source.py"),
               Path(__file__).with_name("m1_family_spint_comparison.py"), proof)
    image_python = sorted(Path("/src").rglob("*.py"))
    paths = {str(path): path for path in (*EXPECTED, *image_python, *helpers)}
    return {name: sha(path) for name, path in paths.items()}


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("_original_m1_frozen_wrapper", WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("released original M1 wrapper unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(archive: dict[str, np.ndarray], source: dict) -> dict[str, dict]:
    """Use exact finalized target/session/start arrays; cache contributes raw only."""
    rows = {}
    with np.load(source["cache"], allow_pickle=False) as cache:
        offset = 0
        for session, expected_count, calls in zip(SESSIONS, (10567, 9705, 10980), PUBLIC_CALLS, strict=True):
            ids = np.flatnonzero(archive["session"] == session)
            if (len(ids) != expected_count or not np.array_equal(ids, np.arange(offset, offset + expected_count))
                    or not np.all(archive["session"][ids] == session)):
                raise RuntimeError("finalized source session ordering/cardinality drift")
            starts = np.asarray(archive["start"][ids], dtype=np.int64)
            if (not len(starts) or not np.all(np.diff(starts) > 0)
                    or int(starts[-1] - starts[0]) != calls or starts[0] < 0):
                raise RuntimeError("finalized source endpoint topology drift")
            raw_value = cache[f"raw_neural/{session}"]
            if (raw_value.dtype != np.float32 or raw_value.ndim != 2 or raw_value.shape[1] != UNITS
                    or not np.isfinite(raw_value).all()):
                raise RuntimeError("raw cache must already be finite FP32 [time,64]")
            raw = np.ascontiguousarray(raw_value)
            target = archive["target"][ids]
            if (target.dtype != np.float32 or target.shape != (expected_count, OUTPUTS)
                    or not np.isfinite(target).all()
                    or not np.all(raw[: W - 1] == 0)
                    or array_sha(raw) != source["cache_receipt"]["arrays"][session]["raw_neural_sha256"]
                    or starts[0] < 0 or starts[-1] + W > len(raw)):
                raise RuntimeError("raw cache/finalized endpoint mismatch")
            rows[session] = {"raw": raw, "starts": starts, "target": np.ascontiguousarray(target.copy())}
            offset += expected_count
        if offset != COUNT:
            raise RuntimeError("finalized source cardinality drift")
    return rows


def validate_row(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reject malformed source rows before the stateful decoder is reset."""
    if not isinstance(row, dict) or set(row) != {"raw", "starts", "target"}:
        raise RuntimeError("replay row schema drift")
    raw, starts, target = row["raw"], row["starts"], row["target"]
    if (not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.ndim != 2
            or raw.shape[1] != UNITS or not raw.flags.c_contiguous or not np.isfinite(raw).all()):
        raise RuntimeError("replay raw must be finite C-order FP32 [time,64]")
    if (not isinstance(starts, np.ndarray) or starts.dtype != np.int64 or starts.ndim != 1
            or not len(starts) or not np.all(np.diff(starts) > 0) or int(starts[0]) < 0
            or int(starts[-1]) + W > len(raw)):
        raise RuntimeError("replay start geometry/order/range drift")
    if (not isinstance(target, np.ndarray) or target.dtype != np.float32
            or target.shape != (len(starts), OUTPUTS) or not target.flags.c_contiguous
            or not np.isfinite(target).all()):
        raise RuntimeError("replay target must be finite C-order FP32 [n,16]")
    return raw, starts, target


def replay_session(engine, row: dict, session: str, progress=None) -> dict:
    """One B1 reset plus all raw bins between first and last scored endpoint."""
    raw, starts, target = validate_row(row)
    engine.reset([tag_for(session)])
    if (engine.device.type != "cpu" or engine.window_size != W or engine.behavior_scaling_factor != SCALE
            or engine.smooth_calibration or any(module.training for module in engine.local_clf.modules())
            or len(engine.local_calib_trial_features) != 1
            or tuple(engine.local_calib_trial_features[0].shape) != (10, 1024, UNITS)):
        raise RuntimeError("as-shipped original M1 reset contract drift")
    calib_shape = tuple(int(v) for v in engine.local_calib_trial_features[0].shape)
    first, last = int(starts[0]), int(starts[-1])
    history = np.ascontiguousarray(raw[first:first + W][None].copy())
    engine.observation_buffer = np.ascontiguousarray(history.transpose(1, 0, 2).copy())
    if not np.array_equal(_raw(engine), history):
        raise RuntimeError("released original B1 primed raw-history drift before native initial prediction")
    # The first endpoint comes from an independent native forward: it is not
    # represented as a fictional public predict call.
    predictions = [assert_public(_native(engine, history))[0].copy()]
    endpoint_to_index = {int(start + W - 1): index for index, start in enumerate(starts)}
    subset = {first + W - 1, first + W + 3, first + 2 * W - 1, first + 2 * W, last + W - 1}
    gaps = starts[1:][np.diff(starts) > 1]
    if len(gaps): subset.add(int(gaps[0] + W - 1))
    direct_count, maximum, public_calls = 1, 0.0, 0
    cursor = first + W - 1
    while cursor < last + W - 1:
        cursor += 1
        history[:, :-1] = history[:, 1:]
        history[:, -1] = raw[cursor]
        item = assert_input(np.ascontiguousarray(raw[cursor:cursor + 1]))
        public = assert_public(engine.predict(item))
        public_calls += 1
        if not np.array_equal(_raw(engine), history):
            raise RuntimeError("released original B1 W100 raw-history drift")
        if cursor in subset:
            direct = assert_public(_native(engine, history.copy()))
            np.testing.assert_allclose(public, direct, atol=1e-5, rtol=1e-5)
            maximum = max(maximum, float(np.abs(public - direct).max())); direct_count += 1
        if cursor in endpoint_to_index:
            predictions.append(public[0].copy())
        if progress is not None and (public_calls % 2048 == 0 or cursor == last + W - 1):
            progress({"session": session, "public_calls": public_calls, "scored_count": len(predictions)})
    prediction = np.asarray(predictions, dtype=np.float32)
    if prediction.shape != target.shape or len(prediction) != len(starts):
        raise RuntimeError("source endpoint prediction cardinality drift")
    return {"prediction": prediction, "target": target, "session": np.asarray([session] * len(starts)),
            "start": starts, "public_calls": public_calls, "initial_native_predictions": 1,
            "actual_calibration_shape": list(calib_shape),
            "direct_native_count": direct_count, "max_native_abs_error": maximum}


def run(run_root: Path, out: Path, proof: Path, proof_sha256: str, *, container_reference=False,
        image_digest=IMAGE, threads=2) -> dict:
    out, run_root, proof = Path(out), Path(run_root), Path(proof)
    _require_gate(out, container_reference=container_reference, image_digest=image_digest, threads=threads)
    # Completion and proof gates precede image/model/source imports.
    audit = artifact_audit(run_root)
    full_proof = require_proof(proof, proof_sha256, audit)
    pre_image = image_and_helper_closure(proof)
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("original frozen M1 scorer requires CPU")
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    # Bind the image namespace before trainer/cache helpers modify import paths.
    import src.models.components.spint as image_spint
    if Path(image_spint.__file__).resolve() != Path("/src/models/components/spint.py"):
        raise RuntimeError("image-owned SPINT module was not imported")
    wrapper = _load_wrapper()
    source = source_audit(audit)
    if source != full_proof["pre_source"]:
        raise RuntimeError("current source authority differs from completed P1 proof")
    archive_path = run_root / "finalized_p1" / "flat_selected_native_source_dev.npz"
    # The finalized FLAT and ROUTE archives must have equal target/session/start
    # identity; either archive is valid as the common source endpoint authority.
    flat = archive_audit(archive_path, audit["final"]["exports"]["flat_selected"])
    route = archive_audit(run_root / "finalized_p1" / "route_selected_native_source_dev.npz",
                          audit["final"]["exports"]["route_selected"])
    for key in ("target", "session", "start"):
        if not np.array_equal(flat[key], route[key]):
            raise RuntimeError("finalized selected archive query arrays differ")
    rows = _rows(flat, source)
    from falcon_challenge.config import FalconConfig, FalconTask
    engine = wrapper.SpintDecoder(FalconConfig(task=FalconTask.m1), str(PAYLOAD), batch_size=1)
    # This is deliberately before the first reset: a CUDA/default-device
    # constructor is an authority failure, not something reset may repair.
    if (engine.window_size != W or engine.behavior_scaling_factor != SCALE or engine.smooth_calibration):
        raise RuntimeError("as-shipped original M1 CPU constructor contract drift")
    assert_constructor_classifier_cpu(engine)
    out.mkdir(parents=True)
    atomic_json(out / "input_authority.json", {"status": "RUNNING", "artifact": audit,
                "proof_sha256": proof_sha256, "source": source, "image_pre": pre_image})
    started, pieces, reports = time.monotonic(), [], []
    def progress(value):
        atomic_json(out / "live.json", {"status": "RUNNING", "elapsed_seconds": time.monotonic()-started, **value})
    for session in SESSIONS:
        row = replay_session(engine, rows[session], session, progress)
        pieces.append({key: row.pop(key) for key in ("prediction", "target", "session", "start")})
        reports.append({"session": session, **row})
    arrays = {key: np.concatenate([piece[key] for piece in pieces]) for key in ("prediction", "target", "session", "start")}
    if (arrays["prediction"].shape != (COUNT, OUTPUTS) or arrays["prediction"].dtype != np.float32
            or sum(row["public_calls"] for row in reports) != sum(PUBLIC_CALLS)
            or sum(row["initial_native_predictions"] for row in reports) != len(SESSIONS)):
        raise RuntimeError("fixed 31252 endpoint / 112985 public-call cardinality drift")
    observed = metric(arrays)
    post_audit = artifact_audit(run_root)
    post_source = source_audit(post_audit)
    post_image = image_and_helper_closure(proof)
    if pre_image != post_image:
        raise RuntimeError("image/wrapper/payload/self/proof changed during replay")
    if post_audit != audit or post_source != source:
        raise RuntimeError("fresh post-replay artifact/source authority mismatch")
    require_same_files(source["files"])
    archive = out / "original_m1_frozen_selected_source_dev_fp32.npz"
    atomic_npz(archive, arrays)
    receipt = {"schema": "original_m1_as_shipped_frozen_selected_source_dev_v1",
               "status": "PASS_AS_SHIPPED_ORIGINAL_M1_REFERENCE_ONLY", "image": IMAGE,
               "scope": "exact finalized 31252 source endpoints; original M10 per-session reset; no fitting/selection/update",
               "pre_artifact": audit, "post_artifact": post_audit,
               "pre_source": source, "post_source": post_source,
               "full_selected_proof_sha256": proof_sha256,
               "image_pre": pre_image, "image_post": post_image, "source": source,
               "archive": {"path": str(archive), "sha256": sha(archive),
                           "typed_array_sha256": {key: typed_array_sha(value) for key, value in arrays.items()}},
               "metrics_float64": observed, "sessions": reports, "scored_count": COUNT,
               "public_calls": sum(PUBLIC_CALLS), "initial_native_predictions": len(SESSIONS),
               "window": W, "units": UNITS, "outputs": OUTPUTS, "divisor": SCALE,
               "smooth_calibration": False, "batch": 1, "threads": threads,
               "max_native_abs_error": max(row["max_native_abs_error"] for row in reports),
               "direct_native_count": sum(row["direct_native_count"] for row in reports),
               "parameter_updates": 0, "no_selection_or_fit": True, "outer_query_opened": False,
               "not_latency_measurement": True,
               "original_m10_training_calibration": "historical original-M10 payload; its training/calibration procedure is not reconstructed or equalized here",
               "family_p1_training_calibration": "source_dev protocol begins after M10; first 10 source calibration trials feed the frozen B3 identity encoder, while rSyn3-refit-v1 is the separate source-only T-carrier revision",
               "training_exposure_comparison": "both are M1/M10-context comparators, but their historical training and calibration methods are not equalized; reference-only",
               "elapsed_seconds": time.monotonic()-started}
    atomic_json(out / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--container-reference", action="store_true")
    parser.add_argument("--image-digest", default=IMAGE)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    result = run(args.run_root, args.out, args.proof, args.proof_sha256,
                 container_reference=args.container_reference, image_digest=args.image_digest, threads=args.threads)
    print(json.dumps({key: result[key] for key in ("status", "scored_count", "public_calls", "metrics_float64", "max_native_abs_error")}, sort_keys=True))
