"""Fail-closed full-public-stream parity validator for frozen e8 M2."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Decoder
from tfpd_exploration.src.m2_same_query_comparator_v1 import core

from .m2_lifted import LiftedFiveTokenM2Decoder

ROOT = Path(__file__).resolve().parents[3]
PAYLOAD = ROOT / (
    "tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/"
    "artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"
)
ADDENDUM = ROOT / "tfpd_exploration/results/m2/family_v1/ext4_e8_spint_dev_pooled_addendum_v1.json"
PAYLOAD_SHA256 = "4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4"
ADDENDUM_SHA256 = "a10963c3b87081fa7f401eeed1ab9c6829c613e479abe835acc0ff43405649e0"
ABS_TOL = 1.0e-5
REL_TOL = 1.0e-5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_tag(session: str) -> str:
    fields = session.removeprefix("ses-").split("-")
    if len(fields) != 4 or not fields[-1].startswith("Run"):
        raise RuntimeError(f"unrecognized ext4 session name: {session!r}")
    return f"{fields[-1]}_{''.join(fields[:3])}"


def require_close(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise RuntimeError(f"{label}: shape drift {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError(f"{label}: non-finite output")
    error = np.abs(actual - expected)
    maximum = float(error.max())
    if not np.all(error <= ABS_TOL + REL_TOL * np.abs(expected)):
        raise RuntimeError(f"{label}: tolerance failure; max absolute error={maximum:.9g}")
    return maximum


def native_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    if target64.shape != prediction64.shape or target64.ndim != 2:
        raise RuntimeError("invalid native R2 array shape")
    denominator = np.square(target64 - target64.mean(axis=0, keepdims=True)).sum()
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("native R2 target variance is invalid")
    return float(1.0 - np.square(target64 - prediction64).sum() / denominator)


def authority_paths(session: str) -> dict[str, Path]:
    root = data._session_dir("ext4", session)
    return {
        "X_store": root / "X_store.npy",
        "target_store": root / "target_store.npy",
        "eligible_starts": root / "eligible_starts.npy",
        "mapping": root / "mapping.json",
        "provenance": root / "provenance.json",
        "bank_e0_u": root / "e0_u.pt",
        "bank_t": root / "T.npy",
    }


def authority_hashes(session: str) -> dict[str, str]:
    paths = authority_paths(session)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"{session}: missing compact-cache authority: {missing}")
    return {name: file_sha256(path) for name, path in paths.items()}


def module_hashes() -> dict[str, str]:
    runtime = ROOT / "tfpd_exploration/src/m2_runtime_v3/runtime.py"
    siblings = Path(__file__).parent
    paths = {
        "complete_m2_lifted.py": Path(__file__),
        "m2_lifted.py": siblings / "m2_lifted.py",
        "m2.py": siblings / "m2.py",
        "runtime_v3.py": runtime,
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def load_sealed_rows() -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if file_sha256(PAYLOAD) != PAYLOAD_SHA256:
        raise RuntimeError("frozen e8 payload byte hash drift before compute")
    if file_sha256(ADDENDUM) != ADDENDUM_SHA256:
        raise RuntimeError("sealed ext4 addendum byte hash drift before compute")
    receipt = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    sessions = list(plan.EXT4_SESSIONS)
    rows = receipt.get("rows")
    if (
        receipt.get("schema") != "m2_family_v1_ext4_e8_spint_dev_replay_v1"
        or receipt.get("e8_payload_sha256") != PAYLOAD_SHA256
        or receipt.get("sessions") != sessions
        or not isinstance(rows, list)
        or len(rows) != len(sessions)
    ):
        raise RuntimeError("sealed ext4 addendum structure drift")
    by_session = {row.get("session"): row for row in rows if isinstance(row, dict)}
    if set(by_session) != set(sessions):
        raise RuntimeError("sealed ext4 addendum session/cardinality drift")
    required = {
        "session", "window_count", "ordered_window_starts_sha256", "target_sha256",
    }
    if any(not required.issubset(row) for row in by_session.values()):
        raise RuntimeError("sealed ext4 addendum row fields drift")
    return by_session, receipt


def validate_session(
    session: str, sealed: dict[str, object], pre_authority: dict[str, str]
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    bank = data.load_session_bank("ext4", session)
    starts = np.ascontiguousarray(bank.eligible_starts, dtype=np.int64)
    target = np.ascontiguousarray(bank.target_store, dtype=np.float32)
    raw = np.ascontiguousarray(bank.X_store, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != plan.CHANNELS or raw.shape[0] < plan.WINDOW:
        raise RuntimeError(f"{session}: raw stream geometry drift: {raw.shape}")
    if target.ndim != 2 or target.shape != (starts.size, plan.OUT_DIM):
        raise RuntimeError(f"{session}: target/start cardinality drift")
    if starts.size == 0 or np.any(np.diff(starts) <= 0):
        raise RuntimeError(f"{session}: starts must be nonempty and strictly increasing")
    endpoint_ticks = starts + plan.WINDOW - 1
    if int(starts[0]) < 0 or int(endpoint_ticks[-1]) >= raw.shape[0]:
        raise RuntimeError(f"{session}: endpoint exceeds raw public stream")
    if not np.array_equal(raw[: plan.WINDOW - 1], np.zeros_like(raw[: plan.WINDOW - 1])):
        raise RuntimeError(f"{session}: W49 leading padding drift")
    if (
        starts.size != int(sealed["window_count"])
        or core.array_sha256(starts) != sealed["ordered_window_starts_sha256"]
        or core.array_sha256(target) != sealed["target_sha256"]
    ):
        raise RuntimeError(f"{session}: sealed start/target/cardinality drift")

    # reset() supplies the specified zero W49 startup state.  Those padding
    # rows are not public observations; only post-padding raw bins are fed to
    # predict(), while endpoint coordinates remain in padded raw coordinates.
    v3 = RuntimeV3Decoder(PAYLOAD, batch_size=1)
    lifted = LiftedFiveTokenM2Decoder(PAYLOAD, batch_size=1)
    tag = payload_tag(session)
    v3.reset([tag])
    lifted.reset([tag])
    prediction = np.empty_like(target)
    direct_full = np.empty_like(target)
    v3_oracle = np.empty_like(target)
    endpoint_index = {int(tick): index for index, tick in enumerate(endpoint_ticks)}
    public_max_error = 0.0
    visited_count = 0

    for tick, bin_value in enumerate(raw[plan.WINDOW - 1:], start=plan.WINDOW - 1):
        observation = np.ascontiguousarray(bin_value[None, :], dtype=np.float32)
        lifted_native = lifted.predict(observation)
        v3_native = v3.predict(observation)
        public_max_error = max(
            public_max_error,
            require_close(lifted_native, v3_native, f"{session}: public bin {tick}"),
        )
        endpoint = endpoint_index.get(tick)
        if endpoint is not None:
            window_start = int(starts[endpoint])
            independent_window = np.ascontiguousarray(
                raw[window_start:window_start + plan.WINDOW], dtype=np.float32
            )
            engine_window = lifted._engine.raw[0].numpy()
            if not np.array_equal(engine_window, independent_window):
                raise RuntimeError(f"{session}: stream window drift at endpoint {tick}")
            # The full-model oracle receives the fixed-cache window directly;
            # it does not reuse the streaming engine's raw tensor.
            direct_native = direct_full_native(lifted, independent_window)
            require_close(lifted_native[:1], direct_native, f"{session}: direct endpoint {tick}")
            prediction[endpoint] = lifted_native[0]
            direct_full[endpoint] = direct_native[0]
            v3_oracle[endpoint] = v3_native[0]
            visited_count += 1

    if visited_count != starts.size:
        raise RuntimeError(f"{session}: visited {visited_count} of {starts.size} sealed endpoints")
    endpoint_direct_error = require_close(prediction, direct_full, f"{session}: endpoint/direct")
    endpoint_v3_error = require_close(prediction, v3_oracle, f"{session}: endpoint/v3")
    score = native_r2(target, prediction)
    if abs(score - float(sealed["e8_r2"])) > ABS_TOL:
        raise RuntimeError(f"{session}: sealed historical native R2 drift: {score:.12g}")
    post_authority = authority_hashes(session)
    if post_authority != pre_authority:
        raise RuntimeError(f"{session}: compact-cache/provenance/map/bank authority mutated")
    report = {
        "payload_dataset_tag": tag,
        "runtime_bank_authority": "hash-bound frozen e8 payload bank; compact-cache E0/T hashes are provenance-only",
        "public_bins_consumed_after_initial_W49": int(raw.shape[0] - (plan.WINDOW - 1)),
        "endpoint_count": int(starts.size),
        "visited_endpoint_count": visited_count,
        "starts_sha256": core.array_sha256(starts),
        "targets_sha256": core.array_sha256(target),
        "lifted_prediction_sha256": core.array_sha256(prediction),
        "native_r2": score,
        "max_public_v3_lifted_abs_error": public_max_error,
        "max_endpoint_direct_abs_error": endpoint_direct_error,
        "max_endpoint_v3_abs_error": endpoint_v3_error,
    }
    arrays = {
        "prediction": prediction, "direct_full": direct_full, "v3_oracle": v3_oracle,
        "target": target, "start": starts,
    }
    return report, arrays


@torch.no_grad()
def direct_full_native(
    lifted: LiftedFiveTokenM2Decoder, independent_window: np.ndarray
) -> np.ndarray:
    engine = lifted._engine
    return (
        lifted.model.forward_last(
            torch.from_numpy(np.ascontiguousarray(independent_window[None, :, :])),
            engine.bank,
            engine.bank.unit_mask,
        ).numpy()
        / plan.BEHAVIOR_SCALE
    )


def preflight() -> dict[str, object]:
    """Cheap metadata/cache authority check; does not construct a decoder."""
    sealed_rows, sealed_receipt = load_sealed_rows()
    reports: dict[str, dict[str, object]] = {}
    count = 0
    for session in plan.EXT4_SESSIONS:
        paths = authority_paths(session)
        starts = np.ascontiguousarray(np.load(paths["eligible_starts"]), dtype=np.int64)
        target = np.ascontiguousarray(np.load(paths["target_store"], mmap_mode="r"), dtype=np.float32)
        raw = np.asarray(np.load(paths["X_store"], mmap_mode="r"))
        sealed = sealed_rows[session]
        if (
            starts.size != int(sealed["window_count"])
            or target.shape != (starts.size, plan.OUT_DIM)
            or core.array_sha256(starts) != sealed["ordered_window_starts_sha256"]
            or core.array_sha256(target) != sealed["target_sha256"]
            or raw.ndim != 2 or raw.shape[1] != plan.CHANNELS
            or not np.all(raw[: plan.WINDOW - 1] == 0)
        ):
            raise RuntimeError(f"{session}: metadata preflight authority drift")
        count += int(starts.size)
        reports[session] = {
            "window_count": int(starts.size),
            "starts_sha256": core.array_sha256(starts),
            "targets_sha256": core.array_sha256(target),
            "cache_authority_sha256": authority_hashes(session),
        }
    if count != 2069:
        raise RuntimeError(f"expected exact ext4 endpoint cardinality 2069, got {count}")
    return {
        "schema": "family_runtime_v1_m2_lifted_complete_preflight_v1",
        "status": "PASS_METADATA_ONLY_NO_DECODER",
        "endpoint_count": count,
        "payload_sha256": PAYLOAD_SHA256,
        "sealed_ext4_addendum_sha256": ADDENDUM_SHA256,
        "sealed_e8_pooled_native_r2": sealed_receipt["e8_pooled_r2"],
        "module_sha256": module_hashes(),
        "per_session": reports,
    }


def run(output: Path) -> None:
    output = Path(output)
    archive = output.with_suffix(".npz")
    if output.exists() or archive.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {archive}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sealed_rows, sealed_receipt = load_sealed_rows()
    modules_before = module_hashes()
    cache_before = {session: authority_hashes(session) for session in plan.EXT4_SESSIONS}
    started = time.perf_counter()
    reports: dict[str, dict[str, object]] = {}
    collected: dict[str, list[np.ndarray | str]] = {
        "prediction": [], "direct_full": [], "v3_oracle": [], "target": [],
        "start": [], "session": [],
    }
    for session in plan.EXT4_SESSIONS:
        report, arrays = validate_session(session, sealed_rows[session], cache_before[session])
        reports[session] = report
        for name in ("prediction", "direct_full", "v3_oracle", "target", "start"):
            collected[name].append(arrays[name])
        collected["session"].extend([session] * len(arrays["start"]))
    prediction = np.concatenate(collected["prediction"])
    direct_full = np.concatenate(collected["direct_full"])
    v3_oracle = np.concatenate(collected["v3_oracle"])
    target = np.concatenate(collected["target"])
    starts = np.concatenate(collected["start"])
    sessions = np.asarray(collected["session"])
    if prediction.shape[0] != 2069:
        raise RuntimeError(f"expected exact ext4 endpoint cardinality 2069, got {prediction.shape[0]}")
    direct_error = require_close(prediction, direct_full, "global endpoint/direct")
    v3_error = require_close(prediction, v3_oracle, "global endpoint/v3")
    public_error = max(float(row["max_public_v3_lifted_abs_error"]) for row in reports.values())
    pooled_r2 = native_r2(target, prediction)
    historical_pooled_r2 = float(sealed_receipt["e8_pooled_r2"])
    if abs(pooled_r2 - historical_pooled_r2) > ABS_TOL:
        raise RuntimeError(
            f"pooled native R2 drift from sealed e8 authority: {pooled_r2:.12g}"
        )
    modules_after = module_hashes()
    cache_after = {session: authority_hashes(session) for session in plan.EXT4_SESSIONS}
    if modules_after != modules_before or cache_after != cache_before:
        raise RuntimeError("code/module or compact cache/provenance/map/bank authority mutated")

    # Save targets, starts, and session IDs only after every check above passes.
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive, prediction=prediction, direct_full=direct_full, v3_oracle=v3_oracle,
        target=target, session=sessions, start=starts,
    )
    receipt = {
        "schema": "family_runtime_v1_m2_lifted_complete_ext4_v3",
        "status": "PASS",
        "payload_sha256": PAYLOAD_SHA256,
        "sealed_ext4_addendum_sha256": ADDENDUM_SHA256,
        "per_session": reports,
        "endpoint_count": int(prediction.shape[0]),
        "pooled_native_r2": pooled_r2,
        "sealed_historical_e8_pooled_native_r2": historical_pooled_r2,
        "historical_prediction_bitwise_comparison": "not required; lifted attention reassociation is tolerance-checked against exact direct full-model endpoints",
        "global_public_v3_lifted_max_abs_error": public_error,
        "global_endpoint_direct_max_abs_error": direct_error,
        "global_endpoint_v3_max_abs_error": v3_error,
        "stream_npz_sha256": file_sha256(archive),
        "elapsed_seconds": time.perf_counter() - started,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "module_sha256_pre": modules_before,
        "module_sha256_post": modules_after,
        "cache_provenance_map_bank_sha256_pre": cache_before,
        "cache_provenance_map_bank_sha256_post": cache_after,
        "parameter_updates": 0,
        "not_selection_or_ranking": True,
        "not_hidden_or_official": True,
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        if args.output is not None:
            parser.error("--preflight does not write --output")
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    elif args.output is None:
        parser.error("--output is required unless --preflight is supplied")
    else:
        run(args.output)
