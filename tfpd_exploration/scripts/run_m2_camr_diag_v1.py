#!/usr/bin/env python3
"""CAMR diagnostics V1: cross-record episodes, zero training (CPU only).

Arms per WORKORDER_M2_CAMR_DIAG_V1_20260903.md.  Every continual arm decodes
the full query stream with the exact shipped-decoder state machine (49-row
zero pre-history, 100-bin tumbling candidates, decode-before-commit, capacity
30, chronological prefix-4 protected).
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "tfpd_exploration/results/m2_camr_diag_v1"
PKG = ROOT / "tfpd_exploration/submissions/evalai_m2_cenat_chunk100e_v1"
AJPF_C_TRAINING = ROOT / "tfpd_exploration/results/m2_ajpf_c_v2"

CHECKPOINT_BODIES = {
    "pooled": "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e",
    "Ce-NAT": "c5672a2bfc94f55729d64e5eeb2f6917162c345713e7102dec4ad371cd147ea2",
}
SEALED_SCREEN = ROOT / "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
R2_ABS_TOLERANCE = 1.0e-7
CHUNK_BINS, POOL_CAPACITY, PROTECTED_PREFIX = 100, 30, 4
PREHISTORY_ZEROS = 49
WARMUP_K = 4


class DiagError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run_date(session: str):
    # 'ses-2020-10-19-Run1' -> ('Run1', '20201019')
    run = session.split("-")[-1]
    date = "".join(session.split("-")[1:4])
    return run, date


def cross_partner(session: str, sessions: list) -> str | None:
    run, date = parse_run_date(session)
    other = "Run2" if run == "Run1" else "Run1"
    for candidate in sessions:
        r2, d2 = parse_run_date(candidate)
        if r2 == other and d2 == date:
            return candidate
    return None


def main() -> None:
    import os
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "diagnostics must run CPU-only")

    for path in (ROOT, PKG, ROOT / "streaming_calibration_exp",
                 ROOT / "tfpd_exploration/submissions/evalai_m2_apfg_static_v1"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import src.models.streaming_calibration_module  # noqa: F401
    sys.path.append(str(ROOT / "SPINT-main"))

    import copy
    import io

    import numpy as np
    import torch

    from falcon_challenge.config import FalconConfig, FalconTask
    from laws import sealed_fit_ridge_side, select_dopt4_support, select_first30_activity_pool
    from sua_exploration.evalai_t4_m2.export_t4_payload import (
        load_frozen_model_and_data, manual_decode)
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts, variance_weighted_r2)
    from cenat_chunk_decoder import PayloadIdEncoder

    torch.set_num_threads(min(8, torch.get_num_threads()))
    started = time.monotonic()
    payload_path = PKG / "artifacts/t4_m2_seed42_cenat_chunk100e_tta.pkl"
    task_config = FalconConfig(task=FalconTask.m2)

    # -- strict-load both checkpoints into one base each
    base, data_module, _tc, meta = load_frozen_model_and_data()
    require(meta["checkpoint_sha256"] == CHECKPOINT_BODIES["pooled"], "base drift")
    cenat_body = AJPF_C_TRAINING / "checkpoints" / "Ce-NAT_epoch12.pt"
    require(sha256_file(cenat_body) == CHECKPOINT_BODIES["Ce-NAT"], "Ce-NAT body drift")
    cenat_state = torch.load(io.BytesIO(cenat_body.read_bytes()), map_location="cpu", weights_only=True)
    cenat = copy.deepcopy(base)
    cenat.load_state_dict(cenat_state, strict=True)
    cenat.eval()
    for parameter in cenat.parameters():
        parameter.requires_grad_(False)

    sealed = json.loads(SEALED_SCREEN.read_text(encoding="utf-8"))
    sealed_rows = {(r["surface"], r["session"]): float(r["r2"])
                   for r in sealed["rows"] if r["cell"] == "ridge_activity30_m4"}

    def session_material(dataset, session: str):
        angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
        selected = select_dopt4_support(angles)
        calibration = np.asarray(dataset.calib_trialized_neural_features[session], dtype=np.float32)
        seed_rows = [np.ascontiguousarray(row, dtype=np.float32) for row in calibration[:30]]
        side, _ = sealed_fit_ridge_side(dataset, session, selected)
        calib_energies = [float(row.mean(dtype=np.float64)) for row in seed_rows]
        return {
            "session": session,
            "neural": np.asarray(dataset.neural_data[session], dtype=np.float32),
            "targets": np.asarray(dataset.covariate_data[session], dtype=np.float32),
            "seed_rows": seed_rows,
            "calib_energies": calib_energies,
            "side": torch.from_numpy(np.ascontiguousarray(side)).unsqueeze(0),
            "selected": [int(v) for v in selected],
        }

    def windows_targets(dataset, session, surface):
        starts = np.asarray([x for name, x in dataset.window_indices if name == session], dtype=np.int64)
        if surface == "within_post30":
            starts = select_common_post30_window_starts(starts, dataset.trial_start_indices[session])
        return starts

    class Pool:
        def __init__(self, seed_rows, protected: int):
            from collections import deque
            self.rows = deque(np.ascontiguousarray(r, dtype=np.float32) for r in seed_rows)
            self.protected = int(protected)

        def commit(self, row):
            self.rows.append(np.ascontiguousarray(row, dtype=np.float32))
            if len(self.rows) > POOL_CAPACITY:
                del self.rows[self.protected]

        def stack(self):
            return np.ascontiguousarray(np.stack(list(self.rows)), dtype=np.float32)

    def identity_for(student, id_encoder, stack, side, gated):
        with torch.inference_mode():
            support = torch.from_numpy(stack).unsqueeze(0)
            if gated:
                return id_encoder.forward_batch(support, side_features=side)
            return student.compute_identity(support, side_features=side)

    def windows_window(stream, t):
        lo = max(0, t - 49)
        window = np.zeros((50, stream.shape[1]), dtype=np.float32)
        take = stream[max(0, t - 49): t + 1]
        window[50 - take.shape[0]:] = take
        return window

    # -- data
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_heldout_dataset
    data_module_for_surface = lambda surface: (
        val_dataset if surface == "external_official_query" else train_dataset)

    # -- manual decode reference (payload decoder, proven bitwise to the pipeline)
    with (PKG / "artifacts/t4_m2_seed42_cenat_chunk100e_tta.pkl").open("rb") as handle:
        import pickle
        payload = pickle.load(handle)
    # rebuild Ce-NAT student pieces from its checkpoint for gated identity
    cenat_decoder = cenat.student.decoder.cpu().eval()
    cenat_encoder_state = None
    # Ce-NAT has a plain native encoder; export its state for PayloadIdEncoder
    cenat_encoder_state = copy.deepcopy(cenat.student.id_encoder.state_dict())
    cenat_id_encoder = PayloadIdEncoder()
    cenat_id_encoder.load_state_dict(cenat_encoder_state, strict=True)
    cenat_id_encoder.eval()

    payload_decoder = payload["decoder"]
    payload_encoder = PayloadIdEncoder()
    payload_encoder.load_state_dict(payload["id_encoder_state"], strict=True)
    payload_encoder.eval()

    def student_modules(arm):
        if arm.startswith("S0") or arm == "LP-cross":
            return payload_decoder, payload_encoder, False
        return cenat_decoder, cenat_id_encoder, False

    def gated_modules():
        return cenat_decoder, cenat_id_encoder, True

    surfaces = (("external_official_query", val_dataset),
                ("within_post30", train_dataset))
    episodes = []
    for surface_name, dataset in surfaces:
        sessions = sorted(dataset.calib_trialized_neural_features)
        for session in sessions:
            partner = cross_partner(session, sessions)
            if partner:
                episodes.append((surface_name, dataset, session, partner))
    require(len(episodes) == 10, f"episode coverage drift: {len(episodes)}")

    # -- decode cache to avoid recomputing static arms per surface
    results = {"schema": "m2_camr_diag_v1", "episodes": []}
    anchor_failures = []
    for surface_name, dataset, query_session, seed_session in episodes:
        query_mat = session_material(dataset, query_session)
        seed_mat = session_material(dataset, seed_session)
        entry = {"surface": surface_name, "query": query_session, "seed": seed_session,
                 "arms": {}}

        # S0-same anchor: pretrained, same-record seed, static (sealed parity)
        starts = windows_targets(dataset, query_session, surface_name)
        windows = np.ascontiguousarray(
            query_mat["neural"][starts[:, None] + np.arange(50)[None, :]], dtype=np.float32)
        targets = np.ascontiguousarray(query_mat["targets"][starts + 49], dtype=np.float32)
        s0_same_seed = session_material(dataset, query_session)
        with torch.inference_mode():
            support = torch.from_numpy(np.stack(s0_same_seed["seed_rows"])).unsqueeze(0)
            identity = payload_encoder.forward_batch(support, side_features=s0_same_seed["side"])
            preds = manual_decode(payload_decoder, torch.from_numpy(windows), identity)
        r2 = float(variance_weighted_r2(targets, preds[:, -1, :].numpy() / 5.0))
        delta = abs(r2 - sealed_rows[(surface_name, query_session)])
        if delta > R2_ABS_TOLERANCE:
            anchor_failures.append((surface_name, query_session, delta))
        entry["arms"]["S0_same"] = {"r2": r2, "sealed": sealed_rows[(surface_name, query_session)],
                                    "delta": delta}

        # static arms: identity computed once, single decode over windows
        def late_pooled_identity(enc, seed_material):
            with torch.inference_mode():
                support = torch.from_numpy(np.stack(seed_material["seed_rows"])).unsqueeze(0)
                x = support.permute(0, 1, 3, 2)  # [B,M,N,T]
                feats = enc.pre_pool(x)
                side_b = seed_material["side"].unsqueeze(1).expand(-1, feats.shape[1], -1, -1)
                vals = enc.post_pool(torch.cat((feats, side_b), dim=-1))
                total = vals[:, 0]
                for i in range(1, int(vals.shape[1])):
                    total = total + vals[:, i]
                return total / int(vals.shape[1])

        def static_decode(dec, enc, seed_material, identity_override=None):
            with torch.inference_mode():
                identity = identity_override if identity_override is not None else enc.forward_batch(
                    torch.from_numpy(np.stack(seed_material["seed_rows"])).unsqueeze(0),
                    side_features=seed_material["side"])
                preds = manual_decode(dec, torch.from_numpy(windows), identity)
            return float(variance_weighted_r2(targets, preds[:, -1, :].numpy() / 5.0))

        entry["arms"]["S0_cross"] = {"r2": static_decode(payload_decoder, payload_encoder, seed_mat)}
        entry["arms"]["S1_same"] = {"r2": static_decode(
            cenat_decoder, cenat_id_encoder, session_material(dataset, query_session))}
        entry["arms"]["S1_cross"] = {"r2": static_decode(cenat_decoder, cenat_id_encoder, seed_mat)}
        entry["arms"]["LP_cross"] = {"r2": static_decode(
            payload_decoder, payload_encoder, seed_mat,
            identity_override=late_pooled_identity(payload_encoder, seed_mat))}

        # continual arms: full-stream decode with per-law gates
        def continual_decode(law):
            neural = query_mat["neural"]
            real_stream = np.ascontiguousarray(neural[PREHISTORY_ZEROS:], dtype=np.float32)
            n_bins = int(real_stream.shape[0])
            pool = Pool(seed_mat["seed_rows"], PROTECTED_PREFIX)
            with torch.inference_mode():
                identity = cenat_id_encoder.forward_batch(
                    torch.from_numpy(pool.stack()).unsqueeze(0), side_features=query_mat["side"])
            candidate = np.zeros((PREHISTORY_ZEROS, neural.shape[1]), dtype=np.float32)
            rate_history = list(seed_mat["calib_energies"]) if law == "G1" else []
            warmup_left = WARMUP_K if law == "G2" else 0
            commits: list[int] = []
            predictions = np.zeros((n_bins, 2), dtype=np.float32)
            for t in range(n_bins):
                with torch.inference_mode():
                    out = manual_decode(cenat_decoder, torch.from_numpy(
                        windows_window(real_stream, t)), identity)
                predictions[t] = out[0, -1, :].astype(np.float32) / 5.0
                candidate = np.concatenate(
                    [candidate, real_stream[t].reshape(1, -1)], axis=0)
                if candidate.shape[0] == CHUNK_BINS:
                    rate = float(candidate.mean(dtype=np.float64))
                    accepted = False
                    if law == "G0":
                        accepted = (len(rate_history) == 0) or (
                            rate >= float(np.median(np.asarray(rate_history))))
                        rate_history.append(rate)
                    elif law == "G1":
                        accepted = rate >= float(np.median(np.asarray(rate_history)))
                        rate_history.append(rate)
                    elif law == "G2":
                        if warmup_left > 0:
                            warmup_left -= 1
                            rate_history.append(rate)
                        else:
                            accepted = rate >= float(np.median(np.asarray(rate_history)))
                            rate_history.append(rate)
                    if accepted:
                        pool.commit(candidate)
                        commits.append(t + PREHISTORY_ZEROS)
                        with torch.inference_mode():
                            identity = cenat_id_encoder.forward_batch(
                                torch.from_numpy(pool.stack()).unsqueeze(0),
                                side_features=query_mat["side"])
                    candidate = np.zeros((0, candidate.shape[1]), dtype=np.float32)
            counts = (np.searchsorted(np.asarray(commits, dtype=np.int64), endpoints, side="left")
                      if commits else np.zeros(len(endpoints), dtype=np.int64))
            bands = {}
            for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 11), (11, 10 ** 9)):
                mask = (counts >= lo) & (counts < hi)
                if int(mask.sum()) >= 8:
                    bands[f"commits_{lo}-{hi}"] = float(
                        variance_weighted_r2(targets[mask], predictions[starts][mask]))
            return {"r2": float(variance_weighted_r2(targets, predictions[starts])),
                    "commits": len(commits), "commit_bins": commits, "bands": bands}

        for arm, law in (("S2_cross_G0", "G0"), ("S3_cross_G1", "G1"), ("S4_cross_G2", "G2")):
            entry["arms"][arm] = continual_decode(law)

        results["episodes"].append(entry)
        print(json.dumps({"episode": f"{seed_session}->{query_session}",
                          **{k: round(v.get("r2", float("nan")), 4) for k, v in entry["arms"].items()}}),
              flush=True)

    require(not anchor_failures, f"sealed anchor drift: {anchor_failures}")
    results["anchor_failures"] = anchor_failures
    results["wall_seconds"] = time.monotonic() - started

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(results, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target = RESULT_ROOT / "diag.json"
    target.write_bytes(body)
    target.chmod(0o444)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(hashlib.sha256(body).hexdigest() + "  diag.json\n", encoding="ascii")
    sidecar.chmod(0o444)
    print(json.dumps({"status": "DONE", "receipt": str(target),
                      "anchor_failures": anchor_failures}, sort_keys=True))


if __name__ == "__main__":
    main()
