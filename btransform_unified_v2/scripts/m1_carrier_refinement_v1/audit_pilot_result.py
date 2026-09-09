"""Replay and bind a completed source-selected M1 refinement result on CPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_pilot as p
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=p.m1_data.FOLDS)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--b-dest", type=Path, required=True)
    a = parser.parse_args()
    a.dest = a.dest.resolve()
    a.b_dest = a.b_dest.resolve()
    a.seed, a.epochs, a.threads, a.train_stride = 42, 20, 4, 8
    p.configure_cpu(a.threads)
    bound = {}

    def bind(path):
        digest = p.sha_file(path)
        bound[str(path)] = digest
        return digest

    def read(path):
        bind(path)
        return json.loads(path.read_text())

    receipt = read(a.dest / "fit_score_receipt.json")
    selection = read(a.dest / "sealed_selection.json")
    preflight = read(a.dest / "preflight.json")
    protocol = read(a.dest / "protocol.json")
    require(receipt["status"] == "COMPLETED" and selection["status"] == "SEALED" and preflight["status"] == "PASSED", "incomplete pilot")
    require(protocol == p.protocol(a), "protocol mismatch")
    for key, filename in (("selection_sha256", "sealed_selection.json"), ("source_cache_sha256", "source_selected_b_cache.npz"), ("residual_checkpoint_sha256", "residual_head.pt"), ("source_carrier_sha256", "frozen_source_carriers.npz")):
        require(receipt[key] == bind(a.dest / filename), f"receipt binding mismatch: {key}")
    require(receipt["target_optimizer_steps"] == 0 and receipt["target_labels_used_for_fit_or_selection"] is False, "target selection contract")
    model, src, b_train, _ = p.load_base(a, include_target=False)
    require(preflight["binding"] == p.binding(a), "preflight code/checkpoint/runtime binding mismatch")
    sessions = tuple(src["sources"])
    base_hash = p.model_hash(model)
    require(base_hash == preflight["base_hash"] == selection["base_hash"] == receipt["base_hash"], "base state mismatch")
    for key, value in preflight["binding"]["runtime_sources"].items():
        require(bind(Path(value["file"])) == value["sha256"], f"runtime source mismatch: {key}")
    for filename, digest in b_train["checkpoint_sha256"].items():
        require(bind(a.b_dest / filename) == digest, "B checkpoint mismatch")

    cache_meta = read(a.dest / "source_selected_b_cache.json")
    require(cache_meta["binding"] == preflight["binding"], "source cache binding mismatch")
    require(cache_meta["npz_sha256"] == bind(a.dest / "source_selected_b_cache.npz"), "source cache checksum")
    require(cache_meta["source_evidence"] == p.m1_train.source_evidence(src), "source cache raw data mismatch")
    train_rows, val_rows = p.load_source_cache(a.dest / "source_selected_b_cache.npz", sessions)
    with np.load(a.dest / "frozen_source_carriers.npz", allow_pickle=False) as z:
        frozen_arrays = {k: z[k].copy() for k in z.files}
    require(selection["frozen_source_carriers_sha256"] == bind(a.dest / "frozen_source_carriers.npz"), "selection carrier binding")
    carriers = {s: frozen_arrays[f"carrier/{s}"] for s in sessions}
    grids = {split: p.dataset_features(src[split], carriers) for split in ("train", "val")}
    subset, _ = p.stride_subset(src["train"], a.train_stride)
    original_indices = {}
    for i in subset.indices:
        s, _ = src["train"].window_indices[i]
        original_indices.setdefault(s, []).append(i)
    p.cache_identity(model, sessions)
    counts, replay_errors = {}, {}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        ds = src[split]
        for s in sessions:
            row = rows[s]
            reference_windows = subset.window_indices if split == "train" else ds.window_indices
            expected = np.asarray([int(st) for name, st in reference_windows if name == s], dtype=np.int64)
            require(np.array_equal(expected, row["start"]), f"source coordinates: {split}/{s}")
            require(cache_meta["feature_grid_sha256"][split][s] == p.sha_array(grids[split][s]), "source feature grid checksum")
            endpoint = row["start"] + 99
            require(np.array_equal(row["feature"], grids[split][s][endpoint]), "source feature row mismatch")
            require(np.array_equal(row["target"], ds.covariate_data[s][endpoint]), "source target row mismatch")
            expected_valid = np.arange(100)[None, :] >= np.maximum(0, 99 - row["start"])[:, None]
            require(np.array_equal(row["valid"], expected_valid), "source validity mask mismatch")
            if split == "train":
                require(np.array_equal(row["original_index"], original_indices[s]), "source original index mismatch")
            x = np.stack([ds.neural_data[s][st:st + 100] for st in row["start"][:32]])
            with torch.inference_mode():
                prediction = model(torch.from_numpy(x), src["banks"][s], input_valid_mask=torch.from_numpy(row["valid"][:32])).numpy()
            error = float(np.max(np.abs(prediction - row["prediction"][:32])))
            require(error <= 2e-6, "representative B source replay mismatch")
            replay_errors[f"{split}/{s}"] = error
            counts[f"{split}/{s}"] = len(expected)

    checkpoint = torch.load(a.dest / "residual_head.pt", map_location="cpu", weights_only=False)
    require(checkpoint["selection_sha256"] == bind(a.dest / "sealed_selection.json") and checkpoint["base_hash"] == base_hash, "selected head provenance")
    wrapper = p.FrozenBCarrierResidual(model, feature_dim=24)
    wrapper.residual_head.load_state_dict(checkpoint["residual_head"], strict=True)
    # Independent weighted-moment formula, in float64, rather than invoking
    # the runner's statistics helper a second time.
    feature_arrays = [train_rows[s]["feature"].astype(np.float64) for s in sorted(sessions)]
    independent_mean = np.mean([v.mean(axis=0) for v in feature_arrays], axis=0)
    independent_second = np.mean([np.square(v).mean(axis=0) for v in feature_arrays], axis=0)
    independent_scale = np.maximum(np.sqrt(np.maximum(independent_second - independent_mean ** 2, 0.)), 1e-6)
    require(np.allclose(independent_mean, wrapper.residual_head.carrier_mean.numpy(), rtol=2e-5, atol=2e-7) and np.allclose(independent_scale, wrapper.residual_head.carrier_scale.numpy(), rtol=2e-5, atol=2e-7), "independent source-only feature moments mismatch")
    baseline = {s: float(p.variance_weighted_r2(val_rows[s]["target"], val_rows[s]["prediction"])) for s in sessions}
    require(baseline == selection["baseline"] == receipt["source_validation_cpu_baseline_recomputed"], "source baseline mismatch")
    curve = selection["curve"]
    require(len(curve) == 84 and curve == receipt["source_curve"], "full source selection curve missing")
    for index, row in enumerate(curve):
        require(row["epoch"] == index // 4 and row["alpha"] == (0., .25, .5, 1.)[index % 4], "source curve ordering")
        values = row["per_session"]
        require(set(values) == set(sessions) and np.isfinite(list(values.values())).all(), "source curve roster/finite")
        require(abs(np.mean(list(values.values())) - row["equal_session_mean"]) <= 1e-12, "source curve mean")
        require(row["eligible"] == all(values[s] >= baseline[s] - 1e-6 for s in sessions), "eligibility rule")
    best = max((row for row in curve if row["eligible"]), key=lambda row: row["equal_session_mean"])
    require(best == selection["selection"] == receipt["source_selected"], "earliest source-only selection mismatch")
    _, replay_val = p.evaluate(wrapper, val_rows, best["alpha"])
    require(all(abs(replay_val[s] - best["per_session"][s]) <= 1e-12 for s in sessions), "selected head full-source replay")
    print(json.dumps({"stage": "source_audit", "status": "PASSED", "counts": counts, "selected": best}), flush=True)

    strict = read(a.b_dest / "score_receipt.json")
    require(receipt["strict_b_score_receipt_sha256"] == bind(a.b_dest / "score_receipt.json"), "strict B score receipt binding")
    baseline_path = a.b_dest / strict["target_audit_arrays"]["selected"]["file"]
    require(bind(baseline_path) == strict["target_audit_arrays"]["selected"]["sha256"] == receipt["target_npz_sha256"], "strict B target checksum")
    with np.load(baseline_path, allow_pickle=False) as z:
        original = {k: z[k].copy() for k in z.files}
    require(receipt["target_coordinate_sha256"] == p.sha_array(original["window_start_padded"]), "target coordinate binding")
    candidate_path = a.dest / "target_candidate_predictions.npz"
    bind(candidate_path)
    with np.load(candidate_path, allow_pickle=False) as z:
        candidate = {k: z[k].copy() for k in z.files}
    for key in ("target", "window_start_padded", "output_index_padded", "output_index_query_relative", "prefix_bins"):
        require(candidate[key].dtype == original[key].dtype and candidate[key].tobytes() == original[key].tobytes(), f"target bytes mismatch: {key}")
    require(np.array_equal(candidate["b_prediction"], original["prediction"]), "B target prediction changed")
    target_ds = p.m1_score._target_dataset(a.target)
    starts = np.asarray([st for s, st in target_ds.window_indices], dtype=np.int64)
    endpoint = starts + 99
    require(np.array_equal(starts, original["window_start_padded"]) and np.array_equal(target_ds.covariate_data[a.target][endpoint], original["target"]), "physical target query mismatch")
    support = p.aligned_carrier.load_aligned_support(p.m1_data.nwb_path(a.target), emg_trial_stop=10)
    scores = syn3.nnls_activations(syn3.apply_scale(np.maximum(support.emg, 0.), frozen_arrays["basis_scale"]), frozen_arrays["basis_dictionary"])
    weights, intercept = syn3.fit_all_units(scores, support.rates)
    raw = syn3.carrier_from_encoding(weights, intercept)
    carrier = np.asarray(syn3.normalize_carriers(raw, frozen_arrays["normalizer_mean"], frozen_arrays["normalizer_scale"]), dtype=np.float32)
    require(p.aligned_carrier._array_sha256(carrier) == receipt["target"]["carrier"]["carrier_sha256"], "target M10 carrier replay")
    feature_grid = p.aligned_carrier.causal_carrier_projection(target_ds.neural_data[a.target], carrier)
    require(np.array_equal(feature_grid[endpoint], candidate["feature"]), "target causal features replay")
    wrapper.eval()
    with torch.inference_mode():
        b = torch.from_numpy(original["prediction"])
        f = torch.from_numpy(candidate["feature"])
        residual = wrapper.forward_cached(b, f, carrier_visible=torch.ones(len(b), dtype=torch.bool)) - b
        replay = (b + best["alpha"] * residual).numpy()
        fallback = wrapper.forward_cached(b, f, carrier_visible=torch.zeros(len(b), dtype=torch.bool)).numpy()
    require(np.array_equal(replay, candidate["prediction"]) and np.array_equal(fallback, original["prediction"]), "target prediction/fallback replay")
    b_r2 = float(p.variance_weighted_r2(original["target"], original["prediction"]))
    new_r2 = float(p.variance_weighted_r2(candidate["target"], candidate["prediction"]))
    require(b_r2 == receipt["target"]["b_r2"] and new_r2 == receipt["target"]["candidate_r2"], "target R2 replay")
    require(p.model_hash(model) == base_hash and all(q.grad is None for q in model.parameters()), "audit mutated B")
    bind(Path(__file__).resolve())
    audit = {"status": "PASSED", "schema": "m1_refinement_result_replay_v1", "fold": a.target, "bound_sha256": bound, "counts": counts, "source_B_representative_replay_max_abs": replay_errors, "source_B_network_replay_scope": "First 32 cached rows of each source train/validation pair; remaining B rows are bound to the observed completed inference cache by checksum and full input-coordinate verification.", "independent_equal_session_feature_moments": True, "full_source_selected_head_replay": replay_val, "source_selection": best, "target_physical_query_and_causal_feature_replay": True, "target_prediction_and_zero_visibility_fallback_byte_replay": True, "base_hash_unchanged": True, "target": {"B": b_r2, "candidate": new_r2, "delta": new_r2 - b_r2}, "scope": "Exploratory public LOSO result; no hidden evaluation or target-based tuning."}
    p.atomic_json(a.dest / "result_replay_audit.json", audit)
    print(json.dumps({"status": "PASSED", "target": audit["target"], "audit_sha256": p.sha_file(a.dest / "result_replay_audit.json")}), flush=True)


if __name__ == "__main__":
    main()
