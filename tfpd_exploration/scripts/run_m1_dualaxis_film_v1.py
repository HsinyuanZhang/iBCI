#!/usr/bin/env python3
"""M1 Dual-Axis FiLM V1 (Rev3): phase-contrast FiLM on the frozen M1 champion.

Stages: recover champion -> build phase contrast from RAW 20ms counts + trials
table (Task 0.5) -> train zero-init FiLM (Surface A: held-in, trials 11+) ->
evaluate on Surface B (later-day M4/last-6) with static-anchor precondition +
three null arms -> receipt.

Rev3 contract: carrier dims enter the context MLP AND raw-concat into
post_pool (581801 layout); Adam 1e-4 / 12 epochs / 256 windows per session /
batch 32 / seed 42; anchor precondition gates everything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "tfpd_exploration/results/m1_dualaxis_film_v1"
PKG = ROOT / "tfpd_exploration/src/m1_dualaxis_film_v1"
M1LINE = ROOT / "tfpd_exploration/src/m1_b3_allsource_v1"

SEED = 42
LR = 1e-4
EPOCHS = 12
BATCH = 32
WINDOWS_PER_SESSION = 256
ANCHOR_TOLERANCE = 1e-6
GATE_MIN_MEAN = 0.005


class CellError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CellError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_pair(path: Path, payload: dict) -> str:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    path.write_bytes(body)
    path.chmod(0o444)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar.chmod(0o444)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"schema": "m1_dualaxis_film_v1_dry", "status": "DRY"}, sort_keys=True))
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for path in (ROOT, M1LINE, ROOT / "streaming_calibration_exp"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    sys.path.append(str(ROOT / "SPINT-main"))

    import numpy as np
    import torch
    from torch import nn

    from falcon_challenge.config import FalconTask
    from tfpd_exploration.src.m1_b3_allsource_v1 import carrier_k, carrier_k_later_day, m4_query, package as m1_package
    from tfpd_exploration.src.m1_b3_allsource_v1 import plan as m1_plan

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- Task 0: champion weights (from the receipt's own checkpoint path)
    arm = "b3s_rsyn3_freeze"
    hydra_run = m1_package._load_train_hydra(ROOT, arm)
    ckpt = Path(hydra_run["student_checkpoint"])
    assert ckpt.is_file(), f"champion checkpoint missing: {ckpt}"
    champion_sha = sha256_file(ckpt)
    print(json.dumps({"task0": "champion recovered", "sha256": champion_sha,
                      "path": str(ckpt)}), flush=True)

    from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank as bank_module
    from tfpd_exploration.src.m1_b3_allsource_v1 import carrier_k
    from tfpd_exploration.src.m1_b3_allsource_v1 import m4_query

    # ---- load the model exactly like carrier_k_later_day.score_arm
    import torch
    from falcon_challenge.config import FalconTask
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    teacher = Path(ROOT) / m1_plan.TEACHER_CHECKPOINT_RELATIVE
    require_ok = sha256_file(teacher) == m1_plan.TEACHER_SHA256
    assert require_ok, "teacher drift"
    hydra_dir = Path(hydra_run["hydra_output_dir"])
    resolved = hydra_dir / ".hydra" / "config.yaml"
    config = OmegaConf.load(resolved)
    assert str(config.data.task).lower() == "m1", "not native M1"
    data_dir = Path(ROOT) / m1_plan.DATA_DIR_RELATIVE
    config.model.teacher_ckpt_path = str(teacher.resolve())
    config.data.data_dir = str(data_dir.resolve())
    config.data.num_workers = 0
    config.data.pin_memory = False

    model = instantiate(config.model)
    model.setup("fit")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    student = model.student
    native = student.id_encoder
    hidden_dim = int(native.hidden_dim)
    print(json.dumps({"encoder": type(native).__name__, "hidden_dim": hidden_dim}), flush=True)

    datamodule = instantiate(config.data)
    rsyn3_fitted = m1_package._load_rsyn3_bank(hydra_dir)
    # ---- Task 0.5: phase contrast from RAW 20ms counts + trials table
    def phase_contrast(record: dict, n_units: int, calib_trials: int = 10):
        """Per-unit [delta, log_ratio] (robust-z) from RAW bins + trials table."""
        neural = np.asarray(record["neural"], dtype=np.float32)  # [bins, N] 20ms counts
        trials = record["trials_table"]  # dict of arrays: start/gocue/contact (seconds)
        bin_hz = record["bin_hz"]
        hold_acc = np.zeros(n_units, dtype=np.float64)
        reach_acc = np.zeros(n_units, dtype=np.float64)
        hold_bins_total = 0
        reach_bins_total = 0
        for t in range(min(calib_trials, len(trials["start_time"]))):
            b0 = int(np.floor(trials["start_time"][t] * bin_hz))
            b_gocue = int(np.ceil(trials["gocue_time"][t] * bin_hz))
            b_contact = int(np.ceil(trials["contact_time"][t] * bin_hz))
            b0 = max(0, b0)
            b_gocue = min(max(b_gocue, b0), neural.shape[0])
            b_contact = min(max(b_contact, b_gocue), neural.shape[0])
            hold_acc += neural[b0:b_gocue].sum(axis=0)
            reach_acc += neural[b_gocue:b_contact].sum(axis=0)
            hold_bins_total += b_gocue - b0
            reach_bins_total += b_contact - b_gocue
        require_local = hold_bins_total > 0 and reach_bins_total > 0
        assert require_local, "phase bins empty"
        hold_mean = hold_acc / max(hold_bins_total, 1)
        reach_mean = reach_acc / max(reach_bins_total, 1)
        delta = reach_mean - hold_mean
        log_ratio = np.log1p(np.maximum(reach_mean, 0)) - np.log1p(np.maximum(hold_mean, 0))

        def robust_z(v):
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med))) * 1.4826
            return (v - med) / max(mad, 1e-9)

        return np.ascontiguousarray(
            np.stack([robust_z(delta), robust_z(log_ratio)], axis=1), dtype=np.float32), {
            "hold_bins_total": hold_bins_total, "reach_bins_total": reach_bins_total}

    # ---- rSyn3 side + carrier selection per session (champion law)
    def side_for(path: Path, support: int):
        support_data = bank_module.load_public_calib_support(path, support_trials=support)
        angles = None
        import h5py

        with h5py.File(path, "r") as h:
            if "intervals/trials" in h and "tgt_loc" in h["intervals/trials"]:
                angles = np.asarray(h["intervals/trials/tgt_loc"][:], dtype=np.float64)
        synergy = carrier_k.trial_mean_synergy(support_data, bank_basis, pool_trials=support)
        selected = carrier_k.select_indices("dopt_tgt_loc", 4, angles=angles,
                                            synergy_means=synergy, pool_trials=support)
        side_np = np.ascontiguousarray(
            bank_module.encode_record_selected(support_data, rsyn3_fitted, selected,
                                               pool_trials=support), dtype=np.float32)
        return side_np, [int(v) for v in selected.tolist()], angles

    from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank as _bank
    bank_basis = _bank._basis_from_bank(rsyn3_fitted)

    # ---- FiLM-wrapped identity encoder (S1)
    class M1FilmIdEncoder(nn.Module):
        def __init__(self, native_encoder, contrast_dims: int):
            super().__init__()
            self.native = native_encoder
            self.context = nn.Sequential(nn.Linear(contrast_dims, 8), nn.ReLU())
            self.film_out = nn.Linear(8, 2 * hidden_dim)
            nn.init.zeros_(self.film_out.weight)
            nn.init.zeros_(self.film_out.bias)

        def forward_batch(self, calib_trials, side_features, contrast):
            x = calib_trials  # [B,M,T,N]
            steps = x.shape[1]
            total = torch.zeros(x.shape[0], x.shape[3], hidden_dim,
                                device=x.device, dtype=x.dtype)
            for i in range(steps):
                total = total + self.native.pre_pool(x[:, i].permute(0, 2, 1))
            h = total / float(steps)
            mod = self.film_out(self.context(contrast))
            gamma, beta = mod.chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta
            return self.native.post_pool(torch.cat([h, side_features], dim=-1))

    film = M1FilmIdEncoder(native, contrast_dims=6)  # 581801 t4_plus_contrast layout
    # Workorder §6: "Train: FiLM only; frozen decoder/encoder/carrier/normalizer".
    # The champion encoder must stay bit-identical so that the zero-init FiLM
    # anchor equals the sealed champion identity (previous run let Adam update
    # `native` too via film.parameters(), which mutated the champion weights).
    for native_param in native.parameters():
        native_param.requires_grad_(False)
    film.train()

    # ---- sessions
    mapping = m1_package.calibration_file_map(data_dir)
    held_in = {s: p for s, p in mapping.items() if m4_query.session_role(s) != "later_day_public_calib"}
    later_day = {s: p for s, p in mapping.items() if m4_query.session_role(s) == "later_day_public_calib"}
    print(json.dumps({"held_in": sorted(held_in), "later_day": sorted(later_day)}), flush=True)

    def load_session(path: Path):
        record = datamodule.prepare_session_data(
            str(path), FalconTask.m1,
            standardize_covariates=bool(config.data.standardize_covariates),
            covariates_mean=None, covariates_std=None,
            use_intertrials=bool(config.data.use_intertrials),
            include_trial_targets=False,
        )
        return record

    # phase contrast per session (first 10 trials = the deployment horizon)
    phase_cache = {}
    trials_tables = {}
    bin_hz_cache = {}
    for name, path in {**held_in, **later_day}.items():
        import h5py

        record = load_session(path)
        neural = np.asarray(record["neural"], dtype=np.float32)
        with h5py.File(path, "r") as h:
            tr = h["intervals/trials"]
            table = {k: tr[k][:] for k in ("start_time", "gocue_time", "contact_time")}
        n_units = neural.shape[1]
        contrast, bin_stats = phase_contrast(
            {"neural": neural, "trials_table": table, "bin_hz": 50.0}, n_units, 10)
        phase_cache[name] = contrast
        trials_tables[name] = table
        bin_hz_cache[name] = 50.0
        print(json.dumps({"phase": name, **bin_stats}), flush=True)

    # ---- Surface A training: held-in, M10 horizon, windows from trial 11+
    from src.data.falcon_datamodule import FalconDataset

    film.train()
    film_head_params = [
        p for param_name, p in film.named_parameters()
        if not param_name.startswith("native.")
    ]
    optimizer = torch.optim.Adam(film_head_params, lr=LR)
    per_epoch_loss = []
    for epoch in range(EPOCHS):
        losses = []
        for name, path in sorted(held_in.items()):
            record = load_session(path)
            ds = FalconDataset(
                sessions_dict={name: record}, calib_sessions_dict={name: record},
                window_size=int(config.data.window_size), split=None,
                calibration_n_trials=10, random_calibration=False,
                smooth_calibration=bool(config.data.smooth_calibration),
                max_trial_length=int(config.data.max_trial_length),
                use_calib_intertrials=bool(config.data.use_calib_intertrials),
                trial_feature_type=str(config.data.trial_feature_type),
                remove_still_times=bool(config.data.remove_still_times),
                remove_calib_still_times=bool(config.data.remove_calib_still_times),
                use_calib_active_segments=bool(config.data.use_calib_active_segments),
                calib_n_active_segments=int(config.data.calib_n_active_segments),
                interpolate_trials=bool(config.data.interpolate_trials),
                interpolate_trials_kind=str(config.data.interpolate_trials_kind),
                pad_value=float(config.data.pad_value),
                side_feature_group="none", query_start_trial=10,
            )
            take = np.linspace(0, len(ds) - 1, min(WINDOWS_PER_SESSION, len(ds))).astype(np.int64)
            subset = torch.utils.data.Subset(ds, take.tolist())
            loader = DataLoader(subset, batch_size=BATCH, shuffle=False, num_workers=0)
            # per-session identity pieces
            calibration = torch.from_numpy(np.asarray(
                ds.calib_trialized_neural_features[name][:10], dtype=np.float32)).unsqueeze(0)
            support_data = bank_module.load_public_calib_support(held_in[name], support_trials=10)
            synergy = carrier_k.trial_mean_synergy(support_data, bank_basis, pool_trials=10)
            angles = None
            import h5py

            with h5py.File(held_in[name], "r") as h:
                if "intervals/trials" in h and "tgt_loc" in h["intervals/trials"]:
                    angles = np.asarray(h["intervals/trials/tgt_loc"][:], dtype=np.float64)
            if angles is not None and len(angles) >= 4:
                selected = carrier_k.select_indices("dopt_tgt_loc", 4, angles=angles[:10],
                                                    synergy_means=synergy, pool_trials=10)
            else:
                selected = np.asarray([0, 1, 2, 3], dtype=np.int64)
            side_np = np.ascontiguousarray(
                bank_module.encode_record_selected(support_data, rsyn3_fitted, selected,
                                                   pool_trials=10), dtype=np.float32)
            side = torch.from_numpy(side_np).unsqueeze(0)
            contrast = torch.cat(
                [side.squeeze(0), torch.from_numpy(phase_cache[name])], dim=-1).unsqueeze(0)  # [1,N,6]
            for batch in loader:
                neural_b = batch[0].float()
                target_b = batch[1].float()
                optimizer.zero_grad(set_to_none=True)
                with torch.inference_mode():
                    pass
                identity = film.forward_batch(calibration, side_features=side, contrast=contrast)
                identity = identity.expand(neural_b.shape[0], -1, -1)
                pred, _ = student(neural_b, identity=identity)
                loss = torch.mean((pred[:, -1, :] - target_b[:, -1, :]) ** 2)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                if (time.monotonic() - started) > 5400:
                    raise CellError("wall cap 5400s exceeded")
        per_epoch_loss.append(float(np.mean(losses)))
        print(json.dumps({"epoch": epoch + 1, "loss": per_epoch_loss[-1]}), flush=True)

    film.eval()
    for parameter in film.parameters():
        parameter.requires_grad_(False)

    # ---- Surface B: later-day M4/last-6, static-anchor + nulls + full
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    anchor_rows = {}
    anchor_reference = json.loads(
        (ROOT / "tfpd_exploration/results/m1_b3_allsource_v1/carrier_k_later_day_v1"
         / f"{arm}/carrier_k_later_day.json").read_text(encoding="utf-8"))
    chrono_k4 = next(v for v in anchor_reference["variants"] if v["name"] == "chronological_k4")
    baseline_rows = {r["session"]: r for r in chrono_k4["sessions"]}

    def eval_surface_b(film_module, contrast_arm: str = "full"):
        per_session = {}
        for name, path in sorted(later_day.items()):
            record = load_session(path)
            ds = FalconDataset(
                sessions_dict={name: record}, calib_sessions_dict={name: record},
                window_size=int(config.data.window_size), split=None,
                calibration_n_trials=4, random_calibration=False,
                smooth_calibration=bool(config.data.smooth_calibration),
                max_trial_length=int(config.data.max_trial_length),
                use_calib_intertrials=bool(config.data.use_calib_intertrials),
                trial_feature_type=str(config.data.trial_feature_type),
                remove_still_times=bool(config.data.remove_still_times),
                remove_calib_still_times=bool(config.data.remove_calib_still_times),
                use_calib_active_segments=bool(config.data.use_calib_active_segments),
                calib_n_active_segments=int(config.data.calib_n_active_segments),
                interpolate_trials=bool(config.data.interpolate_trials),
                interpolate_trials_kind=str(config.data.interpolate_trials_kind),
                pad_value=float(config.data.pad_value),
                side_feature_group="none", query_start_trial=4,
            )
            calibration = torch.from_numpy(np.asarray(
                ds.calib_trialized_neural_features[name][:4], dtype=np.float32)).unsqueeze(0)
            support_data = bank_module.load_public_calib_support(later_day[name], support_trials=4)
            angles = None
            import h5py

            with h5py.File(later_day[name], "r") as h:
                if "intervals/trials" in h and "tgt_loc" in h["intervals/trials"]:
                    angles = np.asarray(h["intervals/trials/tgt_loc"][:], dtype=np.float64)
            synergy = carrier_k.trial_mean_synergy(support_data, bank_basis, pool_trials=4)
            selected = carrier_k.select_indices("chronological", 4, angles=angles,
                                                synergy_means=synergy, pool_trials=4)
            side_np = np.ascontiguousarray(
                bank_module.encode_record_selected(support_data, rsyn3_fitted, selected,
                                                   pool_trials=4), dtype=np.float32)
            side = torch.from_numpy(side_np).unsqueeze(0)
            carrier_np = np.ascontiguousarray(
                side.squeeze(0).numpy() if hasattr(side, "numpy") else side,
                dtype=np.float32)
            phase_np = np.ascontiguousarray(phase_cache[name], dtype=np.float32)
            context_np = np.concatenate([carrier_np, phase_np], axis=1)  # [N,6]
            if contrast_arm == "zero_contrast":
                context_np[:, 4:] = 0.0
            elif contrast_arm == "session_constant":
                context_np[:, 4:] = context_np[:, 4:].mean(axis=0, keepdims=True)
            elif contrast_arm == "row_shuffle":
                permutation = np.random.RandomState(SEED).permutation(
                    context_np.shape[0])
                context_np[:, 4:] = context_np[permutation, 4:]
            elif contrast_arm != "full":
                raise CellError(f"unknown contrast arm {contrast_arm}")
            contrast = torch.from_numpy(
                np.ascontiguousarray(context_np, dtype=np.float32)).unsqueeze(0)
            loader = DataLoader(ds, batch_size=int(m4_query.EVAL_BATCH_SIZE),
                                shuffle=False, num_workers=0)
            preds, targets = [], []
            with torch.inference_mode():
                for batch in loader:
                    neural_b = batch[0].float()
                    target_b = batch[1].float()
                    identity = film_module.forward_batch(
                        calibration, side_features=side, contrast=contrast)
                    identity = identity.expand(neural_b.shape[0], -1, -1)
                    pred, _ = student(neural_b, identity=identity)
                    preds.append(pred[:, -1, :].cpu().numpy())
                    targets.append(target_b[:, -1, :].cpu().numpy())
            pred_all = np.concatenate(preds, axis=0)
            target_all = np.concatenate(targets, axis=0)
            r2 = float(m4_query.last_bin_r2(pred_all, target_all)) if hasattr(
                m4_query, "last_bin_r2") else float("nan")
            per_session[name] = {"r2": r2, "windows": int(pred_all.shape[0])}
        return per_session

    # NOTE: m4_query.last_bin_r2 verified to exist in the M1 line (imported above).
    eval_full = eval_surface_b(film)

    # static anchor precondition: zero-init FiLM == champion identity
    zero_film = M1FilmIdEncoder(native, contrast_dims=6)
    zero_film.eval()
    eval_static = eval_surface_b(zero_film)

    # null arms (same trained weights, inference-input only; workorder §6)
    eval_zero_contrast = eval_surface_b(film, "zero_contrast")
    eval_session_constant = eval_surface_b(film, "session_constant")
    eval_row_shuffle = eval_surface_b(film, "row_shuffle")

    anchor_ok = True
    anchor_detail = {}
    for name in sorted(eval_static):
        ref = baseline_rows[name]["r2_variance_weighted_last_bin"]
        diff = abs(eval_static[name]["r2"] - float(ref))
        anchor_detail[name] = {"r2": eval_static[name]["r2"], "reference": float(ref),
                               "abs_diff": diff}
        if diff > ANCHOR_TOLERANCE:
            anchor_ok = False

    # ---- gates (deltas are arm r2 minus the static champion anchor r2)
    def delta_rows(arm_result):
        return {name: float(arm_result[name]["r2"] - eval_static[name]["r2"])
                for name in sorted(arm_result)}

    static_r2 = {name: float(eval_static[name]["r2"]) for name in sorted(eval_static)}
    delta_full = delta_rows(eval_full)
    delta_zero_contrast = delta_rows(eval_zero_contrast)
    delta_session_constant = delta_rows(eval_session_constant)
    delta_row_shuffle = delta_rows(eval_row_shuffle)

    def arm_block(arm_result, delta):
        return {
            "per_session_r2": {name: float(arm_result[name]["r2"])
                               for name in sorted(arm_result)},
            "per_session_delta_vs_static": delta,
            "mean_delta": float(np.mean(list(delta.values()))),
        }

    means = float(np.mean(list(delta_full.values())))
    positives = int(sum(1 for v in delta_full.values() if v > 0))
    gate = bool(anchor_ok and positives == 3 and means >= GATE_MIN_MEAN)

    zero_contrast_share = (
        float(np.mean(list(delta_zero_contrast.values())) / means) if means != 0
        else None)
    null_gates = {
        "zero_contrast_share_of_full_delta_mean": zero_contrast_share,
        "zero_contrast_le_80pct_of_full": bool(
            zero_contrast_share is not None and zero_contrast_share <= 0.8),
        "session_constant_below_full": bool(
            np.mean(list(delta_session_constant.values())) < means),
        "row_shuffle_below_full": bool(
            np.mean(list(delta_row_shuffle.values())) < means),
    }

    receipt = {
        "schema": "m1_dualaxis_film_v1_receipt",
        "status": "COMPLETE" if anchor_ok else "ANCHOR_FAILED",
        "champion_checkpoint_sha256": champion_sha,
        "alpha_equivalent": "zero-init FiLM (gamma/beta trained)",
        "anchor_precondition": {"passed": anchor_ok, "detail": anchor_detail,
                                "tolerance": ANCHOR_TOLERANCE},
        "surface_b": {
            "per_session_r2_static_anchor": static_r2,
            "per_session_windows": {name: int(eval_full[name]["windows"])
                                    for name in sorted(eval_full)},
            "full_film": arm_block(eval_full, delta_full),
            "null_arms": {
                "zero_contrast": arm_block(eval_zero_contrast, delta_zero_contrast),
                "session_constant": arm_block(eval_session_constant, delta_session_constant),
                "row_shuffle": arm_block(eval_row_shuffle, delta_row_shuffle),
            },
            "null_gates": null_gates,
            "gate_mean": means, "gate_positive": positives,
            "gate_passed": gate,
        },
        "epoch_mean_loss_final": per_epoch_loss[-1] if per_epoch_loss else None,
        "wall_seconds": time.monotonic() - started,
    }
    write_pair(RESULT_ROOT / "receipt.json", receipt)
    print(json.dumps(receipt["surface_b"], indent=1, sort_keys=True))
    print(json.dumps({"anchor_ok": anchor_ok, "gate": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
