"""Stage 0B: acoustic basis, SFC constructibility, TPL/DR closed-form baselines."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .acoustic_basis import fit_acoustic_basis
from .baselines import fit_dr_family, tpl_family
from .constants import FOLDS, RESULTS_ROOT, VALID_END, VALID_START
from .data import calib_trials, first_m3, load_all_files, next_m3, query_trials, alignment_table
from .metric import compare_official_bitwise, official_metric_from_trials, trials_to_time_major, valid_frame_mask_2d
from .sfc import (
    derange_trials,
    encoding_r2,
    fit_coefficient_normalizer,
    fit_sfc_on_trials,
    pad_carrier,
    select_fold_lag,
    split_half_report,
    standardize_coefficients,
    zero9,
)
from .util import sha256_array, write_json


def _json_fit(fit, extra=None):
    out = {
        "q": fit.q,
        "lag_ms": fit.lag_ms,
        "finite": fit.finite,
        "design_rank": fit.design_rank,
        "condition_number": fit.condition_number,
        "coefficient_norm": fit.coefficient_norm,
        "encoding_r2_mean": fit.encoding_r2_mean,
        "raw_sha256": sha256_array(fit.raw_vector),
        "padded_sha256": sha256_array(fit.padded),
    }
    if extra:
        out.update(extra)
    return out


def run_stage0b(out_dir: Path | None = None, inventory: dict | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else RESULTS_ROOT
    stage_dir = out_dir / "stage0b"
    stage_dir.mkdir(parents=True, exist_ok=True)
    load_all_files()
    input_sha = None if inventory is None else inventory.get("input_sha256s")
    fold_summaries = []
    source_split_half = {}

    # Official metric bitwise sentinel on a real trial
    probe = first_m3("20210626")[0]
    pred_stream, tgt_stream, mask = trials_to_time_major([probe.spectrogram], [probe.spectrogram])
    metric_parity = compare_official_bitwise(pred_stream, tgt_stream, mask)

    for spec in FOLDS:
        fold = spec["fold"]
        fold_dir = stage_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_dates = spec["train_dates"]
        val_date = spec["val_date"]
        train_trials = []
        for d in train_dates:
            train_trials.extend(calib_trials(d))
        basis = fit_acoustic_basis(train_trials, train_dates)
        basis_rec = {
            "train_dates": list(train_dates),
            "val_date_excluded": val_date not in train_dates,
            "n_frames": basis.n_frames,
            "mean_sha256": basis.mean_sha256,
            "std_sha256": basis.std_sha256,
            "basis_sha256": basis.basis_sha256,
            "sign_sha256": basis.sign_sha256,
            "singular_values": basis.singular_values.tolist(),
        }

        lag_rec = select_fold_lag(train_dates, first_m3, lambda d: calib_trials(d)[3:], basis, q=8)
        lag = lag_rec["selected_lag_ms"]

        # split-half on training dates only
        split_rows = {}
        train_fits4 = []
        train_fits9 = []
        for d in train_dates:
            f4 = fit_sfc_on_trials(first_m3(d), basis, 3, lag)
            f9 = fit_sfc_on_trials(first_m3(d), basis, 8, lag)
            n4 = fit_sfc_on_trials(next_m3(d), basis, 3, lag)
            n9 = fit_sfc_on_trials(next_m3(d), basis, 8, lag)
            train_fits4.append(f4)
            train_fits9.append(f9)
            split_rows[d] = {
                "sfc4": split_half_report(f4, n4),
                "sfc9": split_half_report(f9, n9),
            }
            source_split_half.setdefault(d, []).append(split_rows[d]["sfc9"])

        mean4, std4 = fit_coefficient_normalizer(train_fits4)
        mean9, std9 = fit_coefficient_normalizer(train_fits9)

        val_m3 = first_m3(val_date)
        val4 = fit_sfc_on_trials(val_m3, basis, 3, lag)
        val9 = fit_sfc_on_trials(val_m3, basis, 8, lag)
        std_raw4 = standardize_coefficients(val4.raw_vector, mean4, std4)
        std_raw9 = standardize_coefficients(val9.raw_vector, mean9, std9)
        padded4 = pad_carrier(std_raw4, 3)
        padded9 = pad_carrier(std_raw9, 8)
        # SFC4 is not a truncation of SFC9
        trunc = np.zeros_like(padded9)
        trunc[:, :4] = std_raw9[:, :4]
        independent = bool(not np.allclose(padded4[:, :4], trunc[:, :4]))

        held = calib_trials(val_date)[3:]
        held_r2_4 = encoding_r2(held, val4, basis)
        held_r2_9 = encoding_r2(held, val9, basis)
        der = derange_trials(val_m3)
        der4 = fit_sfc_on_trials(der, basis, 3, lag)
        der9 = fit_sfc_on_trials(der, basis, 8, lag)
        der_r2_4 = encoding_r2(held, der4, basis)
        der_r2_9 = encoding_r2(held, der9, basis)

        queries = query_trials(val_date)
        in_range = queries[: spec["n_in_range"]]
        src_members = train_trials
        tpl_src = tpl_family(src_members, queries)
        tpl_m3 = tpl_family(val_m3, queries)
        tpl_src_in = tpl_family(src_members, in_range)
        tpl_m3_in = tpl_family(val_m3, in_range)

        dr = fit_dr_family(val_m3, queries, in_range, basis)
        dr_public = {k: v for k, v in dr.items() if not k.startswith("_")}

        cond4_std = float(np.linalg.cond(std_raw4)) if np.isfinite(std_raw4).all() else float("inf")
        cond9_std = float(np.linalg.cond(std_raw9)) if np.isfinite(std_raw9).all() else float("inf")

        gates = {
            "sfc9_finite": bool(val9.finite),
            "sfc4_finite": bool(val4.finite),
            "design_rank_sfc4": val4.design_rank,
            "design_rank_sfc9": val9.design_rank,
            "design_rank_ok": val4.design_rank == 4 and val9.design_rank == 9,
            "condition_number_sfc4": val4.condition_number,
            "condition_number_sfc9": val9.condition_number,
            "condition_number_after_coeff_standardization_sfc4": cond4_std,
            "condition_number_after_coeff_standardization_sfc9": cond9_std,
            "condition_ok": val4.condition_number <= 1e4 and val9.condition_number <= 1e4,
            "held_calibration_encoding_r2_sfc4": held_r2_4,
            "held_calibration_encoding_r2_sfc9": held_r2_9,
            "label_derangement_r2_sfc4": der_r2_4,
            "label_derangement_r2_sfc9": der_r2_9,
            "derangement_does_not_exceed_correct_sfc4": der_r2_4 <= held_r2_4,
            "derangement_does_not_exceed_correct_sfc9": der_r2_9 <= held_r2_9,
            "sfc4_independent_of_sfc9_truncation": independent,
            "zero9_literal": sha256_array(zero9()),
            "split_half_training_dates": split_rows,
        }
        gates["fold_constructibility_fields_written"] = True

        receipt = {
            "fold": fold,
            "train_dates": list(train_dates),
            "val_date": val_date,
            "input_sha256s": input_sha,
            "alignment_table": alignment_table(),
            "acoustic_basis": basis_rec,
            "sfc_lag": lag_rec,
            "sfc4": _json_fit(val4, {"padded_standardized_sha256": sha256_array(padded4), "std_cond": cond4_std}),
            "sfc9": _json_fit(val9, {"padded_standardized_sha256": sha256_array(padded9), "std_cond": cond9_std}),
            "normalizer_sfc4": {"mean_sha256": sha256_array(mean4), "std_sha256": sha256_array(std4)},
            "normalizer_sfc9": {"mean_sha256": sha256_array(mean9), "std_sha256": sha256_array(std9)},
            "gates": gates,
            "TPL-SRC": {k: v for k, v in tpl_src.items()},
            "TPL-M3": {k: v for k, v in tpl_m3.items()},
            "TPL-SRC-in_range": {k: v for k, v in tpl_src_in.items()},
            "TPL-M3-in_range": {k: v for k, v in tpl_m3_in.items()},
            "DR": dr_public,
            "n_query_full": len(queries),
            "n_query_in_range": len(in_range),
        }
        write_json(fold_dir / "receipt.json", receipt)
        fold_summaries.append(receipt)

    # source-session split-half gate: 3 dates, mean over folds where date was training
    session_corr = {}
    session_cos = {}
    for date, rows in source_split_half.items():
        session_corr[date] = float(np.mean([r["flattened_W_correlation"] for r in rows]))
        session_cos[date] = float(np.mean([r["median_per_channel_cosine"] for r in rows]))
    n_corr = int(sum(v >= 0.50 for v in session_corr.values()))
    n_cos = int(sum(v >= 0.60 for v in session_cos.values()))

    freeze = {
        "sfc9_finite_all_folds": all(r["gates"]["sfc9_finite"] for r in fold_summaries),
        "sfc4_finite_all_folds": all(r["gates"]["sfc4_finite"] for r in fold_summaries),
        "design_rank_ok_all_folds": all(r["gates"]["design_rank_ok"] for r in fold_summaries),
        "condition_ok_all_folds": all(r["gates"]["condition_ok"] for r in fold_summaries),
        "split_half_W_corr_ge_0.50_sessions": session_corr,
        "split_half_W_corr_n_pass": f"{n_corr}/3",
        "median_cosine_ge_0.60_sessions": session_cos,
        "median_cosine_n_pass": f"{n_cos}/3",
        "derangement_ok_all_folds": all(
            r["gates"]["derangement_does_not_exceed_correct_sfc4"]
            and r["gates"]["derangement_does_not_exceed_correct_sfc9"]
            for r in fold_summaries
        ),
        "metric_bitwise_parity": metric_parity,
    }
    freeze["pass"] = bool(
        freeze["sfc9_finite_all_folds"]
        and freeze["sfc4_finite_all_folds"]
        and freeze["design_rank_ok_all_folds"]
        and freeze["condition_ok_all_folds"]
        and n_corr >= 2
        and n_cos >= 2
        and freeze["derangement_ok_all_folds"]
        and metric_parity["bitwise_equal"]
    )

    def _tpl_row(name_full, name_in):
        return {
            str(r["fold"]): {
                "val_date": r["val_date"],
                "full": {how: r[name_full][how]["mse_mean"] for how in ("raw_mean", "log_mean", "median")},
                "full_best": r[name_full]["best"],
                "in_range": {how: r[name_in][how]["mse_mean"] for how in ("raw_mean", "log_mean", "median")},
            }
            for r in fold_summaries
        }

    summary = {
        "input_sha256s": input_sha,
        "freeze_gate": freeze,
        "lags": {str(r["fold"]): r["sfc_lag"]["selected_lag_ms"] for r in fold_summaries},
        "TPL-SRC": _tpl_row("TPL-SRC", "TPL-SRC-in_range"),
        "TPL-M3": _tpl_row("TPL-M3", "TPL-M3-in_range"),
        "DR": {
            str(r["fold"]): {
                "val_date": r["val_date"],
                **{
                    name: {
                        "lambda": r["DR"][name]["selected_lambda"],
                        "lag_ms": r["DR"][name]["selected_lag_ms"],
                        "full": r["DR"][name]["full_stream"],
                        "in_range": r["DR"][name]["in_range"],
                    }
                    for name in ("DR-158-ML", "DR-158-SL", "DR-PC8")
                },
            }
            for r in fold_summaries
        },
        "sfc_constructibility": {
            str(r["fold"]): {
                "lag_ms": r["sfc_lag"]["selected_lag_ms"],
                "sfc4": r["sfc4"],
                "sfc9": r["sfc9"],
                "gates": r["gates"],
            }
            for r in fold_summaries
        },
    }
    write_json(out_dir / "stage0b_summary.json", summary)
    return summary


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from .data import run_stage0a

    inv = run_stage0a()
    run_stage0b(inventory=inv)
