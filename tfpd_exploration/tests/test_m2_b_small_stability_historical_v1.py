"""Historical last-k patch: same-surface gaps only; no new scoring."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tfpd_exploration.src.m2_b_small_stability_v1 import config as cfg
from tfpd_exploration.src.m2_b_small_stability_v1.report import (
    FORBIDDEN_MAMBA_MINIVAL_SUBTRACTION,
    build_historical_trajectory_summary,
    last_k_stats,
)


REPO = Path(__file__).resolve().parents[2]
OLD_ROOT = REPO / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_input_shas_bind_old_root_read_only() -> None:
    assert _sha256(OLD_ROOT / "ANALYSIS_B_TRAJECTORY_INSTABILITY.md") == cfg.OLD_ANALYSIS_SHA256
    assert _sha256(OLD_ROOT / "comparison.csv") == cfg.OLD_COMPARISON_SHA256
    assert cfg.OLD_ANALYSIS_SHA256 == "af64ee40fa5dc3c0a4036f1d67c2ea48e92866b33d5803d1f48309d2ce780d60"
    assert cfg.OLD_COMPARISON_SHA256 == "94ab3271896a01feb68b090c1896e4a893c92ad46556d803b4b122965977a95a"


def test_last_k_uses_ddof0_and_workorder_numbers() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    stats = last_k_stats(values)
    assert abs(stats["mean"] - 0.25) < 1e-15
    assert abs(stats["std"] - float(np.std(values, ddof=0))) < 1e-15
    assert stats["std_ddof"] == 0


def test_historical_summary_same_surface_gap() -> None:
    summary = build_historical_trajectory_summary()
    assert summary["R_REF_session"] == 0.3582396424175502
    assert abs(summary["R_REF_date"] - 0.3279795072) < 1e-10
    trf = summary["transformer"]
    mamba = summary["mamba"]
    assert trf["source_pick_epoch"] == 9
    assert abs(trf["source_pick_ext4"] - 0.360008) < 5e-6
    assert abs(trf["delta_vs_ref"] - 0.001768) < 5e-6
    assert abs(trf["endpoint24"] - 0.321388) < 5e-6
    assert abs(trf["last4"]["mean"] - 0.332195) < 5e-6
    assert abs(trf["last4"]["std"] - 0.008984) < 5e-6
    assert abs(trf["last8"]["mean"] - 0.340155) < 5e-6
    assert abs(trf["last8"]["std"] - 0.024152) < 5e-6
    assert abs(trf["visible_ext4_epoch20"] - 0.371954) < 5e-6
    assert mamba["source_pick_epoch"] == 24
    assert abs(mamba["source_pick_ext4"] - 0.343016) < 5e-6
    assert abs(mamba["delta_vs_ref"] + 0.015224) < 5e-6
    assert abs(mamba["last4"]["mean"] - 0.339657) < 5e-6
    assert abs(mamba["last4"]["std"] - 0.015601) < 5e-6
    assert abs(mamba["last8"]["mean"] - 0.348886) < 5e-6
    assert abs(mamba["last8"]["std"] - 0.016601) < 5e-6
    assert abs(mamba["visible_ext4_epoch18"] - 0.377224) < 5e-6
    # CRITICAL: do not subtract Mamba source-minival 0.309 from REF 0.3582
    assert abs(mamba["same_surface_gap"] + 0.015224) < 5e-6
    assert mamba["same_surface_gap"] != (0.309186 - 0.3582396424175502)
    assert FORBIDDEN_MAMBA_MINIVAL_SUBTRACTION not in (
        mamba["delta_vs_ref"],
        mamba["same_surface_gap"],
    )
    assert summary["note_same_surface"] == "Mamba source-pick ext-4 minus REF is -0.015224; do not subtract minival 0.309"


def test_historical_summary_binds_input_shas() -> None:
    summary = build_historical_trajectory_summary()
    assert summary["inputs"]["analysis_sha256"] == cfg.OLD_ANALYSIS_SHA256
    assert summary["inputs"]["comparison_csv_sha256"] == cfg.OLD_COMPARISON_SHA256
    assert summary["inputs"]["old_root"] == str(cfg.OLD_ROOT.relative_to(REPO))
    assert "transformer_ext4_scan_sha256" in summary["inputs"]
    assert "mamba_ext4_scan_sha256" in summary["inputs"]
    assert summary["std_ddof"] == 0
    assert summary["last_k_is_not_a_checkpoint"] is True
