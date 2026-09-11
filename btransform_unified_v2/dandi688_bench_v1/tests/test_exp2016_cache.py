"""exp2016 extension-cache correctness tests (variant exp2016_sua_t4).

Covers the admission law of the 2016 remote exam (coordinator ruling
2026-09-10, plan A) and the byte-level contract of the built cache:
  1. schema compatibility: run_688_bench.verify_prepared_cache passes; the
     source-manifest attestation (27/6) carries the explicit semantics note;
  2. roster law: 18 train + 14 exam2016 sessions, exactly the protocol face;
  3. train rows: neural/behavior/starts/mask byte-copied from the frozen
     cache, carrier/E0 recomputed under the 18-session normalizer (differ
     from the frozen 27-session-domain bytes by design);
  4. exam rows: first-91-by-table-order law recorded verbatim, Nmax-91
     geometry with an all-real mask, finite carrier/E0, non-empty Q50
     starts, kept-vs-dropped firing-rate diagnostics present.

Run:
  cd /home/xinyuan/Work_host/SPINT && PYTHONNOUSERSITE=1 \
    /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest \
    btransform_unified_v2/dandi688_bench_v1/tests/test_exp2016_cache.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
for p in (PKG_ROOT / "src", PKG_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

WS = PKG_ROOT.parents[1]
for p in (WS / "btransform_unified_v2" / "src", WS / "btransform_unified_v1" / "src",
          WS / "sua_exploration", WS / "streaming_calibration_exp", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dandi688_bench_v1 import plan  # noqa: E402

EXT_DEST = PKG_ROOT / "results" / "cache_exp2016"
TRAIN_SAMPLE = "sub-C_ses-CO-20150716"
EXAM_SAMPLE = "sub-C_ses-CO-20160909"


@pytest.mark.skipif(
    not (EXT_DEST / "prepared_contract.json").is_file(),
    reason="exp2016 extension cache not built yet "
           "(scripts/build_exp2016_cache.py)",
)
class TestExp2016Cache:
    def test_schema_and_attestation(self):
        from run_688_bench import verify_prepared_cache

        meta = verify_prepared_cache(EXT_DEST)
        assert meta["variant"] == "exp2016_sua_t4"
        assert meta["formal_test_used"] is False
        assert "SOURCE manifest" in meta["split_counts_semantics"]
        assert meta["estimator"]["protocol"] == "exp2016"
        assert meta["estimator"]["sua_parity_gate"]["pass"] is True
        assert meta["estimator"]["normalizer_domain"]["train_sessions"] == 18
        assert "first-91-by-table-order" in meta["estimator"]["units_selection_law"]
        assert "Nmax locked to train roster" in meta["estimator"]["units_selection_law"]

    def test_roster_matches_protocol(self):
        meta = json.loads((EXT_DEST / "prepared_contract.json").read_text())
        sessions = plan.protocol_sessions("exp2016")
        assert set(meta["sessions"]) == set(sessions["train"]) | set(sessions["exam"])
        assert sum(s["split"] == "train" for s in meta["sessions"].values()) == 18
        assert sum(s["split"] == "exam2016" for s in meta["sessions"].values()) == 14
        assert not set(meta["sessions"]) & set(plan.FORMAL_TEST_SESSIONS)

    def test_train_rows_frozen_bytes_and_18_domain_recompute(self):
        from dandi688_bench_v1 import eval_local

        frozen = eval_local.load_session(plan.prepared_cache_path(), TRAIN_SAMPLE)
        row = eval_local.load_session(EXT_DEST, TRAIN_SAMPLE)
        for key in ("neural", "behavior", "starts", "mask"):
            assert plan.array_digest(row[key]) == plan.array_digest(frozen[key])
        # the 18-session normalizer domain must differ from the frozen
        # 27-session domain on the carrier bytes (protocol-bound recompute)
        assert row["carrier"].shape == frozen["carrier"].shape
        assert not np.array_equal(row["carrier"], frozen["carrier"])
        assert not np.array_equal(row["e0"], frozen["e0"])
        n_real = int(row["mask"].sum())
        assert bool(np.any(row["e0"][:n_real]))
        assert not bool(np.any(row["e0"][n_real:]))

    def test_exam_rows_first91_law_and_geometry(self):
        from dandi688_bench_v1 import eval_local

        meta = json.loads((EXT_DEST / "prepared_contract.json").read_text())
        for name in (EXAM_SAMPLE, plan.EXP2016_EXAM_SESSIONS[-1]):
            row = eval_local.load_session(EXT_DEST, name)
            info = meta["identity_log"][name]
            # first-91 law recorded with diagnostics
            assert info["units_selection_law"] == \
                meta["estimator"]["units_selection_law"]
            assert info["n_units_kept"] == 91
            assert info["n_units_total"] == info["n_units_kept"] + info["n_units_dropped"]
            assert info["n_units_total"] >= 188  # roster-law conflict disclosed
            diag = info["firing_rate_diagnostics"]
            assert diag["kept_first_91"]["n"] == 91
            assert diag["dropped_remainder"]["n"] == info["n_units_dropped"]
            assert diag["kept_minus_dropped_mean_hz"] == pytest.approx(
                diag["kept_first_91"]["mean_hz"] - diag["dropped_remainder"]["mean_hz"])
            # geometry: exactly 91 real channels, Nmax width, no padding
            assert int(row["mask"].sum()) == 91 and bool(row["mask"].all())
            assert row["neural"].shape[1] == row["carrier"].shape[0] == 91
            assert row["e0"].shape == (91, plan.E0_DIM)
            assert bool(np.any(row["e0"])) and np.isfinite(row["carrier"]).all()
            assert row["starts"].dtype == np.int64 and row["starts"].size > 0
            assert np.isfinite(row["behavior"]).all()

    def test_runner_loads_protocol_face(self):
        from run_688_bench import load_rows, verify_prepared_cache

        meta = verify_prepared_cache(EXT_DEST)
        rows = load_rows(EXT_DEST, meta, "exp2016")
        assert len(rows) == 32
        assert sum(r["protocol_role"] == "train" for r in rows.values()) == 18
        assert sum(r["protocol_role"] == "exam" for r in rows.values()) == 14
        # every row shares the Nmax-91 geometry (model width consistency)
        assert all(r["neural"].shape[1] == 91 for r in rows.values())
