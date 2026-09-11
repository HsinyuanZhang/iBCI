"""CPU-only pytest suite for the dandi688_bench_v1 skeleton.

Run:
  cd /home/xinyuan/Work_host/SPINT && PYTHONNOUSERSITE=1 \
    /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest \
    btransform_unified_v2/dandi688_bench_v1/tests/ -q

Discipline: no CUDA initialization anywhere; the formal-test split is never
loaded (its names are exercised only as rejection cases).
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
for p in (PKG_ROOT / "src", PKG_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dandi688_bench_v1 import arms, eval_local, plan, receipts, support_resample, ts4_shuffle  # noqa: E402


def synthetic_carrier(n_real: int = 12, n_pad: int = 3, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    carrier = np.zeros((n_real + n_pad, plan.CARRIER_DIM), np.float32)
    carrier[:n_real] = rng.normal(size=(n_real, plan.CARRIER_DIM)).astype(np.float32)
    mask = np.zeros(n_real + n_pad, bool)
    mask[:n_real] = True
    return carrier, mask


# --------------------------------------------------------------------------
# plan constants are frozen
# --------------------------------------------------------------------------
def test_plan_constants_frozen():
    assert plan.SCHEMA == "dandi688_bench_v1"
    assert plan.ARMS == (
        "t4", "f0", "ts4", "t4_concat", "z0",              # original five
        "floor", "equiv_zero",                             # component ablation (2026-09-09)
        "vstate", "vstate_concat",                          # vstate main + concat reading
        "z_vstate_srcbank", "f_labelfree",                  # vstate-688 ablations
        "norm_only",       # NORM_ONLY zero-calibration baseline (guide 2026-09-10)
    )
    assert plan.FUSION_MODES == ("proj_add", "concat")
    assert plan.LOCAL_CONV_CHANNELS == 16
    assert plan.CONCAT_TOKEN_IN == 70 == 16 + plan.E0_DIM + plan.CARRIER_DIM
    assert plan.GATE_T4_MINUS_F0 == 0.03
    assert plan.GATE_T4_MINUS_TS4 == 0.03
    assert plan.GATE_VSTATE_MINUS_T4 == 0.03
    assert plan.REF_SPINT_T4_DEV6 == 0.5750
    assert plan.NONINFERIORITY_TOLERANCE == 0.01
    assert plan.VAL_FACE_ROLE == "ext6-equivalent: 6 val sessions"
    assert plan.FORMAL_TEST_POLICY == (
        "sealed until all preregistered val-face verdicts land; one-shot unseal"
    )
    assert plan.GPU_POLICY == "ALLOWED_REVISION_20260909"
    assert plan.WINDOW_BINS == 50 and plan.CARRIER_DIM == 4 and plan.E0_DIM == 50
    assert plan.SEED == 42 and plan.EPOCHS == 12
    assert tuple(plan.AVG_EPOCHS_ZERO_BASED) == (8, 9, 10, 11)
    # manifest sha must match the on-disk frozen manifest (read-only check)
    assert plan.file_sha256(plan.manifest_path()) == plan.MANIFEST_SHA256
    assert plan.MANIFEST_SHA256 == "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
    assert plan.PREPARED_CACHE_SCHEMA == "dandi688_rift_v2_prepared"
    assert plan.PREPARED_CACHE_RELATIVE.endswith("dandi688_prepared_cache_contract_v2")
    # two-stage protocol constants (ADDENDUM-TWO-STAGE) + the 2026-09-10
    # extreme-poverty ablation protocol (user directive, plan B) + the
    # 2026-09-09 2015-only remote-exam protocol + the 2026-09-10 2016
    # remote-exam protocol (first-91 ruling)
    assert tuple(plan.PROTOCOLS) == ("exp1_narrow", "exp2_full", "exp1_poverty",
                                     "exp2015_full", "exp2016")
    assert plan.DEFAULT_PROTOCOL == "exp1_narrow"
    assert plan.PROTOCOLS["exp1_narrow"]["train_sessions"] == plan.EXP1_NARROW_TRAIN_SESSIONS
    assert plan.PROTOCOLS["exp1_narrow"]["exam_sessions"] == plan.EXP1_NARROW_EXAM_SESSIONS
    assert plan.PROTOCOLS["exp1_narrow"]["span_days"] == 12
    assert plan.PROTOCOLS["exp1_narrow"]["m2_anchor"] == "7 train/10d; ext4 days-after"
    assert plan.PROTOCOLS["exp2_full"]["train_sessions"] == "ALL_TRAIN"
    assert plan.PROTOCOLS["exp2_full"]["exam_sessions"] == "ALL_VAL"
    assert "20150713" in plan.CROSS_PROTOCOL_DISCLOSURE and "EXP-2" in plan.CROSS_PROTOCOL_DISCLOSURE


def test_plan_session_lists_match_manifest():
    manifest = json.loads(plan.manifest_path().read_text())
    splits = manifest["session_splits"]
    assert tuple(splits["val"]) == plan.VAL_SESSIONS
    assert tuple(splits["test"]) == plan.FORMAL_TEST_SESSIONS
    assert len(splits["train"]) == plan.SPLIT_COUNTS["train"] == 27
    assert not set(plan.VAL_SESSIONS) & set(plan.FORMAL_TEST_SESSIONS)


# --------------------------------------------------------------------------
# two-stage protocols (ADDENDUM-TWO-STAGE)
# --------------------------------------------------------------------------
def test_exp1_narrow_protocol_geometry():
    """9+4 sessions, 12-day inclusive train span, exam 3-6 days after train."""
    train, exam = plan.EXP1_NARROW_TRAIN_SESSIONS, plan.EXP1_NARROW_EXAM_SESSIONS
    assert len(train) == 9 and len(exam) == 4
    assert train[0] == "sub-C_ses-CO-20150629" and train[-1] == "sub-C_ses-CO-20150710"
    assert exam == ("sub-C_ses-CO-20150713", "sub-C_ses-CO-20150714",
                    "sub-C_ses-CO-20150715", "sub-C_ses-CO-20150716")
    assert not set(train) & set(exam)
    dtrain = [plan.session_date(n) for n in train]
    dexam = [plan.session_date(n) for n in exam]
    assert dtrain == sorted(dtrain) and dexam == sorted(dexam)
    assert (dtrain[-1] - dtrain[0]).days + 1 == plan.PROTOCOLS["exp1_narrow"]["span_days"] == 12
    last = dtrain[-1]
    for d in dexam:
        assert 3 <= (d - last).days <= 6  # "days-after" exam paper, mirrors M2 ext4
    # session_date parses the trailing YYYYMMDD and refuses nameless dates
    assert plan.session_date("sub-C_ses-CO-20150713").isoformat() == "2015-07-13"
    with pytest.raises(RuntimeError, match="YYYYMMDD"):
        plan.session_date("not-a-session")


def test_protocol_resolution_matches_manifest():
    resolved = plan.verify_protocol_definitions()
    splits = json.loads(plan.manifest_path().read_text())["session_splits"]
    assert resolved["exp1_narrow"]["train"] == plan.EXP1_NARROW_TRAIN_SESSIONS
    assert resolved["exp1_narrow"]["exam"] == plan.EXP1_NARROW_EXAM_SESSIONS
    assert resolved["exp2_full"]["train"] == tuple(splits["train"])
    assert len(resolved["exp2_full"]["train"]) == 27
    assert resolved["exp2_full"]["exam"] == tuple(splits["val"]) == plan.VAL_SESSIONS
    with pytest.raises(ValueError, match="unknown protocol"):
        plan.protocol_sessions("exp3_half")
    # receipt block carries the protocol field, session lists and span
    block = plan.protocol_receipt_block("exp1_narrow")
    assert block["name"] == "exp1_narrow"
    assert block["train_sessions"] == list(plan.EXP1_NARROW_TRAIN_SESSIONS)
    assert block["exam_sessions"] == list(plan.EXP1_NARROW_EXAM_SESSIONS)
    assert block["n_train"] == 9 and block["n_exam"] == 4 and block["span_days"] == 12
    assert block["cross_protocol_disclosure"] == plan.CROSS_PROTOCOL_DISCLOSURE


def test_cross_protocol_disclosure_exp1_exam_inside_exp2_train():
    resolved = plan.verify_protocol_definitions()
    assert set(plan.EXP1_NARROW_EXAM_SESSIONS) <= set(resolved["exp2_full"]["train"])
    assert set(plan.EXP1_NARROW_TRAIN_SESSIONS) <= set(resolved["exp2_full"]["train"])
    # never a val/formal-test session under any protocol
    exp1_all = set(plan.EXP1_NARROW_TRAIN_SESSIONS) | set(plan.EXP1_NARROW_EXAM_SESSIONS)
    assert not exp1_all & set(plan.VAL_SESSIONS)
    assert not exp1_all & set(plan.FORMAL_TEST_SESSIONS)
    # the disclosure states the overlap and the no-cross-comparison rule
    text = plan.CROSS_PROTOCOL_DISCLOSURE
    assert "EXP-1" in text and "EXP-2" in text and "train" in text
    assert "never compare" in text.lower()


def test_default_protocol_is_exp1_narrow():
    import run_688_bench as runner

    assert plan.DEFAULT_PROTOCOL == "exp1_narrow"  # user ruling: narrow first, then full
    parser = runner.build_parser()
    assert parser.get_default("protocol") == plan.DEFAULT_PROTOCOL
    protocol_action = next(a for a in parser._actions if "--protocol" in a.option_strings)
    assert set(protocol_action.choices) == {
        "exp1_narrow", "exp2_full", "exp1_poverty", "exp2015_full", "exp2016"}


def test_filter_rows_by_protocol_on_synthetic_rows():
    import run_688_bench as runner

    splits = json.loads(plan.manifest_path().read_text())["session_splits"]
    rows = {
        name: {"split": "train" if name in splits["train"] else "val",
               "starts": np.arange(10, dtype=np.int64)}
        for name in list(splits["train"]) + list(splits["val"])
    }
    assert len(rows) == 33

    exp1 = runner.filter_rows_by_protocol(rows, "exp1_narrow")
    assert len(exp1) == 13
    assert {n for n, r in exp1.items() if r["protocol_role"] == "train"} == set(plan.EXP1_NARROW_TRAIN_SESSIONS)
    assert {n for n, r in exp1.items() if r["protocol_role"] == "exam"} == set(plan.EXP1_NARROW_EXAM_SESSIONS)
    # a cache-train session outside the narrow window is filtered out
    assert "sub-C_ses-CO-20150309" not in exp1 and "sub-C_ses-CO-20131003" not in exp1
    # inputs never mutated
    assert all("protocol_role" not in r for r in rows.values())

    exp2 = runner.filter_rows_by_protocol(rows, "exp2_full")
    assert len(exp2) == 33
    assert sum(r["protocol_role"] == "train" for r in exp2.values()) == 27
    assert sum(r["protocol_role"] == "exam" for r in exp2.values()) == 6
    # fail closed when a protocol session is missing
    with pytest.raises(RuntimeError, match="expects 13"):
        runner.filter_rows_by_protocol(
            {n: r for n, r in rows.items() if n != plan.EXP1_NARROW_EXAM_SESSIONS[-1]},
            "exp1_narrow",
        )


def test_gate_report_arithmetic():
    good = plan.gate_report(0.60, 0.55, 0.55)
    assert good["gate1_t4_minus_f0"]["pass"] and good["gate2_t4_minus_ts4"]["pass"]
    assert good["gate3_noninferiority_vs_ref"]["floor"] == pytest.approx(0.565)
    assert good["gate3_noninferiority_vs_ref"]["pass"]
    bad = plan.gate_report(0.56, 0.54, 0.56)
    assert not bad["gate1_t4_minus_f0"]["pass"] and not bad["gate2_t4_minus_ts4"]["pass"]
    assert not bad["gate3_noninferiority_vs_ref"]["pass"]


def test_gate_report_never_gates_the_concat_arm():
    # The fusion-ablation arm reports val-face readings only: gate arithmetic
    # stays three-armed (t4/f0/ts4) regardless of plan.ARMS growth.
    report = plan.gate_report(0.60, 0.55, 0.55)
    assert set(report["r2"]) == {"t4", "f0", "ts4"}
    assert "t4_concat" not in plan.json_dumps(report)


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def test_arm_t4_is_identity_and_pure():
    carrier, mask = synthetic_carrier()
    before = carrier.copy()
    out = arms.apply_arm("t4", "sub-C_ses-CO-20151103", carrier, mask)
    assert np.array_equal(out, carrier)
    assert out is not carrier
    assert np.array_equal(carrier, before)  # input never mutated
    rec = arms.verify_arm("t4", "sub-C_ses-CO-20151103", carrier, mask, out)
    assert rec["verify"] == "PASSED" and rec["rows_moved"] == 0


def test_arm_f0_is_zero_and_pure():
    carrier, mask = synthetic_carrier()
    out = arms.apply_arm("f0", "s", carrier, mask)
    assert out.shape == carrier.shape and not np.any(out)
    assert np.array_equal(carrier, synthetic_carrier()[0])  # input untouched
    assert arms.verify_arm("f0", "s", carrier, mask, out)["verify"] == "PASSED"
    # a nonzero f0 must fail verification
    bad = out.copy(); bad[0, 0] = 1.0
    with pytest.raises(RuntimeError, match="exactly zero"):
        arms.verify_arm("f0", "s", carrier, mask, bad)
    # floor shares the all-zero carrier; E0 zeroing is the runner's job.
    # z0 (component-ablation redefinition 2026-09-09) KEEPS the true carrier
    # and only zeroes E0 -- the carrier-only cell of the 2x2.
    assert np.array_equal(arms.apply_arm("floor", "s", carrier, mask), out)
    assert np.array_equal(arms.apply_arm("z0", "s", carrier, mask), carrier)
    assert arms.arm_e0_action("z0") == "zero" and arms.arm_e0_action("f0") == "identity"
    assert arms.arm_e0_action("floor") == "zero"


def test_arm_t4_tamper_rejected():
    carrier, mask = synthetic_carrier()
    bad = carrier.copy(); bad[0, 0] += 0.5
    with pytest.raises(RuntimeError, match="byte-for-byte"):
        arms.verify_arm("t4", "s", carrier, mask, bad)


def test_arm_enum_is_closed():
    for bad_arm in ("ts5", "", "T4", None):
        with pytest.raises(ValueError):
            arms.apply_arm(bad_arm, "s", *synthetic_carrier())


def test_ts4_preserves_column_marginals_and_padding():
    carrier, mask = synthetic_carrier(n_real=40, n_pad=5)
    out = arms.apply_arm("ts4", "sub-C_ses-CO-20151103", carrier, mask)
    # per-column marginal equality, whole matrix and real-row subset
    for j in range(plan.CARRIER_DIM):
        assert np.array_equal(np.sort(carrier[:, j]), np.sort(out[:, j]))
        real = np.flatnonzero(mask)
        assert np.array_equal(np.sort(carrier[real, j]), np.sort(out[real, j]))
    # padded rows stay exactly where they were (zero)
    padded = np.flatnonzero(~mask)
    assert np.array_equal(out[padded], carrier[padded]) and not np.any(out[padded])
    # real rows are a permutation: same row multiset, but assignment moved
    a = carrier[np.lexsort(carrier.T[::-1])]
    b = out[np.lexsort(out.T[::-1])]
    assert np.array_equal(a, b)
    assert not np.array_equal(out, carrier)
    rec = arms.verify_arm("ts4", "sub-C_ses-CO-20151103", carrier, mask, out)
    assert rec["verify"] == "PASSED" and rec["rows_moved"] > 0


def test_ts4_deterministic_and_session_independent():
    carrier, mask = synthetic_carrier(n_real=40)
    s1, s2 = "sub-C_ses-CO-20151103", "sub-C_ses-CO-20151104"
    out1 = arms.apply_arm("ts4", s1, carrier, mask)
    assert np.array_equal(out1, arms.apply_arm("ts4", s1, carrier, mask))  # deterministic
    out2 = arms.apply_arm("ts4", s2, carrier, mask)
    assert not np.array_equal(out1, out2)  # independent per-session permutations
    # seed derivation is name-sensitive and stable
    assert ts4_shuffle.permutation_seed(s1) == ts4_shuffle.permutation_seed(s1)
    assert ts4_shuffle.permutation_seed(s1) != ts4_shuffle.permutation_seed(s2)
    d = ts4_shuffle.derivation_record(s1, carrier, mask)
    assert d["n_real_rows"] == 40 and d["rows_moved"] > 0 and d["seed"] == plan.SEED


def test_ts4_value_tamper_rejected_by_digest_assertion():
    carrier, mask = synthetic_carrier(n_real=40)
    out = arms.apply_arm("ts4", "s", carrier, mask)
    tampered = out.copy(); tampered[0, 1] += 1e-3
    with pytest.raises(RuntimeError, match="column marginal"):
        arms.verify_arm("ts4", "s", carrier, mask, tampered)
    with pytest.raises(RuntimeError, match="column marginal"):
        arms.assert_column_marginals_equal(carrier, tampered)
    # a hand-rolled different permutation of the same real rows is still legal
    real = np.flatnonzero(mask)
    manual = carrier.copy()
    manual[real] = carrier[np.roll(real, 1)]
    assert arms.verify_arm("ts4", "s", carrier, mask, manual)["verify"] == "PASSED"


def test_arm_specs_map_every_arm_to_legal_carrier_and_fusion():
    assert set(arms.ARM_SPECS) == set(plan.ARMS)
    for arm, spec in arms.ARM_SPECS.items():
        assert spec["carrier"] in arms.CARRIER_ACTIONS
        assert spec["fusion"] in plan.FUSION_MODES
        assert spec["e0"] in arms.E0_ACTIONS
    # the three gated carrier arms keep the settled proj_add frontend
    assert all(arms.ARM_SPECS[a]["fusion"] == "proj_add" for a in ("t4", "f0", "ts4"))
    # the fusion-ablation arm: true carrier (原样) + concat fusion
    assert arms.ARM_SPECS["t4_concat"] == {
        "carrier": "identity", "fusion": "concat", "e0": "identity",
    }
    # z0 = the carrier-only cell of the component 2x2 (user clarification
    # 2026-09-09): carrier identity (like t4) + E0 zero.  Its pre-2026-09-09
    # both-zero semantics live on as floor.
    assert arms.ARM_SPECS["z0"] == {"carrier": "identity", "fusion": "proj_add", "e0": "zero"}
    assert arms.ARM_SPECS["floor"] == {"carrier": "zero", "fusion": "proj_add", "e0": "zero"}
    assert arms.ARM_SPECS["equiv_zero"] == {
        "carrier": "zero", "fusion": "proj_add", "e0": "identity",
    }
    # vstate-688 series (user directive 2026-09-09): variant-cache bound;
    # z-series bank swap is a runner-level cross-session operation
    assert arms.ARM_SPECS["vstate"] == {
        "carrier": "identity", "fusion": "proj_add", "e0": "identity",
    }
    # vstate_concat: the M2-mainline concat-fusion reading of the vstate
    # carrier (alignment matrix "fusion" row); readings only, never gates
    assert arms.ARM_SPECS["vstate_concat"] == {
        "carrier": "identity", "fusion": "concat", "e0": "identity",
    }
    assert arms.ARM_SPECS["z_vstate_srcbank"] == {
        "carrier": "identity", "fusion": "proj_add", "e0": "identity",
    }
    assert arms.ARM_SPECS["f_labelfree"] == {
        "carrier": "zero", "fusion": "proj_add", "e0": "identity",
    }
    # norm_only (NORM_ONLY zero-calibration baseline, guide 2026-09-10): the
    # same arm-level law as f_labelfree -- zero carrier, identity E0; the
    # rate-deviation identity itself lives in the norm_only cache bytes
    assert arms.ARM_SPECS["norm_only"] == {
        "carrier": "zero", "fusion": "proj_add", "e0": "identity",
    }
    # the 11 pre-existing arm cells are unchanged by the new rung (frozen law)
    assert {arm: dict(spec) for arm, spec in arms.ARM_SPECS.items()
            if arm != "norm_only"} == {
        "t4": {"carrier": "identity", "fusion": "proj_add", "e0": "identity"},
        "f0": {"carrier": "zero", "fusion": "proj_add", "e0": "identity"},
        "ts4": {"carrier": "shuffle", "fusion": "proj_add", "e0": "identity"},
        "t4_concat": {"carrier": "identity", "fusion": "concat", "e0": "identity"},
        "z0": {"carrier": "identity", "fusion": "proj_add", "e0": "zero"},
        "floor": {"carrier": "zero", "fusion": "proj_add", "e0": "zero"},
        "equiv_zero": {"carrier": "zero", "fusion": "proj_add", "e0": "identity"},
        "vstate": {"carrier": "identity", "fusion": "proj_add", "e0": "identity"},
        "vstate_concat": {"carrier": "identity", "fusion": "concat", "e0": "identity"},
        "z_vstate_srcbank": {"carrier": "identity", "fusion": "proj_add", "e0": "identity"},
        "f_labelfree": {"carrier": "zero", "fusion": "proj_add", "e0": "identity"},
    }
    # cache-variant families: vstate/vstate_concat accept every vstate-family
    # variant; z_vstate_srcbank is defined against the MAIN variant only;
    # equiv_zero carries its random-projection E0 melt in its own cache;
    # norm_only carries the broadcast rate-deviation identity in its own
    family = frozenset(plan.VSTATE_VARIANTS)
    assert plan.ARM_REQUIRED_CACHE_VARIANT == {
        "vstate": family,
        "vstate_concat": family,
        "z_vstate_srcbank": frozenset({"vstate"}),
        "f_labelfree": frozenset({"f_labelfree"}),
        "equiv_zero": frozenset({"equiv_zero"}),
        "norm_only": frozenset({"norm_only"}),
    }
    # protocol-bound variant caches (train-session fit domain) vs label-free
    # protocol-free ones (f_labelfree / equiv_zero)
    assert plan.PROTOCOL_BOUND_CACHE_VARIANTS == (
        *plan.VSTATE_VARIANTS, "norm_only")
    for variant in ("f_labelfree", "equiv_zero"):
        assert variant not in plan.PROTOCOL_BOUND_CACHE_VARIANTS
    assert arms.arm_e0_action("f0") == "identity"
    assert arms.arm_e0_action("z0") == "zero"
    assert arms.arm_fusion("t4_concat") == "concat"
    with pytest.raises(ValueError, match="unknown arm"):
        arms.arm_fusion("t4_concqt")


def test_arm_t4_concat_carrier_is_t4_identity():
    carrier, mask = synthetic_carrier()
    before = carrier.copy()
    out = arms.apply_arm("t4_concat", "sub-C_ses-CO-20151103", carrier, mask)
    assert np.array_equal(out, carrier) and out is not carrier  # identity, pure
    assert np.array_equal(carrier, before)  # input never mutated
    rec = arms.verify_arm("t4_concat", "sub-C_ses-CO-20151103", carrier, mask, out)
    assert rec["verify"] == "PASSED" and rec["rows_moved"] == 0
    assert rec["fusion"] == "concat" and rec["carrier_action"] == "identity"
    # tamper rejected under the same byte-for-byte law as t4
    bad = out.copy(); bad[1, 2] += 0.25
    with pytest.raises(RuntimeError, match="byte-for-byte"):
        arms.verify_arm("t4_concat", "sub-C_ses-CO-20151103", carrier, mask, bad)
    # digest level: the fusion axis never touches carrier bytes
    assert (arms.arm_carrier_digest("t4_concat", "s", carrier, mask)
            == arms.arm_carrier_digest("t4", "s", carrier, mask)
            == plan.array_digest(carrier))


def test_arm_carrier_digests_differ_across_arms():
    carrier, mask = synthetic_carrier(n_real=40)
    digests = {
        arm: arms.arm_carrier_digest(arm, "s", carrier, mask) for arm in plan.ARMS
    }
    # 12 arms, 3 distinct carrier transforms: identity {t4, t4_concat, z0,
    # vstate, vstate_concat, z_vstate_srcbank}, zero {f0, floor, equiv_zero,
    # f_labelfree, norm_only}, shuffle {ts4}.
    assert len(plan.ARMS) == 12
    assert len(set(digests.values())) == 3
    assert digests["t4"] == plan.array_digest(carrier)
    assert digests["t4_concat"] == digests["t4"]
    assert digests["vstate"] == digests["vstate_concat"] == digests["t4"]
    assert digests["vstate"] == digests["z_vstate_srcbank"]
    assert digests["z0"] == digests["t4"]  # carrier-only cell keeps the carrier
    assert digests["f0"] == digests["floor"] == digests["equiv_zero"]
    assert digests["f0"] == digests["f_labelfree"]
    assert digests["norm_only"] == digests["f0"]
    assert digests["f0"] == plan.array_digest(np.zeros_like(carrier))


# --------------------------------------------------------------------------
# eval_local
# --------------------------------------------------------------------------
def test_session_r2_hand_computed():
    target = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], np.float32)
    # colmeans = [1, 1]; SS_tot = 4; pred adds 1 to the last query's 2nd dim
    pred = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]], np.float32)
    assert eval_local.session_r2(pred, target) == pytest.approx(1.0 - 1.0 / 4.0)
    assert eval_local.session_r2(target.copy(), target) == pytest.approx(1.0)
    # mean-predictor baseline: pred = column means -> R^2 = 0
    mean_pred = np.repeat(target.mean(axis=0, keepdims=True), 3, axis=0)
    assert eval_local.session_r2(mean_pred, target) == pytest.approx(0.0)
    # constant target -> nan by convention (total variance 0)
    assert np.isnan(eval_local.session_r2(np.zeros((3, 2)), np.ones((3, 2))))


def test_evaluate_val_face_aggregates_six_sessions():
    rng = np.random.default_rng(0)
    outputs = {}
    expected_r2 = {}
    for i, name in enumerate(plan.VAL_SESSIONS):
        target = rng.normal(size=(20 + i, 2)).astype(np.float32)
        noise = rng.normal(scale=0.1, size=target.shape).astype(np.float32)
        outputs[name] = (target + noise, target)
        expected_r2[name] = eval_local.session_r2(target + noise, target)
    face = eval_local.evaluate_val_face(outputs)
    assert face["schema"] == "dandi688_bench_v1_val_face"
    assert face["formal_test_used"] is False
    assert face["equal_session_mean_r2"] == pytest.approx(float(np.mean(list(expected_r2.values()))))
    for name in plan.VAL_SESSIONS:
        assert face["sessions"][name]["n_query"] == 20 + list(plan.VAL_SESSIONS).index(name)
        assert face["sessions"][name]["r2"] == pytest.approx(expected_r2[name])
        assert set(face["sessions"][name]) >= {"prediction_sha256", "target_sha256", "mse"}
    # wrong session count refused; test names refused
    with pytest.raises(RuntimeError, match="exactly 6"):
        eval_local.evaluate_val_face({k: outputs[k] for k in list(outputs)[:5]})
    poisoned = dict(outputs)
    poisoned[plan.FORMAL_TEST_SESSIONS[0]] = outputs[plan.VAL_SESSIONS[0]]
    with pytest.raises(eval_local.TestSplitForbiddenError):
        eval_local.evaluate_val_face(poisoned)


def test_evaluate_val_face_accepts_protocol_exam_list():
    """The exam face takes the protocol-passed session list (4 or 6)."""
    rng = np.random.default_rng(5)
    exam = plan.protocol_sessions("exp1_narrow")["exam"]
    assert len(exam) == 4
    outputs = {}
    for name in exam:
        target = rng.normal(size=(15, 2)).astype(np.float32)
        outputs[name] = (target, target)
    face = eval_local.evaluate_val_face(
        outputs, sessions=exam, face_role=plan.PROTOCOLS["exp1_narrow"]["face_role"]
    )
    assert face["equal_session_mean_r2"] == pytest.approx(1.0)
    assert face["exam_sessions"] == list(exam)
    assert face["val_face_role"] == plan.PROTOCOLS["exp1_narrow"]["face_role"]
    # a 6-session dict scored against the 4-session face must be refused
    six = {name: outputs[exam[0]] for name in plan.VAL_SESSIONS}
    with pytest.raises(RuntimeError, match="exactly 4"):
        eval_local.evaluate_val_face(six, sessions=exam)
    # formal-test names refused even inside the expected session list
    with pytest.raises(eval_local.TestSplitForbiddenError):
        eval_local.evaluate_val_face({}, sessions=plan.FORMAL_TEST_SESSIONS)
    # duplicate entries in the expected list refused
    with pytest.raises(RuntimeError, match="duplicate"):
        eval_local.evaluate_val_face(
            outputs, sessions=(exam[0], exam[0], exam[1], exam[2])
        )
    # default face (no sessions arg) stays the exp2_full 6-val-session face
    default_face = eval_local.evaluate_val_face(
        {name: outputs[exam[0]] for name in plan.VAL_SESSIONS}
    )
    assert default_face["exam_sessions"] == list(plan.VAL_SESSIONS)
    assert default_face["val_face_role"] == plan.VAL_FACE_ROLE


def test_prediction_digest_stability():
    a = np.zeros((3, 2), np.float32)
    b = np.zeros((3, 2), np.float32)
    c = np.zeros((3, 2), np.float64)
    assert eval_local.prediction_digest(a) == eval_local.prediction_digest(b)
    assert eval_local.prediction_digest(a) != eval_local.prediction_digest(c)  # dtype-sensitive
    assert eval_local.prediction_digest(a) != eval_local.prediction_digest(a.T.copy())


# --------------------------------------------------------------------------
# prepared-cache loader: happy path on a synthetic cache, test-split refusal
# --------------------------------------------------------------------------
def make_synthetic_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "prepared_cache_syn"
    (cache / "sessions").mkdir(parents=True)
    name = "sub-C_ses-CO-20151103"
    rng = np.random.default_rng(3)
    row = {
        "neural": rng.random((600, 8), np.float32),
        "behavior": rng.normal(size=(600, 2)).astype(np.float32),
        "starts": np.arange(0, 500, 5, dtype=np.int64),
        "e0": rng.normal(size=(8, plan.E0_DIM)).astype(np.float32),
        "carrier": rng.normal(size=(8, plan.CARRIER_DIM)).astype(np.float32),
        "mask": np.ones(8, bool),
    }
    np.savez_compressed(cache / "sessions" / f"{name}.npz", **row)
    meta = {
        "schema": plan.PREPARED_CACHE_SCHEMA,
        "manifest_sha256": plan.MANIFEST_SHA256,
        "split_counts": {"train": 27, "val": 6},
        "formal_test_used": False,
        "sessions": {name: {"split": "val"}},
        "rows_hashes": {name: {k: plan.array_digest(v) for k, v in row.items()}},
    }
    (cache / "prepared_contract.json").write_text(json.dumps(meta, indent=2))
    return cache


def test_load_session_happy_path_and_hash_tamper(tmp_path):
    cache = make_synthetic_cache(tmp_path)
    row = eval_local.load_session(cache, "sub-C_ses-CO-20151103")
    assert row["split"] == "val" and row["carrier"].shape == (8, 4)
    # tampered array (hash mismatch) must fail closed
    z_path = cache / "sessions" / "sub-C_ses-CO-20151103.npz"
    z = dict(np.load(z_path)); z["carrier"] = z["carrier"] + 1.0
    np.savez_compressed(z_path, **z)
    with pytest.raises(eval_local.PreparedCacheError, match="hash mismatch"):
        eval_local.load_session(cache, "sub-C_ses-CO-20151103")


def test_load_session_refuses_formal_test_and_unknown(tmp_path):
    cache = make_synthetic_cache(tmp_path)
    for name in plan.FORMAL_TEST_SESSIONS:
        with pytest.raises(eval_local.TestSplitForbiddenError, match="formal-test"):
            eval_local.load_session(cache, name)
    with pytest.raises(eval_local.PreparedCacheError, match="not present"):
        eval_local.load_session(cache, "sub-C_ses-CO-20131003")
    with pytest.raises(eval_local.PreparedCacheError, match="contract missing"):
        eval_local.load_session(tmp_path / "nope", "sub-C_ses-CO-20151103")


# --------------------------------------------------------------------------
# receipts
# --------------------------------------------------------------------------
def test_receipts_roundtrip_and_tamper_evidence(tmp_path):
    payload = {"schema": plan.SCHEMA + "_t", "values": [1, 2, 3], "nested": {"b": True, "a": None}}
    path = tmp_path / "sub" / "receipt.json"
    digest = receipts.seal_json(path, payload)
    assert len(digest) == 64
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    sidecar = receipts.sidecar_path(path)
    assert sidecar.read_text().split()[0] == digest
    assert receipts.read_sealed(path) == payload
    # re-seal replaces an existing 0444 seal
    receipts.seal_json(path, {"schema": "v2"})
    assert receipts.read_sealed(path) == {"schema": "v2"}
    # file tamper detected
    os.chmod(path, 0o644)
    data = path.read_bytes()
    path.write_bytes(data.replace(b'"v2"', b'"v3"'))
    os.chmod(path, 0o444)
    with pytest.raises(receipts.ReceiptSealError, match="hash mismatch"):
        receipts.read_sealed(path)
    # sidecar tamper detected
    receipts.seal_json(path, payload)
    sidecar.write_text("0" * 64 + "  receipt.json\n")
    with pytest.raises(receipts.ReceiptSealError):
        receipts.read_sealed(path)
    # missing file / sidecar
    with pytest.raises(receipts.ReceiptSealError, match="missing"):
        receipts.read_sealed(tmp_path / "absent.json")


# --------------------------------------------------------------------------
# support_resample stub
# --------------------------------------------------------------------------
def test_support_resample_stub():
    assert support_resample.SUPPORT_BUDGETS == ("M5", "M8", "M10")
    assert support_resample.SUPPORT_RESAMPLE_PHASE == "PHASE2_OPTIONAL_NOT_BLOCKING"
    carrier, mask = synthetic_carrier()
    with pytest.raises(ValueError):
        support_resample.resample_carrier_support("s", carrier, mask, "M7", 3)
    for budget in support_resample.SUPPORT_BUDGETS:
        with pytest.raises(NotImplementedError, match="Phase 2"):
            support_resample.resample_carrier_support("s", carrier, mask, budget, 3)
    for fn in (support_resample.m5, support_resample.m8, support_resample.m10):
        with pytest.raises(NotImplementedError):
            fn({}, 3)


# --------------------------------------------------------------------------
# preflight smoke: real prepared cache when on disk, synthetic bank otherwise
# --------------------------------------------------------------------------
def _synthetic_row(seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    t_len, n = 400, 12
    return {
        "split": "train",
        "neural": (rng.random((t_len, n)) < 0.2).astype(np.float32),
        "behavior": rng.normal(size=(t_len, 2)).astype(np.float32),
        "starts": np.arange(0, t_len - plan.WINDOW_BINS, 4, dtype=np.int64),
        "e0": rng.normal(size=(n, plan.E0_DIM)).astype(np.float32),
        "carrier": rng.normal(size=(n, plan.CARRIER_DIM)).astype(np.float32),
        "mask": np.ones(n, bool),
    }


def test_preflight_smoke_cpu():
    import run_688_bench as runner
    import torch

    assert not torch.cuda.is_initialized()
    cache = plan.prepared_cache_path()
    source_note = {"source": "synthetic"}
    if cache.is_dir() and (cache / "prepared_contract.json").is_file():
        # real read-only verification of the frozen cache under BOTH protocols
        meta = runner.verify_prepared_cache(cache)
        for protocol, n_expected in (("exp1_narrow", 13), ("exp2_full", 33)):
            rows = runner.load_rows(cache, meta, protocol)
            assert len(rows) == n_expected
            roles = [r["protocol_role"] for r in rows.values()]
            assert roles.count("train") == (9 if protocol == "exp1_narrow" else 27)
            assert roles.count("exam") == (4 if protocol == "exp1_narrow" else 6)
        # every arm smokes on the default protocol's (exp1_narrow) first train
        # session; the model code is protocol-independent, so one t4 smoke on
        # exp2_full keeps the full-cache path covered too
        for protocol in (plan.DEFAULT_PROTOCOL, "exp2_full"):
            rows = runner.load_rows(cache, meta, protocol)
            expected = 33 if protocol == "exp2_full" else 13
            arm_list = plan.ARMS if protocol == plan.DEFAULT_PROTOCOL else ("t4",)
            for arm in arm_list:
                armed, records = runner.apply_arm_to_rows(arm, rows)
                assert len(records) == expected
                assert all(r["verify"] == "PASSED" for r in records.values())
                first = next(n for n, r in sorted(armed.items()) if r["protocol_role"] == "train")
                smoke = runner.smoke_forward_backward(
                    armed[first], first, batch=2, fusion=arms.arm_fusion(arm)
                )
                assert smoke["forward_finite"] and smoke["gradients_finite"]
                assert smoke["stream_parity"]["max_abs"] <= 1e-5
                assert smoke["fusion"] == arms.arm_fusion(arm)
        source_note["source"] = "prepared_cache"

        # vstate-688 variant caches (skip when not built yet): each new arm
        # runs on its BOUND variant cache exactly the way the CLI preflight
        # does -- vstate/vstate_concat/z on the vstate cache (protocol-bound),
        # f_labelfree on the f_labelfree cache, equiv_zero (component
        # ablation) on the equiv_zero cache.
        vstate_cache = PKG_ROOT / "results" / "cache_vstate_exp1_narrow"
        f_cache = PKG_ROOT / "results" / "cache_f_labelfree"
        ez_cache = PKG_ROOT / "results" / "cache_equiv_zero"
        no_cache = PKG_ROOT / "results" / "cache_norm_only"
        for arm, cache in (("vstate", vstate_cache), ("vstate_concat", vstate_cache),
                           ("z_vstate_srcbank", vstate_cache),
                           ("f_labelfree", f_cache), ("equiv_zero", ez_cache),
                           ("norm_only", no_cache)):
            if not (cache / "prepared_contract.json").is_file():
                continue
            vmeta = runner.verify_prepared_cache(cache)
            binding = runner.assert_cache_variant_for_arm(vmeta, arm, "exp1_narrow")
            assert binding["arm"] == arm and binding["variant"] == vmeta["variant"]
            assert binding["allowed_variants"] == sorted(
                plan.ARM_REQUIRED_CACHE_VARIANT[arm])
            if arm not in ("f_labelfree", "equiv_zero"):
                # protocol-bound caches: vstate family + norm_only
                assert binding["protocol"] == "exp1_narrow"
            vrows = runner.load_rows(cache, vmeta, "exp1_narrow")
            varmed, vrecords = runner.apply_arm_to_rows(arm, vrows, protocol="exp1_narrow")
            assert all(r["verify"] == "PASSED" for r in vrecords.values())
            if arm == "z_vstate_srcbank":
                for exam in plan.EXP1_NARROW_EXAM_SESSIONS:
                    rec = vrecords[exam]
                    assert rec["z_srcbank"]["source_session"] == "sub-C_ses-CO-20150710"
                    assert rec["z_srcbank"]["exam_date_after_source"] is True
            if arm in ("f_labelfree", "equiv_zero", "norm_only"):
                assert all(not np.any(r["carrier"]) for r in varmed.values())
            if arm == "norm_only":
                # the broadcast rate-deviation identity: nonzero E0 on real
                # rows, exactly zero on the padded rows, and the identity
                # bytes stay untouched by the arm transform (e0 action
                # "identity" -- the identity lives in the cache)
                for name, row in varmed.items():
                    n_real = int(row["mask"].sum())
                    e0 = row["e0"]
                    assert np.any(e0[:n_real])
                    assert not np.any(e0[n_real:])
                    assert np.array_equal(e0, vrows[name]["e0"])
                    assert vrecords[name]["e0_action"] == "identity"
                no_meta = json.loads((cache / "prepared_contract.json").read_text())
                assert no_meta["variant"] == "norm_only"
                assert no_meta["estimator"]["protocol"] == "exp1_narrow"
                assert no_meta["estimator"]["source_sessions"] == sorted(
                    plan.EXP1_NARROW_TRAIN_SESSIONS)
            if arm == "equiv_zero":
                # the random-projection melt: E0 nonzero (activity flows) and
                # the projection receipt is baked into the cache contract
                assert all(np.any(r["e0"][: int(r["mask"].sum())]) for r in varmed.values())
                ez_meta = json.loads((cache / "prepared_contract.json").read_text())
                assert ez_meta["variant"] == "equiv_zero"
                assert ez_meta["estimator"]["param_parity"] is True
                assert ez_meta["estimator"]["trainable"] is False
                assert ez_meta["estimator"]["seed"] == plan.SEED
            first = next(n for n, r in sorted(varmed.items()) if r["protocol_role"] == "train")
            smoke = runner.smoke_forward_backward(
                varmed[first], first, batch=2, fusion=arms.arm_fusion(arm)
            )
            assert smoke["forward_finite"] and smoke["gradients_finite"]
            assert smoke["stream_parity"]["max_abs"] <= 1e-5
            assert smoke["fusion"] == arms.arm_fusion(arm)
            source_note["variant_caches"] = True
    else:
        # synthetic bank smoke (cache not on disk); arms still exercised
        row = _synthetic_row()
        for arm in plan.ARMS:
            armed_carrier = arms.apply_arm(arm, "synthetic-session", row["carrier"], row["mask"])
            armed_row = dict(row); armed_row["carrier"] = armed_carrier
            smoke = runner.smoke_forward_backward(
                armed_row, "synthetic-session", batch=2, fusion=arms.arm_fusion(arm)
            )
            assert smoke["forward_finite"] and smoke["gradients_finite"]
            assert smoke["stream_parity"]["max_abs"] <= 1e-5
    print(f"[preflight-smoke] source={source_note['source']}")
    assert not torch.cuda.is_initialized()


def test_t4_concat_model_geometry_and_matched_init():
    """Fusion-axis construction contract (arm t4_concat vs t4, same units)."""
    import run_688_bench as runner
    import torch

    device = torch.device("cpu")
    n_units = _synthetic_row()["neural"].shape[1]
    proj_add = runner.build_model(n_units, torch, device, fusion="proj_add")
    concat = runner.build_model(n_units, torch, device, fusion="concat")

    # proj_add (t4/f0/ts4): token_in 20, P: 50->16 added onto the local block
    assert proj_add.frontend.token_in == 16 + 4 == 20
    assert proj_add.frontend.e0_proj is not None
    assert tuple(proj_add.frontend.e0_proj.weight.shape) == (16, plan.E0_DIM)
    assert tuple(proj_add.frontend.token_mlp[0].weight.shape) == (256, 20)
    # concat (t4_concat): token_in 70 = 16 + E0 50 + carrier 4, no e0_proj
    assert concat.frontend.token_in == plan.CONCAT_TOKEN_IN == 70
    assert concat.frontend.e0_proj is None
    assert tuple(concat.frontend.token_mlp[0].weight.shape) == (256, 70)

    # parameter arithmetic: concat widens token_mlp[0] 20->256 into 70->256
    # (+256*50) and drops e0_proj P 50->16 bias-free (-16*50): net +12000.
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(concat) - count(proj_add) == 12000
    frontend_count = lambda m: sum(p.numel() for p in m.frontend.parameters())  # noqa: E731
    assert frontend_count(concat) - frontend_count(proj_add) == 12000

    # matched construction (concat_model fold): at init both fusion modes are
    # the same function — the concat first layer is [Wl | Wl@P | Wc].
    # eval mode: whole-unit dropout (p=0.10) is a train-time augmentation and
    # would draw independent masks for the two forward passes.
    row = _synthetic_row()
    x, _, v = runner.windows(row, np.asarray(row["starts"][:2]))
    tx = torch.from_numpy(x)
    tv = torch.from_numpy(v)
    bank = runner.make_bank("synthetic-session", row, torch)
    proj_add.eval()
    concat.eval()
    with torch.inference_mode():
        a = proj_add.forward_scores(tx, bank, input_valid_mask=tv)[0]
        b = concat.forward_scores(tx, bank, input_valid_mask=tv)[0]
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)
    assert not torch.cuda.is_initialized()


def test_parser_accepts_device_and_resume():
    import run_688_bench as runner

    parser = runner.build_parser()
    assert parser.get_default("device") == "cuda:0"
    device_action = next(a for a in parser._actions if "--device" in a.option_strings)
    resume_action = next(a for a in parser._actions if "--resume" in a.option_strings)
    assert device_action.default == "cuda:0"
    assert resume_action.default is None
    args = parser.parse_args([
        "--stage", "train", "--arm", "t4",
        "--dest", "btransform_unified_v2/dandi688_bench_v1/results/x",
        "--device", "cuda:0",
        "--resume", "btransform_unified_v2/dandi688_bench_v1/results/x/epoch_003.pt",
    ])
    assert args.device == "cuda:0"
    assert args.resume.name == "epoch_003.pt"


def test_bench_contract_digest_stable_train_vs_score():
    import run_688_bench as runner

    meta = {
        "schema": plan.PREPARED_CACHE_SCHEMA,
        "manifest_sha256": plan.MANIFEST_SHA256,
        "split_counts": {"train": 27, "val": 6},
        "formal_test_used": False,
    }
    armed_rows = {
        "sub-C_ses-CO-20150629": {
            "neural": np.zeros((10, 4), np.float32),
            "protocol_role": "train",
        }
    }
    verify_records = {
        "sub-C_ses-CO-20150629": {"carrier_sha256_after": "abc"},
    }
    train_c = runner.make_bench_contract(
        "t4", meta, armed_rows, verify_records, 12, "exp1_narrow", status="FORMAL_TRAIN",
    )
    score_c = runner.make_bench_contract(
        "t4", meta, armed_rows, verify_records, 12, "exp1_narrow", status="FORMAL_TRAIN",
    )
    assert train_c["status"] == score_c["status"] == "FORMAL_TRAIN"
    assert train_c == score_c
    assert runner.bench_contract_digest(train_c) == runner.bench_contract_digest(score_c)


def test_score_stage_requires_formal_artifacts(monkeypatch):
    import run_688_bench as runner

    bench_results = plan.workspace_root() / "btransform_unified_v2/dandi688_bench_v1/results"
    dest = bench_results / "test_score_missing_artifacts"
    monkeypatch.setattr(sys, "argv", [
        "run_688_bench.py", "--stage", "score", "--arm", "f0", "--dest", str(dest),
    ])
    with pytest.raises(RuntimeError, match="score requires a completed formal train destination"):
        runner.main()
    assert not dest.exists()  # nothing written


def test_runner_dest_must_stay_inside_package(monkeypatch, tmp_path):
    import run_688_bench as runner

    monkeypatch.setattr(sys, "argv", [
        "run_688_bench.py", "--stage", "preflight", "--arm", "t4",
        "--dest", str(tmp_path / "outside"),
    ])
    with pytest.raises(RuntimeError, match="--dest must live under"):
        runner.main()


# --------------------------------------------------------------------------
# vstate-688 series (user directive 2026-09-09)
# --------------------------------------------------------------------------
def test_vstate_recipe_constants():
    """M2 vstate4 recipe constants frozen for the 688 adaptation."""
    assert plan.VSTATE_BLOCK_SECONDS == 0.1           # 100 ms blocks
    assert plan.VSTATE_N0_BLOCKS == 10.0              # 10-block shrinkage
    assert plan.VSTATE_R700_SECONDS == 0.7            # -> 7 blocks
    assert plan.VSTATE_H300_SECONDS == 0.3            # -> 3 blocks
    assert plan.VSTATE_STATE_COLUMNS == ("+x", "-x", "+y", "-y")
    assert plan.VSTATE_VELOCITY_SUBBINS == 5 and plan.VSTATE_VELOCITY_BIN_SECONDS == 0.02
    assert 5 * 0.02 == plan.VSTATE_BLOCK_SECONDS      # sub-bins tile the block
    assert plan.VSTATE_SUPPORT_NAMESPACE == "candidate"  # SAME M10 set as t4
    assert plan.VSTATE_SUPPORT_POSITIONS == tuple(range(10))
    assert plan.VSTATE_HOLD == "signed_state_h300_same_affine_n0_10"
    # fourth-column modes + vstate_full support face (alignment matrix rows
    # "readout" and "support"; user ruling 2026-09-09 final: main = mean_k)
    assert plan.VSTATE_B_MODES == ("mean_k", "hold_diff")
    assert plan.VSTATE_B_MODE_MAIN == "mean_k"
    assert plan.VSTATE_FULL_SUPPORT_NAMESPACE == "reliability_audit"
    assert plan.VSTATE_FULL_SUPPORT_POSITIONS == tuple(range(30))
    assert plan.VSTATE_VARIANTS == ("vstate", "vstate_full", "vstate_b_hold")


def test_block_edges_rate_primitive_hand_computed():
    from dandi688_bench_v1 import vstate

    # R700 -> exactly 7 tiling edges over [start, stop), exact endpoints
    edges = vstate.block_edges(1.0, 1.7)
    assert edges.shape == (8,)
    assert edges[0] == 1.0 and edges[-1] == 1.7  # exact: half-open endpoint law
    assert np.allclose(np.diff(edges), 0.1)
    # a window that is not an integer number of blocks fails closed
    with pytest.raises(ValueError, match="integer multiple"):
        vstate.block_edges(0.0, 0.75)
    with pytest.raises(ValueError, match="non-degenerate"):
        vstate.block_edges(1.0, 1.0)

    # half-open [edge_k, edge_{k+1}) counting / block duration (Hz): spikes
    # at 1.05/1.15 fall in blocks 0/1; a spike at an interior edge (1.3)
    # belongs to the RIGHT block (left-inclusive, right-exclusive); a spike
    # exactly at the final edge 1.7 is EXCLUDED.
    spikes = [np.asarray([1.05, 1.15, 1.3, 1.65, 1.7])]
    rates = vstate.block_rate_matrix(spikes, edges)
    assert rates.shape == (7, 1)
    assert rates[:, 0].tolist() == [10.0, 10.0, 0.0, 10.0, 0.0, 0.0, 10.0]
    # single spike at ~1.3: counted exactly once, in the block whose float
    # edge pair contains it (linspace interior edge is 1.2999..., so 1.3
    # lands in block 3); no double counting across adjacent blocks
    spikes2 = [np.asarray([1.3])]
    assert vstate.block_rate_matrix(spikes2, edges)[:, 0].tolist() == [0, 0, 0, 10, 0, 0, 0]
    # unsorted spike trains are refused
    with pytest.raises(ValueError, match="non-decreasing"):
        vstate.block_rate_matrix([np.asarray([1.5, 1.0])], edges)


def test_block_velocity_means_hand_computed():
    from dandi688_bench_v1 import vstate

    edges = vstate.block_edges(0.0, 0.4)
    # linear velocity v(t) = (1 + t, 4t): every 20 ms bin center interpolates
    # exactly, so the block mean equals the value at the block center.
    t = np.asarray([0.0, 0.4])
    v = np.stack([1.0 + t, 4.0 * t], axis=1)
    blocks = vstate.block_velocity_means(t, v, edges)
    assert blocks.shape == (4, 2)
    centers = np.asarray([0.05, 0.15, 0.25, 0.35])
    assert np.allclose(blocks[:, 0], 1.0 + centers)
    assert np.allclose(blocks[:, 1], 4.0 * centers)
    # out-of-range bin centers read 0.0 (datamodule fill_value semantics)
    t_short = np.asarray([0.0, 0.1])
    v_short = np.stack([np.full(2, 5.0), np.zeros(2)], axis=1)
    blocks_short = vstate.block_velocity_means(t_short, v_short, edges)
    # block 0 centers lie inside [0, 0.1]; blocks 1..3 centers read 0
    assert np.allclose(blocks_short[0], (5.0, 0.0))
    assert not blocks_short[1:].any()
    # sub-bins must tile the block exactly
    with pytest.raises(ValueError, match="tile"):
        vstate.block_velocity_means(t, v, edges, subbins=3)


def _softplus(x):
    return float(np.logaddexp(0.0, x))


def test_vstate_carrier_hand_computed():
    """Independent loop-based recomputation of the whole estimator."""
    from dandi688_bench_v1 import vstate

    # 1 unit, 7 R700 blocks + 3 H300 blocks; hand-picked rates/velocities
    r700_rates = np.asarray([[10.0], [20.0], [30.0], [10.0], [20.0], [10.0], [10.0]])
    r700_vel = np.asarray([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.5], [-2.0, -0.5],
                           [0.0, 1.0], [0.5, -1.0], [-0.5, 2.0]])
    h300_rates = np.asarray([[5.0], [10.0], [8.0]])
    h300_vel = np.asarray([[0.0, 0.0], [0.2, -0.1], [-0.1, 0.2]])
    rms = np.asarray([1.5, 2.0])
    n0 = 10.0
    delta = 0.1

    got = vstate.vstate_carrier_from_blocks(
        r700_rates, r700_vel, h300_rates, h300_vel, rms, n0=n0
    )
    assert got.shape == (1, 4)

    # hand computation: Poisson standardization on R700, same affine on H300
    mean = r700_rates[:, 0].mean()
    sigma = np.sqrt(max(mean * delta, 1.0)) / delta
    z_r = (r700_rates[:, 0] - mean) / sigma
    z_h = (h300_rates[:, 0] - mean) / sigma

    def weights(vel):
        rows = []
        for vx, vy in vel:
            rows.append([_softplus(vx / rms[0]), _softplus(-vx / rms[0]),
                         _softplus(vy / rms[1]), _softplus(-vy / rms[1])])
        return np.asarray(rows)

    def cond_resp(z, w):
        out = []
        for k in range(4):
            out.append(float((w[:, k] * z).sum() / (w[:, k].sum() + n0)))
        return np.asarray(out)

    R = cond_resp(z_r, weights(r700_vel))
    R_hold = cond_resp(z_h, weights(h300_vel))
    a = R[0] - R[1]
    c = R[2] - R[3]
    m = float(np.hypot(a, c))
    # MAIN variant (default b_mode="mean_k"): the M2-identical fourth column;
    # the H300 inputs are ignored entirely (user ruling 2026-09-09 final)
    assert got[0].tolist() == pytest.approx(
        [a, c, m, float(R.mean())], rel=1e-12, abs=1e-12
    )
    # b_mode="hold_diff" keeps the original delta_b semantics and leaves
    # a/c/m untouched, so the two modes isolate exactly the fourth column
    got_hold = vstate.vstate_carrier_from_blocks(
        r700_rates, r700_vel, h300_rates, h300_vel, rms, n0=n0,
        b_mode="hold_diff",
    )
    assert got_hold[0].tolist() == pytest.approx(
        [a, c, m, float(R.mean() - R_hold.mean())], rel=1e-12, abs=1e-12
    )
    assert np.array_equal(got_hold[:, :3], got[:, :3])

    # sign semantics: a unit firing only on +x blocks reads a > 0, and the
    # same construction on the y axis reads c > 0
    hi, lo = 40.0, 2.0
    base_vel = np.asarray([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0],
                           [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    rates_x = np.asarray([[hi], [lo], [hi], [lo], [lo], [lo], [lo]])
    got_x = vstate.vstate_carrier_from_blocks(
        rates_x, base_vel, np.full((3, 1), lo), np.zeros((3, 2)), np.ones(2)
    )
    assert got_x[0, 0] > 0 and got_x[0, 2] > 0  # a > 0, m > 0
    vel_y = base_vel[:, ::-1]  # swap axes: same design on y
    got_y = vstate.vstate_carrier_from_blocks(
        rates_x, vel_y, np.full((3, 1), lo), np.zeros((3, 2)), np.ones(2)
    )
    assert got_y[0, 1] > 0  # c > 0
    # a perfectly symmetric design reads a == c == 0
    rates_flat = np.full((7, 1), 10.0)
    got_sym = vstate.vstate_carrier_from_blocks(
        rates_flat, base_vel, np.full((3, 1), 10.0), np.zeros((3, 2)), np.ones(2)
    )
    assert abs(got_sym[0, 0]) < 1e-12 and abs(got_sym[0, 1]) < 1e-12
    # b_mode="hold_diff" keeps the delta_b motion-minus-hold sign: higher
    # hold than motion rates -> b < 0 (the main mean_k mode ignores hold)
    got_b = vstate.vstate_carrier_from_blocks(
        np.full((7, 1), 5.0), base_vel, np.full((3, 1), 50.0), np.zeros((3, 2)),
        np.ones(2), b_mode="hold_diff",
    )
    assert got_b[0, 3] < 0
    got_b_main = vstate.vstate_carrier_from_blocks(
        np.full((7, 1), 5.0), base_vel, None, None, np.ones(2)
    )
    assert got_b_main[0, 3] == pytest.approx(0.0, abs=1e-12)  # flat rates -> z=0
    # hold_diff without H300 inputs fails closed; unknown modes are refused
    with pytest.raises(ValueError, match="hold_diff"):
        vstate.vstate_carrier_from_blocks(
            np.full((7, 1), 5.0), base_vel, None, None, np.ones(2),
            b_mode="hold_diff",
        )
    with pytest.raises(ValueError, match="b_mode"):
        vstate.vstate_carrier_from_blocks(
            np.full((7, 1), 5.0), base_vel, None, None, np.ones(2),
            b_mode="delta_b",
        )


def test_fit_velocity_rms_train_only():
    from dandi688_bench_v1 import vstate

    train_a = np.asarray([[3.0, 0.0], [0.0, 4.0]])
    train_b = np.asarray([[0.0, 0.0]])
    rms = vstate.fit_velocity_rms([train_a, train_b])
    assert rms.shape == (2,)
    # rms = sqrt(mean of squares over the GIVEN blocks only)
    assert rms.tolist() == pytest.approx([np.sqrt(3.0), np.sqrt(16.0 / 3.0)])
    with pytest.raises(ValueError):
        vstate.fit_velocity_rms([np.zeros((0, 2))])


class _MockIdEncoder:
    """Tiny stand-in for the frozen B3S id_encoder: records the side input."""

    def __init__(self, n_units: int, trial_bins: int):
        import torch
        from torch import nn

        self.torch = torch
        self.pre_pool = nn.Linear(trial_bins, 8)      # acts on the time axis
        self.post_pool = nn.Linear(8 + plan.CARRIER_DIM, plan.E0_DIM)
        self.seen_sides = []

    def __call__(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("id_encoder is used via pre_pool/post_pool only")


class _MockStudent:
    def __init__(self, n_units: int, trial_bins: int):
        self.id_encoder = _MockIdEncoder(n_units, trial_bins)
        base_post = self.id_encoder.post_pool

        def post_pool(x):
            self.id_encoder.seen_sides.append(x[..., -plan.CARRIER_DIM:].clone())
            return base_post(x)

        self.id_encoder.post_pool = post_pool


def test_remelt_e0_consumes_side_input():
    """E0 co-variance law: the side input of the remelt IS the passed carrier
    rows; different sides melt different E0s; padding rows stay zero."""
    from dandi688_bench_v1 import vstate

    n_units, trial_bins, n_pad = 6, 40, 9
    rng = np.random.default_rng(17)
    calib = rng.random((30, trial_bins, n_units)).astype(np.float32)
    student = _MockStudent(n_units, trial_bins)
    side_vstate = rng.normal(size=(n_units, plan.CARRIER_DIM)).astype(np.float32)
    side_zero = np.zeros((n_units, plan.CARRIER_DIM), np.float32)
    side_t4 = rng.normal(size=(n_units, plan.CARRIER_DIM)).astype(np.float32)

    e0_vstate = vstate.remelt_e0(student, calib, side_vstate, n_pad)
    e0_zero = vstate.remelt_e0(student, calib, side_zero, n_pad)
    e0_t4 = vstate.remelt_e0(student, calib, side_t4, n_pad)

    assert e0_vstate.shape == e0_zero.shape == (n_pad, plan.E0_DIM)
    assert not np.any(e0_vstate[n_units:])  # padding rows exactly zero
    # the recorded side inputs are byte-equal to what the caller passed
    seen = student.id_encoder.seen_sides
    assert len(seen) == 3
    assert np.array_equal(seen[0].squeeze(0).numpy(), side_vstate)
    assert not np.any(seen[1].numpy())
    assert np.array_equal(seen[2].squeeze(0).numpy(), side_t4)
    # co-variance: distinct sides melt distinct identities
    assert not np.array_equal(e0_vstate, e0_zero)
    assert not np.array_equal(e0_vstate, e0_t4)
    # shape law: side rows must match the real unit count
    with pytest.raises(ValueError, match="side_rows"):
        vstate.remelt_e0(student, calib, side_vstate[:-1], n_pad)
    with pytest.raises(ValueError, match="n_pad"):
        vstate.remelt_e0(student, calib, side_vstate, n_units - 1)


def test_z_srcbank_map_law():
    """Frozen nearest-date tables: exact faces, train-only sources, strictly
    earlier dates, and equality with the recomputed rule."""
    resolved = plan.verify_protocol_definitions()
    for protocol in plan.PROTOCOLS:
        table = plan.z_srcbank_map(protocol)
        train, exam = resolved[protocol]["train"], resolved[protocol]["exam"]
        assert set(table) == set(exam)
        for exam_name, src in table.items():
            assert src in train
            assert plan.session_date(exam_name) > plan.session_date(src)
            assert plan.nearest_train_session(exam_name, train) == src
    # exp1 degeneracy disclosed: all four exam sessions map to the LAST
    # train session (the honest "carry the most recent old bank" case)
    assert set(plan.z_srcbank_map("exp1_narrow").values()) == {"sub-C_ses-CO-20150710"}
    # unknown protocol refused
    with pytest.raises(ValueError, match="unknown protocol"):
        plan.z_srcbank_map("exp3_half")
    # tampered tables fail the law: a later-dated source is leakage (checked
    # with inverted roles, since every legal exp1 source predates every exam)
    with pytest.raises(RuntimeError, match="strictly"):
        plan._check_z_srcbank_table(
            "exp1_narrow",
            {"sub-C_ses-CO-20150629": "sub-C_ses-CO-20150713"},
            tuple(plan.EXP1_NARROW_EXAM_SESSIONS),          # pretend train face
            ("sub-C_ses-CO-20150629",),                     # pretend exam face
        )
    # tampered tables fail the law: a non-nearest mapping is refused
    with pytest.raises(RuntimeError, match="nearest-date rule"):
        plan._check_z_srcbank_table(
            "exp1_narrow",
            {**plan.z_srcbank_map("exp1_narrow"),
             "sub-C_ses-CO-20150713": "sub-C_ses-CO-20150629"},
            tuple(plan.EXP1_NARROW_TRAIN_SESSIONS),
            tuple(plan.EXP1_NARROW_EXAM_SESSIONS),
        )


def _z_synthetic_rows():
    def row(n_real, n_pad, seed):
        r = np.random.default_rng(seed)
        return {
            "protocol_role": None,
            "neural": r.random((50, n_real + n_pad), np.float32),
            "behavior": r.random((50, 2), np.float32),
            "starts": np.arange(0, 8, dtype=np.int64),
            "e0": r.normal(size=(n_real + n_pad, plan.E0_DIM)).astype(np.float32),
            "carrier": r.normal(size=(n_real + n_pad, plan.CARRIER_DIM)).astype(np.float32),
            "mask": np.array([True] * n_real + [False] * n_pad),
        }
    rows = {}
    for i, name in enumerate(plan.EXP1_NARROW_TRAIN_SESSIONS):
        rows[name] = row(6, 2, 100 + i)
        rows[name]["protocol_role"] = "train"
    src = "sub-C_ses-CO-20150710"  # the frozen z-source (last train session)
    for j, (name, n_real, n_pad) in enumerate((
            ("sub-C_ses-CO-20150713", 5, 1),
            ("sub-C_ses-CO-20150714", 8, 2))):
        rows[name] = row(n_real, n_pad, 200 + j)
        rows[name]["protocol_role"] = "exam"
    return rows, src


def test_z_srcbank_swap_on_synthetic_rows():
    """z_vstate_srcbank load-time law: exam banks replaced row-for-row by the
    frozen nearest-date train bank; train rows untouched; nothing mutated."""
    import run_688_bench as runner

    rows, src = _z_synthetic_rows()
    before = {n: (r["carrier"].copy(), r["e0"].copy()) for n, r in rows.items()}
    armed, records = runner.apply_arm_to_rows(
        "z_vstate_srcbank", rows, protocol="exp1_narrow"
    )
    # inputs never mutated
    for name, (carrier, e0) in before.items():
        assert np.array_equal(rows[name]["carrier"], carrier)
        assert np.array_equal(rows[name]["e0"], e0)
    # train rows (incl. the source) keep their own banks
    assert np.array_equal(armed[src]["carrier"], before[src][0])
    assert np.array_equal(armed[src]["e0"], before[src][1])
    # each loaded exam row: bank = first k rows of the source bank, rest zero
    for exam in ("sub-C_ses-CO-20150713", "sub-C_ses-CO-20150714"):
        src_real = int(armed[src]["mask"].sum())
        k = min(armed[exam]["mask"].size, src_real)
        assert np.array_equal(armed[exam]["carrier"][:k], before[src][0][:k])
        assert np.array_equal(armed[exam]["e0"][:k], before[src][1][:k])
        assert not np.any(armed[exam]["carrier"][k:])
        assert not np.any(armed[exam]["e0"][k:])
        rec = records[exam]["z_srcbank"]
        assert rec["source_session"] == src
        assert rec["rule"] == plan.Z_SRCBANK_RULE
        assert rec["rows_copied"] == k
        assert rec["exam_date_after_source"] is True
        assert rec["carrier_sha256_source"] == plan.array_digest(before[src][0])
        assert records[exam]["carrier_sha256_after"] == plan.array_digest(armed[exam]["carrier"])
    # a non-z arm never swaps
    armed_t4, records_t4 = runner.apply_arm_to_rows("vstate", rows, protocol="exp1_narrow")
    assert all("z_srcbank" not in r for r in records_t4.values())
    for exam in ("sub-C_ses-CO-20150713", "sub-C_ses-CO-20150714"):
        assert np.array_equal(armed_t4[exam]["carrier"], before[exam][0])


def test_assert_cache_variant_for_arm():
    import run_688_bench as runner

    family = sorted(plan.VSTATE_VARIANTS)

    def meta(variant=None, protocol=None):
        return {"variant": variant, "estimator": {"protocol": protocol}}

    # the five original arms are cache-agnostic
    assert runner.assert_cache_variant_for_arm(meta(), "t4", "exp1_narrow") is None
    # vstate/vstate_concat accept every vstate-family variant cache (with the
    # active-protocol binding); z_vstate_srcbank takes the MAIN variant only
    assert runner.assert_cache_variant_for_arm(
        meta("vstate", "exp1_narrow"), "vstate", "exp1_narrow"
    ) == {"arm": "vstate", "variant": "vstate",
          "allowed_variants": family, "protocol": "exp1_narrow"}
    assert runner.assert_cache_variant_for_arm(
        meta("vstate_full", "exp1_narrow"), "vstate_concat", "exp1_narrow"
    ) == {"arm": "vstate_concat", "variant": "vstate_full",
          "allowed_variants": family, "protocol": "exp1_narrow"}
    assert runner.assert_cache_variant_for_arm(
        meta("vstate_b_hold", "exp1_narrow"), "vstate", "exp1_narrow"
    ) == {"arm": "vstate", "variant": "vstate_b_hold",
          "allowed_variants": family, "protocol": "exp1_narrow"}
    assert runner.assert_cache_variant_for_arm(
        meta("vstate", "exp1_narrow"), "z_vstate_srcbank", "exp1_narrow"
    ) == {"arm": "z_vstate_srcbank", "variant": "vstate",
          "allowed_variants": ["vstate"], "protocol": "exp1_narrow"}
    # f_labelfree binds to its variant but carries no protocol
    assert runner.assert_cache_variant_for_arm(
        meta("f_labelfree"), "f_labelfree", "exp2_full"
    ) == {"arm": "f_labelfree", "variant": "f_labelfree",
          "allowed_variants": ["f_labelfree"]}
    # equiv_zero binds to its own variant cache (protocol-free)
    assert runner.assert_cache_variant_for_arm(
        meta("equiv_zero"), "equiv_zero", "exp2_full"
    ) == {"arm": "equiv_zero", "variant": "equiv_zero",
          "allowed_variants": ["equiv_zero"]}
    # norm_only binds to its own variant cache AND to the protocol it was
    # built for (its source rate distribution is fit on that protocol's
    # train sessions)
    assert runner.assert_cache_variant_for_arm(
        meta("norm_only", "exp1_narrow"), "norm_only", "exp1_narrow"
    ) == {"arm": "norm_only", "variant": "norm_only",
          "allowed_variants": ["norm_only"], "protocol": "exp1_narrow"}
    # label-free + protocol-free: f_labelfree/equiv_zero accept any protocol
    for arm in ("f_labelfree", "equiv_zero"):
        assert "protocol" not in runner.assert_cache_variant_for_arm(
            meta(arm), arm, "exp2015_full")
    # wrong variant / wrong protocol fail closed
    with pytest.raises(RuntimeError, match="variant caches"):
        runner.assert_cache_variant_for_arm(meta("u1_m10"), "vstate", "exp1_narrow")
    with pytest.raises(RuntimeError, match="variant caches"):
        runner.assert_cache_variant_for_arm(meta(), "f_labelfree", "exp1_narrow")
    with pytest.raises(RuntimeError, match="variant caches"):
        runner.assert_cache_variant_for_arm(
            meta("vstate_full"), "z_vstate_srcbank", "exp1_narrow")
    with pytest.raises(RuntimeError, match="variant caches"):
        runner.assert_cache_variant_for_arm(meta(), "equiv_zero", "exp1_narrow")
    with pytest.raises(RuntimeError, match="variant caches"):
        runner.assert_cache_variant_for_arm(meta(), "norm_only", "exp1_narrow")
    # norm_only on the f_labelfree cache fails closed (wrong identity)
    with pytest.raises(RuntimeError, match="variant caches"):
        runner.assert_cache_variant_for_arm(meta("f_labelfree"), "norm_only", "exp1_narrow")
    with pytest.raises(RuntimeError, match="built for that protocol"):
        runner.assert_cache_variant_for_arm(meta("vstate", "exp2_full"), "vstate", "exp1_narrow")
    with pytest.raises(RuntimeError, match="built for that protocol"):
        runner.assert_cache_variant_for_arm(
            meta("norm_only", "exp2_full"), "norm_only", "exp1_narrow")


def test_vstate_gate_report_arithmetic():
    good = plan.vstate_gate_report(0.60, 0.55, 0.50, 0.40)
    assert good["r2"] == {"vstate": 0.60, "t4": 0.55,
                          "z_vstate_srcbank": 0.50, "f_labelfree": 0.40}
    assert good["gate_vstate_minus_t4"]["delta"] == pytest.approx(0.05)
    assert good["gate_vstate_minus_t4"]["pass"]
    assert good["ablation_calibration_cost"]["z_vstate_srcbank_minus_vstate"] == pytest.approx(-0.10)
    assert good["ablation_label_information"]["vstate_minus_f_labelfree"] == pytest.approx(0.20)
    bad = plan.vstate_gate_report(0.56, 0.55, 0.50, 0.40)
    assert not bad["gate_vstate_minus_t4"]["pass"]
    # the legacy three-arm gate arithmetic is unchanged by the new series
    assert set(plan.gate_report(0.6, 0.55, 0.55)["r2"]) == {"t4", "f0", "ts4"}


def test_parser_accepts_vstate_series_arms():
    import run_688_bench as runner

    parser = runner.build_parser()
    arm_action = next(a for a in parser._actions if "--arm" in a.option_strings)
    assert set(arm_action.choices) == set(plan.ARMS)
    for arm in ("vstate", "z_vstate_srcbank", "f_labelfree"):
        args = parser.parse_args([
            "--stage", "preflight", "--arm", arm, "--protocol", "exp1_narrow",
            "--dest", "btransform_unified_v2/dandi688_bench_v1/results/x",
        ])
        assert args.arm == arm


def test_variant_cache_e0_covariance_real():
    """E0 co-variance on the real caches (skip when not built): the vstate
    cache's E0 differs from the frozen t4 cache's E0, the f_labelfree cache's
    E0 is the activity-only melt, neural bytes are identical, and the vstate
    contract carries the protocol + train-only rms."""
    vstate_cache = PKG_ROOT / "results" / "cache_vstate_exp1_narrow"
    f_cache = PKG_ROOT / "results" / "cache_f_labelfree"
    if not ((vstate_cache / "prepared_contract.json").is_file()
            and (f_cache / "prepared_contract.json").is_file()
            and plan.prepared_cache_path().is_dir()):
        pytest.skip("variant caches not built yet")
    name = plan.EXP1_NARROW_TRAIN_SESSIONS[0]
    frozen = eval_local.load_session(plan.prepared_cache_path(), name)
    vrow = eval_local.load_session(vstate_cache, name)
    frow = eval_local.load_session(f_cache, name)
    # neural/behavior/starts/mask copied byte-for-byte
    for key in ("neural", "behavior", "starts", "mask"):
        assert plan.array_digest(vrow[key]) == plan.array_digest(frozen[key])
        assert plan.array_digest(frow[key]) == plan.array_digest(frozen[key])
    # E0 co-variance: different melts differ; f_labelfree carrier is zero
    assert not np.array_equal(vrow["e0"], frozen["e0"])
    assert not np.array_equal(vrow["e0"], frow["e0"])
    assert not np.any(frow["carrier"])
    assert not np.array_equal(vrow["carrier"], frozen["carrier"])
    # contract laws: vstate records the protocol, train-only rms and the
    # M10-matched support; f_labelfree records the zero-side recipe
    vmeta = json.loads((vstate_cache / "prepared_contract.json").read_text())
    fmeta = json.loads((f_cache / "prepared_contract.json").read_text())
    assert vmeta["variant"] == "vstate"
    assert vmeta["estimator"]["protocol"] == "exp1_narrow"
    assert vmeta["estimator"]["rms_train_sessions"] == len(plan.EXP1_NARROW_TRAIN_SESSIONS) == 9
    assert vmeta["estimator"]["support_namespace"] == "candidate"
    assert vmeta["estimator"]["n0_blocks"] == 10.0
    assert len(vmeta["estimator"]["velocity_rms"]) == 2
    assert vmeta["estimator"]["b_mode"] == plan.VSTATE_B_MODE_MAIN == "mean_k"
    assert vmeta["estimator"]["hold"] is None  # main variant has no hold state
    assert fmeta["variant"] == "f_labelfree"
    assert "zero side" in fmeta["estimator"]["e0_side"]


# --------------------------------------------------------------------------
# component ablation (user clarification 2026-09-09; plan.COMPONENT_ABLATION)
# --------------------------------------------------------------------------
def test_component_ablation_ladder_frozen():
    """The frozen nested ladder (2026-09-09 final ruling, extended by the
    2026-09-10 NORM_ONLY zero-baseline guide): FULL=t4 > ACTIVITY_ONLY=
    f_labelfree > NORM_ONLY=norm_only > NONE=floor; NO carrier-only rung (z0
    and z_vstate_srcbank demoted to auxiliary); equiv_zero kept as the
    param-matched control; rung cells agree with ARM_SPECS; formulas
    reference legal arms only."""
    ca = plan.COMPONENT_ABLATION
    assert ca["ladder"] == {"NONE": "floor", "NORM_ONLY": "norm_only",
                            "ACTIVITY_ONLY": "f_labelfree", "FULL": "t4"}
    # the four-level ladder is the frozen ascending order, exactly
    assert plan.COMPONENT_LADDER_RUNGS == ("NONE", "NORM_ONLY", "ACTIVITY_ONLY",
                                           "FULL")
    assert plan.COMPONENT_LADDER_ORDER == ("floor", "norm_only", "f_labelfree",
                                           "t4")
    assert tuple(ca["ladder"]) == plan.COMPONENT_LADDER_RUNGS
    assert tuple(ca["ladder"].values()) == plan.COMPONENT_LADDER_ORDER
    assert tuple(ca["rungs"]) == plan.COMPONENT_LADDER_ORDER
    assert set(ca["auxiliary_arms"]) == {"z0", "z_vstate_srcbank"}
    assert ca["base_arm"] == "t4"
    assert "算 carrier" in ca["clarification"]  # nested-ladder ruling quote
    # rung/control/auxiliary cells agree with the load-time arm law
    specs = arms.ARM_SPECS
    assert specs["z0"]["carrier"] == "identity" and specs["z0"]["e0"] == "zero"
    assert specs["floor"]["carrier"] == "zero" and specs["floor"]["e0"] == "zero"
    assert specs["f_labelfree"]["e0"] == "identity"   # activity-only melt in cache bytes
    assert specs["equiv_zero"]["carrier"] == "zero" and specs["equiv_zero"]["e0"] == "identity"
    # NORM_ONLY rung: zero carrier + identity E0, identity in the cache bytes
    assert specs["norm_only"]["carrier"] == "zero" and specs["norm_only"]["e0"] == "identity"
    assert plan.ARM_REQUIRED_CACHE_VARIANT["norm_only"] == frozenset({"norm_only"})
    # NORM_ONLY sits directly above NONE and carries its own E0 pathway cell
    assert ca["rungs"]["norm_only"][0].startswith("per-unit pooled-rate deviation")
    assert "zero" == ca["rungs"]["norm_only"][1]
    assert ca["rungs"]["norm_only"][2] != ca["rungs"]["f_labelfree"][2]
    # preregistered decomposition: EXACTLY the 3-item ladder (unchanged)
    formulas = ca["formulas"]
    assert tuple(formulas) == ("activity_independent",
                               "carrier_label_increment", "calibration_total")
    assert formulas["activity_independent"] == "f_labelfree - floor"
    assert formulas["carrier_label_increment"] == "t4 - f_labelfree"
    assert formulas["calibration_total"] == "t4 - floor"
    # the four-level refinement formulas beside the frozen 3-item law
    assert ca["norm_only_formulas"] == {
        "rate_deviation_identity": "norm_only - floor",
        "encoder_activity_increment": "f_labelfree - norm_only",
        "label_free_activity_total": "f_labelfree - floor",
    }
    assert "SAME" in ca["norm_only_vs_f_labelfree"]
    # auxiliary formulas never reference a carrier-only rung as preregistered
    aux = ca["auxiliary_formulas"]
    assert aux["equiv_zero_pathway_value"] == "equiv_zero - floor"
    assert aux["equiv_zero_information_value"] == "t4 - equiv_zero"
    assert aux["z0_carrier_only_reading"] == "t4 - z0"
    assert aux["direct_carrier_marginal"] == "t4 - f0"
    assert "exp1_narrow" in ca["judged_on"] and "20150713" in ca["judged_on"]
    assert "ts4" in ca["secondary"]
    # f_labelfree vs floor distinction stated verbatim in the disclosure
    disclosure = ca["f_labelfree_vs_floor"]
    assert "keeps" in disclosure.lower() and "removes" in disclosure.lower()
    # frozen-cache component arms carry no variant binding
    assert plan.COMPONENT_ARMS_ON_FROZEN_CACHE == ("t4", "f0", "z0", "floor")
    for arm in plan.COMPONENT_ARMS_ON_FROZEN_CACHE:
        assert arm not in plan.ARM_REQUIRED_CACHE_VARIANT


def test_component_ablation_report_arithmetic():
    """Hand-computed nested-ladder decomposition + receipt fields."""
    report = plan.component_ablation_report(0.60, 0.44, 0.40,
                                            r2_equiv_zero=0.43, r2_z0=0.47,
                                            r2_f0=0.52, r2_ts4=0.50)
    assert report["schema"] == "dandi688_bench_v1_component_ablation"
    assert report["base_arm"] == "t4"
    assert report["ladder"] == plan.COMPONENT_ABLATION["ladder"]
    assert report["face"] == plan.COMPONENT_ABLATION["judged_on"]
    assert report["r2"] == {"t4": 0.60, "f_labelfree": 0.44, "floor": 0.40,
                            "norm_only": None,
                            "equiv_zero": 0.43, "z0": 0.47, "f0": 0.52,
                            "ts4": 0.50}
    assert report["ladder_order"] == list(plan.COMPONENT_LADDER_ORDER)
    assert report["norm_only_formulas"] == plan.COMPONENT_ABLATION["norm_only_formulas"]
    assert "zero_calibration_baseline" not in report  # rung reading omitted
    d = report["decomposition"]
    # the three preregistered ladder items
    assert tuple(d) == ("activity_independent", "carrier_label_increment",
                        "calibration_total")
    assert d["activity_independent"]["delta"] == pytest.approx(0.44 - 0.40)
    assert d["carrier_label_increment"]["delta"] == pytest.approx(0.60 - 0.44)
    assert d["calibration_total"]["delta"] == pytest.approx(0.60 - 0.40)
    # nested-ladder arithmetic law: the two increments sum to the total
    assert (d["activity_independent"]["delta"]
            + d["carrier_label_increment"]["delta"]) == pytest.approx(
                d["calibration_total"]["delta"])
    for key, entry in d.items():
        assert entry["formula"] == plan.COMPONENT_ABLATION["formulas"][key]
        assert isinstance(entry["reading"], str) and entry["reading"]
    # auxiliary controls: equiv_zero pathway/information, z0 exploratory
    # carrier-only, f0 direct marginal, ts4 content -- never in decomposition
    controls = report["controls"]
    assert controls["equiv_zero"]["pathway_value"]["delta"] == pytest.approx(0.43 - 0.40)
    assert controls["equiv_zero"]["information_value"]["delta"] == pytest.approx(0.60 - 0.43)
    assert controls["z0_carrier_only"]["delta"] == pytest.approx(0.60 - 0.47)
    assert controls["f0_direct_carrier_marginal"]["delta"] == pytest.approx(0.60 - 0.52)
    assert controls["ts4_content"]["delta"] == pytest.approx(0.60 - 0.50)
    assert "AUXILIARY" in controls["z0_carrier_only"]["reading"]
    assert report["formal_test_policy"] == plan.FORMAL_TEST_POLICY
    # the optional readings are absent when not passed
    minimal = plan.component_ablation_report(0.6, 0.4, 0.35)
    assert "controls" not in minimal
    assert minimal["r2"]["equiv_zero"] is None and minimal["r2"]["z0"] is None
    assert minimal["r2"]["f0"] is None and minimal["r2"]["ts4"] is None
    assert minimal["r2"]["norm_only"] is None


def test_arm_floor_and_z0_rows_component_cells():
    """Load-time law of the component 2x2 cells on synthetic rows:
    floor = E0 zero + carrier zero; z0 = E0 zero + carrier KEPT."""
    import run_688_bench as runner

    rows = _z_synthetic_rows()[0]
    before = {n: (r["carrier"].copy(), r["e0"].copy()) for n, r in rows.items()}
    armed, records = runner.apply_arm_to_rows("floor", rows, protocol="exp1_narrow")
    for name, (carrier, e0) in before.items():
        assert not np.any(armed[name]["carrier"])
        assert not np.any(armed[name]["e0"])
        assert armed[name]["carrier"].shape == carrier.shape
        assert armed[name]["e0"].shape == e0.shape
        assert records[name]["e0_action"] == "zero"
        assert records[name]["carrier_action"] == "zero"
        assert np.array_equal(rows[name]["carrier"], carrier)  # never mutated
    armed_z0, records_z0 = runner.apply_arm_to_rows("z0", rows, protocol="exp1_narrow")
    for name, (carrier, e0) in before.items():
        assert np.array_equal(armed_z0[name]["carrier"], carrier)  # carrier kept
        assert not np.any(armed_z0[name]["e0"])                    # E0 zeroed
        assert records_z0[name]["e0_action"] == "zero"
        assert records_z0[name]["carrier_action"] == "identity"


class _MockB3SEncoder:
    """Stand-in for the frozen B3S id_encoder with a Sequential post_pool
    (the real _build_affine_stack geometry: Linear + ReLU + Linear)."""

    def __init__(self, n_units: int, trial_bins: int):
        import torch
        from torch import nn

        self.torch = torch
        self.pre_pool = nn.Sequential(nn.Linear(trial_bins, 8), nn.ReLU())
        self.post_pool = nn.Sequential(
            nn.Linear(8 + plan.CARRIER_DIM, 16), nn.ReLU(), nn.Linear(16, plan.E0_DIM),
        )


class _MockB3SStudent:
    def __init__(self, n_units: int, trial_bins: int):
        self.id_encoder = _MockB3SEncoder(n_units, trial_bins)


def test_equiv_zero_projection_param_parity_and_determinism():
    from dandi688_bench_v1 import equiv_zero
    import torch

    student = _MockB3SStudent(6, 40)
    reference = student.id_encoder.post_pool
    p1 = equiv_zero.build_fixed_projection(student, seed=42)
    p2 = equiv_zero.build_fixed_projection(student, seed=42)
    # exact parameter parity + mirrored Linear geometry
    assert equiv_zero.param_count(p1) == equiv_zero.param_count(reference)
    assert equiv_zero.weights_sha256(p1) == equiv_zero.weights_sha256(p2)  # deterministic
    for a, b in zip(p1, reference):
        assert type(a) is type(b)
    # same seed -> identical weights, different seed -> different weights
    w1 = [p.detach().clone() for p in p1.parameters()]
    w2 = [p.detach().clone() for p in p2.parameters()]
    assert all(torch.equal(x, y) for x, y in zip(w1, w2))
    p3 = equiv_zero.build_fixed_projection(student, seed=7)
    assert not all(
        torch.equal(x, y)
        for x, y in zip(w1, p3.parameters())
    )
    # the random weights are NOT the frozen weights (pathway replaced)
    assert equiv_zero.weights_sha256(p1) != equiv_zero.weights_sha256(reference)
    # frozen: no parameter ever trains
    assert all(not p.requires_grad for p in p1.parameters())
    # RNG state is restored around the draw (no side effects)
    state = torch.get_rng_state()
    equiv_zero.build_fixed_projection(student)
    assert torch.equal(torch.get_rng_state(), state)
    # receipt fields
    receipt = equiv_zero.projection_receipt(student, p1)
    assert receipt["seed"] == 42 and receipt["trainable"] is False
    assert receipt["param_parity"] is True
    assert receipt["param_count_random"] == receipt["param_count_reference"]
    assert receipt["linear_geometry"] == [[12, 16], [16, plan.E0_DIM]]
    assert len(receipt["weights_sha256"]) == 64


def test_equiv_zero_e0_replacement_law():
    """E0 slot: fixed random projection over the pre_pool activity mean with
    a zero side -- nonzero, label-free, session-specific through activity
    only, and different from both the frozen melt and the f_labelfree melt."""
    from dandi688_bench_v1 import equiv_zero, vstate

    n_units, trial_bins, n_pad = 6, 40, 9
    rng = np.random.default_rng(23)
    calib_a = rng.random((30, trial_bins, n_units)).astype(np.float32)
    calib_b = rng.random((30, trial_bins, n_units)).astype(np.float32)
    student = _MockB3SStudent(n_units, trial_bins)
    projection = equiv_zero.build_fixed_projection(student)

    e0_a = equiv_zero.e0_from_projection(projection, student, calib_a, n_pad)
    e0_b = equiv_zero.e0_from_projection(projection, student, calib_b, n_pad)
    assert e0_a.shape == e0_b.shape == (n_pad, plan.E0_DIM)
    assert not np.any(e0_a[n_units:])                       # padding zero
    assert np.any(e0_a[:n_units])                           # nonzero pathway
    assert not np.array_equal(e0_a, e0_b)                   # activity covaries
    # the same projection on the same input is deterministic
    assert np.array_equal(e0_a, equiv_zero.e0_from_projection(projection, student, calib_a, n_pad))
    # differs from the frozen-post_pool melts (label side / trained weights)
    frozen_e0 = vstate.remelt_e0(student, calib_a, np.zeros((n_units, plan.CARRIER_DIM), np.float32), n_pad)
    assert not np.array_equal(e0_a, frozen_e0)
    side = rng.normal(size=(n_units, plan.CARRIER_DIM)).astype(np.float32)
    assert not np.array_equal(e0_a, vstate.remelt_e0(student, calib_a, side, n_pad))
    # shape law
    with pytest.raises(ValueError, match="n_pad"):
        equiv_zero.e0_from_projection(projection, student, calib_a, n_units - 1)


def test_equiv_zero_model_param_parity_with_t4():
    """Strict assertion of the mandate: the TRAINED model's parameter count
    is identical between the t4 and equiv_zero arms (same model class; the
    bank is an input, not model parameters)."""
    import run_688_bench as runner
    import torch

    n_units = _synthetic_row()["neural"].shape[1]
    device = torch.device("cpu")
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    t4_model = runner.build_model(n_units, torch, device, fusion="proj_add")
    ez_model = runner.build_model(n_units, torch, device, fusion="proj_add")
    assert count(t4_model) == count(ez_model) > 0
    assert arms.arm_fusion("equiv_zero") == "proj_add"
    # carrier token SHAPE kept (values zeroed): plan.CARRIER_DIM columns survive
    row = _synthetic_row()
    out = arms.apply_arm("equiv_zero", "s", row["carrier"], row["mask"])
    assert out.shape == row["carrier"].shape and not np.any(out)


# --------------------------------------------------------------------------
# 2026-09-10 ablation-condition axis (user directive: plans B / C1 / D1 / D2)
# --------------------------------------------------------------------------
def test_exp1_poverty_protocol_geometry():
    """Plan B: train = exactly the FIRST THREE exp1_narrow train sessions
    (earliest dates, 3-day span), exam face unchanged, strictly later."""
    train = plan.EXP1_POVERTY_TRAIN_SESSIONS
    assert train == plan.EXP1_NARROW_TRAIN_SESSIONS[:3]
    assert train == ("sub-C_ses-CO-20150629", "sub-C_ses-CO-20150630",
                     "sub-C_ses-CO-20150701")
    assert plan.PROTOCOLS["exp1_poverty"]["exam_sessions"] == plan.EXP1_NARROW_EXAM_SESSIONS
    assert plan.PROTOCOLS["exp1_poverty"]["span_days"] == 3
    dates = [plan.session_date(n) for n in train]
    assert dates == sorted(dates) and (dates[-1] - dates[0]).days + 1 == 3
    for exam in plan.EXP1_NARROW_EXAM_SESSIONS:
        assert plan.session_date(exam) > dates[-1]
    assert not set(train) & set(plan.EXP1_NARROW_EXAM_SESSIONS)
    assert not set(train) & set(plan.FORMAL_TEST_SESSIONS)
    # resolution: poverty train face is a subset of the manifest train split
    resolved = plan.verify_protocol_definitions()
    assert set(resolved["exp1_poverty"]["train"]) <= set(resolved["exp2_full"]["train"])
    assert resolved["exp1_poverty"]["exam"] == resolved["exp1_narrow"]["exam"]
    # z-table totality: nearest-date rule maps every poverty exam to 0701
    assert set(plan.z_srcbank_map("exp1_poverty").values()) == {"sub-C_ses-CO-20150701"}
    # receipt block carries the poverty disclosure
    block = plan.protocol_receipt_block("exp1_poverty")
    assert block["n_train"] == 3 and block["n_exam"] == 4
    assert block["cross_protocol_disclosure"] == plan.CROSS_PROTOCOL_DISCLOSURE


def test_poverty_protocol_filters_rows():
    import run_688_bench as runner

    splits = json.loads(plan.manifest_path().read_text())["session_splits"]
    rows = {name: {"split": "train", "starts": np.arange(10, dtype=np.int64)}
            for name in splits["train"]}
    poverty = runner.filter_rows_by_protocol(rows, "exp1_poverty")
    assert len(poverty) == 7  # 3 train + 4 exam
    assert {n for n, r in poverty.items() if r["protocol_role"] == "train"} == set(
        plan.EXP1_POVERTY_TRAIN_SESSIONS)
    assert {n for n, r in poverty.items() if r["protocol_role"] == "exam"} == set(
        plan.EXP1_NARROW_EXAM_SESSIONS)


# --------------------------------------------------------------------------
# exp2015_full (user directive 2026-09-09: 2015-only remote exam)
# --------------------------------------------------------------------------
def test_exp2015_full_protocol_geometry():
    """18-session 2015-only train face (exp2_full minus the 9 sessions dated
    2013 -- the ONLY difference), exam = the exp2_full val face, all the
    leakage/disclosure laws."""
    train = plan.EXP2015_FULL_TRAIN_SESSIONS
    spec = plan.PROTOCOLS["exp2015_full"]
    assert spec["train_sessions"] == train and spec["exam_sessions"] == "ALL_VAL"
    assert spec["span_days"] == 130
    assert "2015-only remote exam" in spec["face_role"]
    # the train face: 18 sessions, ALL dated 2015 (names carry the 2015 date)
    assert len(train) == 18
    assert all(plan.session_date(n).year == 2015 for n in train)
    months = [plan.session_date(n).month for n in train]
    assert months.count(3) == 5 and months.count(6) == 2 and months.count(7) == 11
    dates = [plan.session_date(n) for n in train]
    assert dates == sorted(dates)
    assert (dates[-1] - dates[0]).days + 1 == 130  # 2015-03-09..07-16 inclusive
    # exam face == the exp2_full val face (the 6 2015-11 sessions, remote)
    resolved = plan.verify_protocol_definitions()
    assert resolved["exp2015_full"]["exam"] == resolved["exp2_full"]["exam"] \
        == plan.VAL_SESSIONS
    # LEAKAGE CHECK: the exam face is disjoint from every train face it could
    # leak through -- its own train face and the exp1_narrow train face (the
    # narrow train sessions are a SUBSET of the exp2015_full train face by
    # construction, so only the exam side can be -- and is -- disjoint).
    assert not set(resolved["exp2015_full"]["exam"]) & set(train)
    assert not set(resolved["exp2015_full"]["exam"]) & set(plan.EXP1_NARROW_TRAIN_SESSIONS)
    # the ONLY difference from exp2_full = the 9 sessions dated 2013 removed
    removed = set(resolved["exp2_full"]["train"]) - set(train)
    assert removed == {n for n in resolved["exp2_full"]["train"]
                       if plan.session_date(n).year == 2013}
    assert len(removed) == 9 and set(train) | removed == set(resolved["exp2_full"]["train"])
    # disclosed overlap (exp2_full discipline): narrow train AND exam sit
    # inside the exp2015_full train face -- independent measurement, never
    # compare faces across protocols
    assert set(plan.EXP1_NARROW_TRAIN_SESSIONS) | set(plan.EXP1_NARROW_EXAM_SESSIONS) \
        <= set(train)
    assert "exp2015_full" in plan.CROSS_PROTOCOL_DISCLOSURE
    # every exam session strictly after the last train session (remote exam)
    for name in resolved["exp2015_full"]["exam"]:
        assert plan.session_date(name) > dates[-1]
    # no formal-test session anywhere on the protocol
    assert not set(train) & set(plan.FORMAL_TEST_SESSIONS)
    assert not set(resolved["exp2015_full"]["exam"]) & set(plan.FORMAL_TEST_SESSIONS)
    # z-table law totality: nearest-date rule maps every val session to the
    # 2015 train face's last session
    assert set(plan.z_srcbank_map("exp2015_full").values()) == {"sub-C_ses-CO-20150716"}
    # receipt block
    block = plan.protocol_receipt_block("exp2015_full")
    assert block["n_train"] == 18 and block["n_exam"] == 6 and block["span_days"] == 130
    assert block["cross_protocol_disclosure"] == plan.CROSS_PROTOCOL_DISCLOSURE


def test_exp2015_full_filters_rows():
    """Runner wiring: load_rows/filter_rows_by_protocol keep exactly the 18
    train + 6 val sessions of exp2015_full (synthetic rows + the real frozen
    cache when on disk)."""
    import run_688_bench as runner

    splits = json.loads(plan.manifest_path().read_text())["session_splits"]
    rows = {name: {"split": "train" if name in splits["train"] else "val",
                   "starts": np.arange(10, dtype=np.int64)}
            for name in list(splits["train"]) + list(splits["val"])}
    kept = runner.filter_rows_by_protocol(rows, "exp2015_full")
    assert len(kept) == 24  # 18 train + 6 exam
    assert {n for n, r in kept.items() if r["protocol_role"] == "train"} == set(
        plan.EXP2015_FULL_TRAIN_SESSIONS)
    assert {n for n, r in kept.items() if r["protocol_role"] == "exam"} == set(
        plan.VAL_SESSIONS)
    # the 2013 sessions and nothing else is filtered out of the train face
    dropped = set(rows) - set(kept)
    assert dropped == {n for n in splits["train"]
                       if plan.session_date(n).year == 2013}
    # inputs never mutated
    assert all("protocol_role" not in r for r in rows.values())

    cache = plan.prepared_cache_path()
    if not (cache / "prepared_contract.json").is_file():
        pytest.skip("frozen prepared cache not on disk")
    meta = runner.verify_prepared_cache(cache)
    real = runner.load_rows(cache, meta, "exp2015_full")
    assert len(real) == 24
    assert sum(r["protocol_role"] == "train" for r in real.values()) == 18
    assert sum(r["protocol_role"] == "exam" for r in real.values()) == 6
    # the f_labelfree variant cache (protocol-free) covers the same face
    f_cache = PKG_ROOT / "results" / "cache_f_labelfree"
    if (f_cache / "prepared_contract.json").is_file():
        fmeta = runner.verify_prepared_cache(f_cache)
        assert runner.assert_cache_variant_for_arm(fmeta, "f_labelfree", "exp2015_full")[
            "variant"] == "f_labelfree"
        frows = runner.load_rows(f_cache, fmeta, "exp2015_full")
        assert len(frows) == 24


# --------------------------------------------------------------------------
# exp2016 (user directive 2026-09-10: 2016 remote exam, first-91 ruling)
# --------------------------------------------------------------------------
def test_exp2016_protocol_geometry():
    """Same 18-session 2015 train face as exp2015_full; exam = the 14 CO
    sessions of 2016, entirely OUTSIDE the frozen manifest (roster-law
    conflict disclosed), requiring the extension cache."""
    train = plan.EXP2015_FULL_TRAIN_SESSIONS
    exam = plan.EXP2016_EXAM_SESSIONS
    spec = plan.PROTOCOLS["exp2016"]
    assert spec["train_sessions"] == train
    assert spec["exam_sessions"] == exam
    assert spec["span_days"] == 130  # train span unchanged
    assert "2016 remote exam" in spec["face_role"]
    # exam face: 14 sessions, all dated 2016, date-ordered, ~14 months remote
    dates = [plan.session_date(n) for n in exam]
    assert len(exam) == 14 and all(d.year == 2016 for d in dates)
    assert dates == sorted(dates)
    assert dates[0].isoformat() == "2016-09-09" and dates[-1].isoformat() == "2016-10-21"
    train_dates = [plan.session_date(n) for n in train]
    for d in dates:
        assert d > train_dates[-1]  # strictly after 2015-07-16
    # train face identical to exp2015_full; leakage: disjoint from every
    # frozen face (train incl. exp1_narrow's, val, formal test, narrow exam)
    resolved = plan.verify_protocol_definitions()
    assert resolved["exp2016"]["train"] == resolved["exp2015_full"]["train"]
    assert not set(exam) & set(train)
    assert not set(exam) & set(plan.EXP1_NARROW_TRAIN_SESSIONS)
    assert not set(exam) & set(plan.VAL_SESSIONS)
    assert not set(exam) & set(plan.FORMAL_TEST_SESSIONS)
    manifest = json.loads(plan.manifest_path().read_text())["session_splits"]
    manifest_all = set(manifest["train"]) | set(manifest["val"]) | set(manifest["test"])
    assert set(exam) & manifest_all == set()  # outside the whole manifest
    # z-table law totality: every 2016 exam maps to the last train session
    assert set(plan.z_srcbank_map("exp2016").values()) == {"sub-C_ses-CO-20150716"}
    # receipt block
    block = plan.protocol_receipt_block("exp2016")
    assert block["n_train"] == 18 and block["n_exam"] == 14 and block["span_days"] == 130
    assert block["cross_protocol_disclosure"] == plan.CROSS_PROTOCOL_DISCLOSURE


def test_exp2016_filters_rows_and_requires_extension_cache():
    """Runner wiring: protocol filter keeps 18 train + 14 exam; the FROZEN
    cache fails closed for exp2016 (rows missing by design); the extension
    cache, when built, loads 32 rows."""
    import run_688_bench as runner

    rows = {name: {"split": "train", "starts": np.arange(10, dtype=np.int64)}
            for name in plan.EXP2015_FULL_TRAIN_SESSIONS}
    rows.update({name: {"split": "exam2016", "starts": np.arange(10, dtype=np.int64)}
                 for name in plan.EXP2016_EXAM_SESSIONS})
    kept = runner.filter_rows_by_protocol(rows, "exp2016")
    assert len(kept) == 32
    assert {n for n, r in kept.items() if r["protocol_role"] == "train"} == set(
        plan.EXP2015_FULL_TRAIN_SESSIONS)
    assert {n for n, r in kept.items() if r["protocol_role"] == "exam"} == set(
        plan.EXP2016_EXAM_SESSIONS)

    cache = plan.prepared_cache_path()
    if (cache / "prepared_contract.json").is_file():
        meta = runner.verify_prepared_cache(cache)
        with pytest.raises(RuntimeError, match="missing from prepared cache"):
            runner.load_rows(cache, meta, "exp2016")
    ext = PKG_ROOT / "results" / "cache_exp2016"
    if (ext / "prepared_contract.json").is_file():
        emeta = runner.verify_prepared_cache(ext)
        erows = runner.load_rows(ext, emeta, "exp2016")
        assert len(erows) == 32
        assert sum(r["protocol_role"] == "train" for r in erows.values()) == 18
        assert sum(r["protocol_role"] == "exam" for r in erows.values()) == 14


def test_model_override_axis_constants_and_parser():
    """D1/D2 override axis: frozen constants + CLI surface."""
    import run_688_bench as runner

    assert plan.MODEL_OVERRIDES == ("none", "shortwin", "shallow")
    assert plan.SHORTWIN_WINDOW_BINS == 10
    assert plan.SHALLOW_TEMPORAL_LAYERS == 1
    assert runner.window_bins_for("none") == runner.window_bins_for("shallow") == plan.WINDOW_BINS
    assert runner.window_bins_for("shortwin") == plan.SHORTWIN_WINDOW_BINS
    with pytest.raises(ValueError, match="model_override"):
        runner.window_bins_for("wide")
    parser = runner.build_parser()
    override_action = next(a for a in parser._actions if "--model-override" in a.option_strings)
    assert set(override_action.choices) == set(plan.MODEL_OVERRIDES)
    assert override_action.default == "none"
    args = parser.parse_args([
        "--stage", "train", "--arm", "t4",
        "--dest", "btransform_unified_v2/dandi688_bench_v1/results/x",
        "--model-override", "shortwin",
    ])
    assert args.model_override == "shortwin"


def test_shortwin_windows_keep_targets_and_shrink_history():
    """W=10 slicing law: same query targets as W=50, only the LAST 10 raw
    bins of each frozen 50-bin window enter the model."""
    import run_688_bench as runner

    row = _synthetic_row()
    starts = np.asarray(row["starts"][:6])
    x50, y50, v50 = runner.windows(row, starts)
    x10, y10, v10 = runner.windows(row, starts, window_bins=plan.SHORTWIN_WINDOW_BINS)
    assert x50.shape == (len(starts), 50, row["neural"].shape[1])
    assert x10.shape == (len(starts), 10, row["neural"].shape[1])
    assert v10.shape == (len(starts), 10)
    # targets byte-identical: the query time does not move
    assert np.array_equal(y50, y10)
    assert np.array_equal(y10, np.stack(
        [row["behavior"][s + plan.WINDOW_BINS - 1] for s in starts]).astype(np.float32))
    # shortwin x = the LAST 10 bins of the 50-bin window
    assert np.array_equal(x10, x50[:, -10:, :])
    with pytest.raises(ValueError, match="window_bins"):
        runner.windows(row, starts, window_bins=4)


def test_shallow_decoder_single_layer_same_receptive_field():
    """D2 subclass law: exactly ONE temporal block spanning the SAME total
    receptive field as the stock D4 stack; frontend/readout inherited."""
    import run_688_bench as runner
    import torch
    from dandi688_bench_v1.model_variants import RiftShallowDecoder

    device = torch.device("cpu")
    n_units = _synthetic_row()["neural"].shape[1]
    stock = runner.build_model(n_units, torch, device, fusion="proj_add")
    shallow = runner.build_model(n_units, torch, device, fusion="proj_add",
                                 model_override="shallow")
    assert type(shallow) is RiftShallowDecoder
    assert len(shallow.temporal.blocks) == 1 == plan.SHALLOW_TEMPORAL_LAYERS
    assert len(stock.temporal.blocks) == 4
    # same total receptive field: stock (75,75,75,74) vs shallow (46,)
    assert shallow.temporal_config.token_receptive_field == \
        stock.temporal_config.token_receptive_field == 46
    assert shallow.context_bins == stock.context_bins == plan.WINDOW_BINS
    # frontend + readout inherited untouched (only the temporal stack shrinks)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(shallow.frontend) == count(stock.frontend)
    assert count(shallow.readout) == count(stock.readout)
    assert 0 < count(shallow) < count(stock)
    # deterministic construction under the seed
    shallow2 = runner.build_model(n_units, torch, device, fusion="proj_add",
                                  model_override="shallow")
    for a, b in zip(shallow.temporal.parameters(), shallow2.temporal.parameters()):
        assert torch.equal(a, b)
    # forward works and stays finite
    row = _synthetic_row()
    x, y, v = runner.windows(row, np.asarray(row["starts"][:2]))
    bank = runner.make_bank("s", row, torch)
    with torch.inference_mode():
        out = shallow(torch.from_numpy(x), bank, input_valid_mask=torch.from_numpy(v))
    assert out.shape == (2, plan.OUT_DIM) and bool(torch.isfinite(out).all())
    assert not torch.cuda.is_initialized()


def test_dir16_mapping_and_readout_math():
    """C1 math: even-bin law (every canonical 8-angle lands on an EVEN
    16-bin), hand-computed first-harmonic readout, and the exact half-scale
    degeneracy of a/c/m under the fixed-table closed form."""
    from dandi688_bench_v1 import dir16
    from btransform_unified_v2 import carrier_profile_v3 as v3

    theta8 = v3.canonical_directions_rad()
    theta16 = dir16.canonical_directions16_rad()
    assert len(theta16) == 16
    # every canonical 8-angle coincides with an EVEN 16-bin center (180 deg
    # == bin 14), so the 16-bin remap can never split an 8-dir population
    for angle in theta8:
        k = dir16.nearest_direction16_index(float(angle))
        assert k % 2 == 0 and theta16[k] == pytest.approx(float(angle))
    idx = dir16.direction16_indices(theta8)
    assert sorted(set(idx.tolist())) == [0, 2, 4, 6, 8, 10, 12, 14]

    # one unit responding only on direction 2 of 8: hand-computed readout.
    # 8 trials x 8 units with the identity one-hot design; every unit reads
    # R8 = [0,0,1,0,0,0,0,0] -> R16 = [0,0,0,0,1,0,...,0] (bin 4); the
    # closed-form table mean halves a and c relative to the 8-bin table.
    z = np.zeros((8, 8))
    z[2, :] = 1.0  # every unit fires only on the direction-2 trial
    r8 = v3.conditional_response(z, np.eye(8, dtype=np.float64), 0.0)
    a8, c8, m8 = v3.harmonic_readout(r8)
    r16 = np.zeros((8, 16))
    r16[:, 4] = r8[:, 2]
    a16, c16, m16 = dir16.harmonic_readout16(r16)
    assert a8 == pytest.approx(2.0 * np.cos(theta8[2]) / 8.0)
    assert a16 == pytest.approx(a8 / 2.0, rel=1e-12)
    assert c16 == pytest.approx(c8 / 2.0, rel=1e-12)
    assert m16 == pytest.approx(m8 / 2.0, rel=1e-12)
    # one-hot shape law + refusal
    assert dir16.one_hot_directions16(np.asarray([0, 15])).shape == (2, 16)
    with pytest.raises(ValueError):
        dir16.one_hot_directions16(np.asarray([16]))
    with pytest.raises(ValueError):
        dir16.direction16_indices(np.asarray([np.nan]))


def test_ablation_condition_comparison_report():
    """2026-09-10 condition comparison: per-condition ladder increments,
    baseline side-by-side, carrier-axis f_labelfree reuse law."""
    report = plan.ablation_condition_comparison({
        "exp1_poverty": {"t4": 0.80, "f_labelfree": 0.60, "floor": 0.20},
        "t4_dir16": {"t4": 0.87},
        "t4_shortwin": {"t4": 0.70, "f_labelfree": 0.55},
    })
    assert report["schema"] == "dandi688_bench_v1_condition_comparison"
    base = plan.EXP1_NARROW_BASELINE_R2
    assert report["baseline"]["r2"] == base
    assert report["baseline"]["carrier_label_increment"] == pytest.approx(
        base["t4"] - base["f_labelfree"])
    poverty = report["conditions"]["exp1_poverty"]
    assert poverty["carrier_label_increment"]["delta"] == pytest.approx(0.20)
    assert poverty["activity_independent"]["delta"] == pytest.approx(0.40)
    assert poverty["calibration_total"]["delta"] == pytest.approx(0.60)
    assert poverty["f_labelfree_reused_from_baseline"] is False
    assert poverty["vs_baseline"]["carrier_label_increment"] == pytest.approx(
        0.20 - (base["t4"] - base["f_labelfree"]))
    # carrier-axis condition may reuse the baseline ACTIVITY_ONLY rung
    d16 = report["conditions"]["t4_dir16"]
    assert d16["f_labelfree_reused_from_baseline"] is True
    assert d16["carrier_label_increment"]["delta"] == pytest.approx(0.87 - base["f_labelfree"])
    assert d16["r2"]["f_labelfree"] is None and "floor" not in d16
    # geometry conditions must carry their own f_labelfree rung
    assert report["conditions"]["t4_shortwin"]["f_labelfree_reused_from_baseline"] is False
    # the u1 estimator-family comparator is also carrier-axis (legal reuse)
    u1_report = plan.ablation_condition_comparison({"t4_u1_m10": {"t4": 0.8826}})
    assert u1_report["conditions"]["t4_u1_m10"]["f_labelfree_reused_from_baseline"] is True
    with pytest.raises(RuntimeError, match="own f_labelfree rung"):
        plan.ablation_condition_comparison({"t4_shallow": {"t4": 0.5}})
    with pytest.raises(RuntimeError, match="must carry a t4 reading"):
        plan.ablation_condition_comparison({"bad": {"f_labelfree": 0.5}})
    assert report["readings_only"] is True


def test_model_override_smoke_and_contract_binding():
    """Every override builds, trains one CPU step, and lands in the contract
    digest (train/score binding law)."""
    import run_688_bench as runner
    import torch

    row = _synthetic_row()
    meta = {
        "schema": plan.PREPARED_CACHE_SCHEMA,
        "manifest_sha256": plan.MANIFEST_SHA256,
        "split_counts": {"train": 27, "val": 6},
        "formal_test_used": False,
    }
    armed_rows = {"sub-C_ses-CO-20150629": {
        "neural": np.zeros((10, 4), np.float32), "protocol_role": "train"}}
    verify_records = {"sub-C_ses-CO-20150629": {"carrier_sha256_after": "abc"}}
    digests = {}
    for override in plan.MODEL_OVERRIDES:
        smoke = runner.smoke_forward_backward(
            row, "synthetic-session", batch=2, fusion="proj_add",
            model_override=override,
        )
        assert smoke["forward_finite"] and smoke["gradients_finite"]
        assert smoke["stream_parity"]["max_abs"] <= 1e-5
        assert smoke["model_override"] == override
        assert smoke["window_bins"] == runner.window_bins_for(override)
        contract = runner.make_bench_contract(
            "t4", meta, armed_rows, verify_records, 12, "exp1_narrow",
            model_override=override,
        )
        assert contract["model_override"] == override
        assert contract["input_window_bins"] == runner.window_bins_for(override)
        assert contract["window_bins"] == plan.WINDOW_BINS  # frozen 50 kept
        digests[override] = runner.bench_contract_digest(contract)
    # the three overrides produce three distinct contracts (fail-closed score)
    assert len(set(digests.values())) == 3
    assert not torch.cuda.is_initialized()


# --------------------------------------------------------------------------
# NORM_ONLY zero-calibration baseline (guide btransform_unified_v2/docs/
# NORM_ONLY_ZERO_BASELINE_GUIDE_20260910.md sections 2/3/5/6; arm norm_only,
# cache variant norm_only, identity construction in src/.../norm_only.py)
# --------------------------------------------------------------------------
def test_norm_only_pooled_rate_hand_computed():
    """rate_sess[u] = total counts / total duration over the support tensor."""
    from dandi688_bench_v1 import norm_only

    calib = np.zeros((2, 2, 3), dtype=np.float64)
    calib[0, 0, 0] = 6.0
    calib[1, 1, 0] = 6.0     # unit 0: 12 counts over 4 x 1 s -> 3.0
    calib[0, 1, 1] = 2.0     # unit 1:  2 counts over 4 s       -> 0.5
    rate = norm_only.pooled_support_rate(calib, bin_seconds=1.0)
    assert rate == pytest.approx([3.0, 0.5, 0.0])
    # the 688 bin (20 ms) is only a constant factor: the RATE changes, the
    # z built from it does not (the factor cancels inside the deviation)
    hz = norm_only.pooled_support_rate(calib)
    assert hz == pytest.approx(np.asarray([3.0, 0.5, 0.0]) / plan.NORM_ONLY_RATE_BIN_SECONDS)
    mu = np.asarray([2.0, 1.0, 0.0])
    sigma = np.asarray([1.0, 0.5, 0.25])
    _, sec = norm_only.e0_from_rate_deviation(calib, mu, sigma, 3, 2, bin_seconds=1.0)
    scale = plan.NORM_ONLY_RATE_BIN_SECONDS  # the same statistic in Hz
    _, msec = norm_only.e0_from_rate_deviation(calib, mu / scale, sigma / scale, 3, 2)
    assert sec["z"][:3] == pytest.approx(msec["z"][:3])
    # fail closed on malformed supports
    with pytest.raises(ValueError, match=r"\[trials, bins, units\]"):
        norm_only.pooled_support_rate(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="non-degenerate"):
        norm_only.pooled_support_rate(np.zeros((0, 2, 3)))
    bad = calib.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="nonfinite"):
        norm_only.pooled_support_rate(bad, bin_seconds=1.0)


def test_norm_only_identity_hand_computed_and_broadcast():
    """z[u] = (rate_sess[u] - mu[u]) / sigma[u], E0[u, :] = z[u]."""
    from dandi688_bench_v1 import norm_only

    calib = np.zeros((2, 2, 3), dtype=np.float64)
    calib[0, 0, 0] = 6.0
    calib[1, 1, 0] = 6.0     # rate 3.0
    calib[0, 1, 1] = 2.0     # rate 0.5
    # unit 2 stays silent  -> rate 0.0
    mu = np.asarray([2.0, 1.0, 0.0, 0.0])
    sigma = np.asarray([1.0, 0.5, 0.25, 0.0])
    e0, info = norm_only.e0_from_rate_deviation(
        calib, mu, sigma, 4, plan.E0_DIM, bin_seconds=1.0)
    # hand-computed deviations: (3-2)/1 = 1, (0.5-1)/0.5 = -1, (0-0)/0.25 = 0
    assert info["z"] == pytest.approx([1.0, -1.0, 0.0, 0.0])
    assert e0.shape == (4, plan.E0_DIM) and e0.dtype == np.float32
    # broadcast: every real row is the same constant across e0_dim
    assert np.array_equal(e0[:3], np.asarray(info["z"][:3], np.float32)[:, None]
                          * np.ones((1, plan.E0_DIM), np.float32))
    # padding row is exactly zero and the identity is finite/O(1)
    assert not np.any(e0[3])
    assert np.isfinite(e0).all()
    assert info["n_real_units"] == 3 and info["n_defined_slots"] == 3
    assert info["n_support_trials"] == 2 and info["n_support_bins"] == 2
    assert abs(info["z"]).max() <= plan.NORM_ONLY_MAX_ABS_Z
    # summaries are receipt-ready floats (distribution range of the identity)
    for key in ("min", "max", "mean", "median", "std"):
        assert np.isfinite(info["z_summary"][key])
        assert np.isfinite(info["rate_summary"][key])
    assert info["rate_summary"]["max"] == pytest.approx(3.0)
    assert info["z_summary"]["n"] == 3
    # fail-closed construction guards
    with pytest.raises(ValueError, match="smaller than real units"):
        norm_only.e0_from_rate_deviation(calib, mu, sigma, 2, plan.E0_DIM, bin_seconds=1.0)
    with pytest.raises(ValueError, match="must have shape"):
        norm_only.e0_from_rate_deviation(calib, mu[:3], sigma, 4, plan.E0_DIM, bin_seconds=1.0)
    with pytest.raises(RuntimeError, match="nonfinite"):
        norm_only.validate_source_rate_normalizer(
            np.asarray([np.nan, 0.0]), np.asarray([1.0, 1.0]), 2)
    with pytest.raises(ValueError, match="non-negative"):
        norm_only.validate_source_rate_normalizer(
            np.asarray([1.0, -1.0]), np.asarray([1.0, 1.0]), 2)


def test_norm_only_sigma_floor_and_undefined_slots():
    """sigma floor 1e-6 verbatim (guide section 3); slots the source list
    never resolved take the frozen fallback z = 0 (no 1e-6 spike)."""
    from dandi688_bench_v1 import norm_only

    calib = np.zeros((1, 1, 2), dtype=np.float64)
    calib[0, 0, 0] = 8.0     # rate 8.0 counts/bin
    calib[0, 0, 1] = 4.0     # rate 4.0
    # slot 0 has a zero-spread source distribution: the floor alone would
    # turn the deviation into an O(1e5) spike -> the build guard must fire
    with pytest.raises(RuntimeError, match="scale violation"):
        norm_only.e0_from_rate_deviation(
            calib, np.asarray([1.0, 1.0]), np.asarray([0.0, 1.0]), 2, 2,
            bin_seconds=1.0)
    # the same construction is legitimate once the slot is declared
    # undefined (no source distribution to compare against)
    e0, info = norm_only.e0_from_rate_deviation(
        calib, np.asarray([1.0, 1.0]), np.asarray([0.0, 1.0]), 2, 2,
        source_rates_defined=np.asarray([False, True]), bin_seconds=1.0)
    assert info["n_undefined_slots"] == 1 and info["n_defined_slots"] == 1
    assert not np.any(e0[0])                            # fallback row, not a spike
    assert e0[1] == pytest.approx(3.0)                  # (4-1)/1 broadcast over e0_dim
    # the 1e-6 floor is honoured where a definition exists
    e0f, _ = norm_only.e0_from_rate_deviation(
        calib, np.asarray([8.0, 1.0]), np.asarray([0.0, 1.0]), 2, 2,
        source_rates_defined=np.asarray([True, True]), sigma_floor=2.0,
        bin_seconds=1.0)
    assert e0f[0, 0] == pytest.approx(0.0)   # (8-8)/floor
    # all slots undefined -> the identity would be identically zero: refuse
    with pytest.raises(RuntimeError, match="identically zero"):
        norm_only.e0_from_rate_deviation(
            calib, np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0]), 2, 2,
            source_rates_defined=np.asarray([False, False]), bin_seconds=1.0)


def test_norm_only_source_normalizer_hand_computed():
    """Per-slot mu/sigma over the source (train) sessions, population std
    (ddof 0), with the >= min_source_sessions definition rule."""
    from dandi688_bench_v1 import norm_only

    rates = [np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 2.0]),
             np.asarray([2.0, 4.0, 3.0])]
    norm = norm_only.fit_source_rate_normalizer(rates, 5)
    # slot 0: 1,3,2 -> mean 2, population std sqrt(2/3)
    assert norm["mu"][0] == pytest.approx(2.0)
    assert norm["sigma"][0] == pytest.approx(np.sqrt(2.0 / 3.0))
    # slot 1: 2,2,4 -> mean 8/3
    assert norm["mu"][1] == pytest.approx(8.0 / 3.0)
    # slot 2: 3,3 -> mean 3 and ZERO spread (a real single-valued source slot)
    assert norm["mu"][2] == pytest.approx(3.0) and norm["sigma"][2] == 0.0
    assert list(norm["counts"]) == [3, 3, 2, 0, 0]
    assert list(norm["defined"]) == [True, True, True, False, False]
    assert norm["n_source_sessions"] == 3 and norm["source_widths"] == [3, 2, 3]
    assert norm["sigma_floor"] == plan.NORM_ONLY_SIGMA_FLOOR == 1e-6
    assert np.isfinite(norm["mu"]).all() and np.isfinite(norm["sigma"]).all()
    # every array survives JSON (no NaN sentinel anywhere)
    json.dumps({key: np.asarray(value).tolist() for key, value in norm.items()
                if isinstance(value, np.ndarray)})
    # the definition rule is a parameter: a 3-session threshold drops the
    # two-session slot
    strict = norm_only.fit_source_rate_normalizer(rates, 5, min_source_sessions=3)
    assert list(strict["defined"]) == [True, True, False, False, False]
    with pytest.raises(RuntimeError, match="definition threshold"):
        norm_only.fit_source_rate_normalizer(rates, 5, min_source_sessions=9)
    with pytest.raises(ValueError, match="nonempty"):
        norm_only.fit_source_rate_normalizer([], 5)
    with pytest.raises(ValueError, match="wider than n_pad"):
        norm_only.fit_source_rate_normalizer([np.zeros(6)], 5)
    # receipt block: source names, statistic definition, digests of the
    # frozen arrays (the JSON mirror re-hashes to the same values)
    receipt = norm_only.source_normalizer_receipt(norm, ["s1", "s2", "s3"])
    assert receipt["name"] == "norm_only_rate_deviation_broadcast"
    assert receipt["source_sessions"] == ["s1", "s2", "s3"]
    assert "total counts / total duration" in receipt["rate_statistic"]
    assert receipt["support_trials"] == 30
    assert receipt["mu_sha256"] == plan.array_digest(np.asarray(receipt["source_rate_arrays"]["mu"]))
    assert receipt["sigma_sha256"] == plan.array_digest(
        np.asarray(receipt["source_rate_arrays"]["sigma"]))
    assert receipt["counts_sha256"] == plan.array_digest(
        np.asarray(receipt["source_rate_arrays"]["counts"]))
    assert receipt["defined_sha256"] == plan.array_digest(
        np.asarray(receipt["source_rate_arrays"]["defined"]))
    assert receipt["mu_sha256"] == plan.array_digest(np.asarray(norm["mu"]))
    assert receipt["mu_summary"]["n"] == 3 and receipt["mu_summary"]["min"] == pytest.approx(2.0)
    assert receipt["sigma_summary"]["min"] == 0.0 and receipt["sigma_summary"]["max"] > 0.9
    assert receipt["n_defined_slots"] == 3 and receipt["n_undefined_slots"] == 2
    assert "no behavioral labels" in receipt["label_disclosure"]
    assert receipt["ladder_position"] == "NONE(floor) < NORM_ONLY(norm_only) < ACTIVITY_ONLY(f_labelfree) < FULL(t4)"
    assert "sigma = 0" in receipt["undefined_slot_rule"]


def test_norm_only_e0_difference_law():
    """The NORM_ONLY E0 must differ from the ACTIVITY_ONLY (f_labelfree) E0:
    NORM_ONLY carries no encoder pathway at all."""
    from dandi688_bench_v1 import norm_only

    rng = np.random.default_rng(11)
    left = rng.normal(size=(6, plan.E0_DIM)).astype(np.float32)
    right = left.copy()
    same = norm_only.e0_difference(left, right)
    assert same["identical"] is True and same["n_differing_rows"] == 0
    right[3, :] += 0.5
    diff = norm_only.e0_difference(left, right)
    assert diff["identical"] is False
    assert diff["n_differing_rows"] == 1 and diff["n_rows"] == 6
    assert diff["max_abs_diff"] == pytest.approx(0.5)
    assert diff["mean_abs_diff"] > 0.0
    with pytest.raises(ValueError, match="shapes must match"):
        norm_only.e0_difference(left, right[:, :10])
    bad = left.copy()
    bad[0, 0] = np.inf
    with pytest.raises(RuntimeError, match="nonfinite"):
        norm_only.e0_difference(bad, right)


def test_norm_only_ladder_report_arithmetic_and_sanity():
    """Guide section 5 sanity verdicts on the four-level ladder; the
    construction-error reading fails closed."""
    report = plan.norm_only_ladder_report(0.3247, 0.55, 0.8257, 0.8729,
                                          dc_penalty=0.05)
    assert report["schema"] == "dandi688_bench_v1_norm_only_ladder"
    assert report["ladder_order"] == list(plan.COMPONENT_LADDER_ORDER) == [
        "floor", "norm_only", "f_labelfree", "t4"]
    assert report["rungs"] == plan.COMPONENT_ABLATION["ladder"]
    assert report["r2"] == {"floor": 0.3247, "norm_only": 0.55,
                            "f_labelfree": 0.8257, "t4": 0.8729}
    deltas = report["deltas"]
    assert deltas["rate_deviation_identity"] == pytest.approx(0.55 - 0.3247)
    assert deltas["encoder_activity_increment"] == pytest.approx(0.8257 - 0.55)
    assert deltas["label_free_activity_total"] == pytest.approx(0.8257 - 0.3247)
    assert deltas["carrier_label_increment"] == pytest.approx(0.8729 - 0.8257)
    assert deltas["calibration_total"] == pytest.approx(0.8729 - 0.3247)
    assert (deltas["rate_deviation_identity"] + deltas["encoder_activity_increment"]
            == pytest.approx(deltas["label_free_activity_total"]))
    sanity = report["sanity"]
    assert sanity["monotone"] is True
    assert sanity["construction_error"] is False
    assert sanity["stronger_than_activity"] is False
    assert sanity["dc_penalty"] == pytest.approx(0.05)
    assert sanity["dc_penalty_flag"] is False      # rung <= 0.1 anyway
    assert report["laws"] == plan.NORM_ONLY_SANITY_CHECKS
    assert report["formal_test_policy"] == plan.FORMAL_TEST_POLICY
    # guide section 5: NORM_ONLY below NONE is a construction bug -> STOP
    with pytest.raises(RuntimeError, match="construction error"):
        plan.norm_only_ladder_report(0.35, 0.30, 0.82, 0.87)
    # NORM_ONLY above ACTIVITY_ONLY is recorded, not gated (report honestly)
    strong = plan.norm_only_ladder_report(0.30, 0.90, 0.82, 0.87, dc_penalty=0.2)
    assert strong["sanity"]["monotone"] is False
    assert strong["sanity"]["stronger_than_activity"] is True
    assert strong["sanity"]["dc_penalty_flag"] is True
    # nonfinite readings are refused
    with pytest.raises(RuntimeError, match="finite"):
        plan.norm_only_ladder_report(0.3, float("nan"), 0.8, 0.8)
    # the rung reading is optional in the component-ablation report, and when
    # passed it lands as the four-level block beside the frozen 3-item law
    with_rung = plan.component_ablation_report(
        0.8729, 0.8257, 0.3247, r2_norm_only=0.55, dc_penalty=0.05)
    assert with_rung["zero_calibration_baseline"]["r2"]["norm_only"] == 0.55
    assert with_rung["r2"]["norm_only"] == 0.55
    assert tuple(with_rung["decomposition"]) == (
        "activity_independent", "carrier_label_increment", "calibration_total")


def test_norm_only_ladder_constants_fail_closed():
    """The four-level ladder constants are frozen and the import-time
    validator rejects any drift."""
    assert plan.NORM_ONLY_VARIANT == "norm_only"
    assert plan.NORM_ONLY_CACHE_DIRNAME == "cache_norm_only"
    assert plan.NORM_ONLY_SIGMA_FLOOR == 1e-6          # guide section 3
    assert plan.NORM_ONLY_RATE_BIN_SECONDS == 0.02     # 688 BIN_MS = 20
    assert plan.NORM_ONLY_RATE_UNITS.startswith("counts per 20 ms bin")
    assert plan.NORM_ONLY_SUPPORT_TRIALS == 30         # = ACTIVITY_SUPPORT_N
    assert "calib_trials" in plan.NORM_ONLY_SUPPORT_NOTE
    assert plan.NORM_ONLY_MIN_SOURCE_SESSIONS == 3
    assert plan.NORM_ONLY_MAX_ABS_Z == 20.0
    assert plan.COMPONENT_LADDER_RUNGS[1] == "NORM_ONLY"
    assert plan.COMPONENT_LADDER_ORDER[1] == "norm_only"
    assert plan.COMPONENT_LADDER_ORDER == (
        plan.COMPONENT_LADDER_ORDER[0], "norm_only", "f_labelfree", "t4")
    assert plan.NORM_ONLY_SANITY_CHECKS["monotone"].startswith("NONE <= NORM_ONLY")
    assert "STOP" in plan.NORM_ONLY_SANITY_CHECKS["construction_error"]
    assert plan.NORM_ONLY_TASK_CRITERIA["M1"] == {
        "metric": "HO3 channel-weighted R2", "min": 0.3,
        "dc_penalty_max": 0.1, "e0_dim": 100}
    assert plan.NORM_ONLY_TASK_CRITERIA["M2"]["min"] == 0.15
    assert plan.NORM_ONLY_TASK_CRITERIA["H1"]["min"] == 0.25
    assert plan.NORM_ONLY_TASK_CRITERIA["H1"]["e0_dim"] == 700
    # a reordered or truncated ladder fails the import-time validator
    original = plan.COMPONENT_ABLATION["ladder"]
    try:
        plan.COMPONENT_ABLATION["ladder"] = {
            "NONE": "floor", "ACTIVITY_ONLY": "f_labelfree",
            "NORM_ONLY": "norm_only", "FULL": "t4"}
        with pytest.raises(RuntimeError, match="rung order"):
            plan._validate_component_ablation()
        plan.COMPONENT_ABLATION["ladder"] = {
            "NONE": "floor", "ACTIVITY_ONLY": "f_labelfree", "FULL": "t4"}
        with pytest.raises(RuntimeError, match="ladder must be exactly"):
            plan._validate_component_ablation()
    finally:
        plan.COMPONENT_ABLATION["ladder"] = original
    plan._validate_component_ablation()  # restored state still validates


def test_norm_only_arm_zero_carrier_identity_e0():
    """Load-time law of the norm_only arm: zero carrier (same law as
    f_labelfree) and E0 kept byte-for-byte (the identity lives in the cache)."""
    import run_688_bench as runner

    carrier, mask = synthetic_carrier()
    before = carrier.copy()
    out = arms.apply_arm("norm_only", "sub-C_ses-CO-20151103", carrier, mask)
    assert not np.any(out) and out.shape == carrier.shape
    assert np.array_equal(carrier, before)  # input never mutated
    rec = arms.verify_arm("norm_only", "sub-C_ses-CO-20151103", carrier, mask, out)
    assert rec["verify"] == "PASSED"
    assert rec["carrier_action"] == "zero" and rec["fusion"] == "proj_add"
    assert arms.arm_carrier_digest("norm_only", "s", carrier, mask) == \
        arms.arm_carrier_digest("f_labelfree", "s", carrier, mask)

    rows, _ = _z_synthetic_rows()
    armed, records = runner.apply_arm_to_rows("norm_only", rows, protocol="exp1_narrow")
    for name, row in rows.items():
        assert not np.any(armed[name]["carrier"])
        assert np.array_equal(armed[name]["e0"], row["e0"])   # E0 untouched
        assert records[name]["e0_action"] == "identity"
        assert records[name]["e0_sha256_after"] == plan.array_digest(row["e0"])
        assert np.array_equal(rows[name]["carrier"], row["carrier"])
    # a full-width synthetic row (the _z fixture carries too few bins to form
    # a batch): the zero-carrier arm forward/backward is finite
    row = _synthetic_row()
    row["carrier"] = arms.apply_arm("norm_only", "synthetic-session",
                                    row["carrier"], row["mask"])
    assert not np.any(row["carrier"])
    full = runner.smoke_forward_backward(row, "synthetic-session", batch=2,
                                         fusion=arms.arm_fusion("norm_only"))
    assert full["forward_finite"] and full["gradients_finite"]


def test_norm_only_cache_structure_and_identity():
    """Structural/finiteness audit of the built cache_norm_only (skips when
    not built): broadcast identity reproducible from the recorded rate /
    mu / sigma, padding discipline, byte-copied neural rows, E0 differing
    from both the frozen and the f_labelfree cache."""
    from dandi688_bench_v1 import norm_only

    cache = PKG_ROOT / "results" / plan.NORM_ONLY_CACHE_DIRNAME
    f_cache = PKG_ROOT / "results" / "cache_f_labelfree"
    if not (cache / "prepared_contract.json").is_file():
        pytest.skip("norm_only cache not built yet")
    meta = json.loads((cache / "prepared_contract.json").read_text())
    assert meta["schema"] == plan.PREPARED_CACHE_SCHEMA
    assert meta["manifest_sha256"] == plan.MANIFEST_SHA256
    assert meta["formal_test_used"] is False
    assert meta["variant"] == plan.NORM_ONLY_VARIANT
    splits = json.loads(plan.manifest_path().read_text())["session_splits"]
    assert set(meta["sessions"]) == set(splits["train"]) | set(splits["val"])
    assert not set(meta["sessions"]) & set(plan.FORMAL_TEST_SESSIONS)
    est = meta["estimator"]
    assert est["name"] == "norm_only_rate_deviation_broadcast"
    assert est["protocol"] == plan.DEFAULT_PROTOCOL
    assert est["source_sessions"] == sorted(plan.EXP1_NARROW_TRAIN_SESSIONS)
    assert est["protocol_train_sessions"] == list(plan.EXP1_NARROW_TRAIN_SESSIONS) or \
        sorted(est["protocol_train_sessions"]) == sorted(plan.EXP1_NARROW_TRAIN_SESSIONS)
    assert est["n_source_sessions"] == len(plan.EXP1_NARROW_TRAIN_SESSIONS) == 9
    assert est["sigma_floor"] == plan.NORM_ONLY_SIGMA_FLOOR
    assert est["support_trials"] == plan.NORM_ONLY_SUPPORT_TRIALS
    assert est["carrier"] == "all-zero [n_pad, 4]"
    assert "no behavioral labels" in est["label_disclosure"]
    assert est["ladder_position"] == norm_only.LADDER_LABEL
    assert est["reference_variant"] == "f_labelfree"
    # the frozen source distribution: digests match the JSON array mirror
    arrays = est["source_rate_arrays"]
    mu = np.asarray(arrays["mu"], dtype=np.float64)
    sigma = np.asarray(arrays["sigma"], dtype=np.float64)
    counts = np.asarray(arrays["counts"], dtype=np.int64)
    defined = np.asarray(arrays["defined"], dtype=bool)
    assert plan.array_digest(mu) == est["mu_sha256"]
    assert plan.array_digest(sigma) == est["sigma_sha256"]
    assert plan.array_digest(counts) == est["counts_sha256"]
    assert plan.array_digest(defined) == est["defined_sha256"]
    assert mu.shape == sigma.shape == counts.shape == defined.shape
    assert np.isfinite(mu).all() and np.isfinite(sigma).all()
    assert (sigma >= 0.0).all() and (mu >= 0.0).all()
    assert int(counts.max()) == len(plan.EXP1_NARROW_TRAIN_SESSIONS)
    assert int(defined.sum()) > 0 and not defined.all()
    assert not np.any(mu[~defined]) and not np.any(sigma[~defined])
    # every source slot's spread is either zero (single-valued slot) or real
    assert int(counts[defined].min()) >= plan.NORM_ONLY_MIN_SOURCE_SESSIONS

    z_all = []
    for name in sorted(meta["sessions"]):
        row = eval_local.load_session(cache, name)  # re-hashes all arrays
        mask = np.asarray(row["mask"])
        n_real = int(mask.sum())
        e0 = np.asarray(row["e0"])
        assert e0.shape == (mask.size, plan.E0_DIM) and e0.dtype == np.float32
        assert np.isfinite(e0).all()
        assert not np.any(row["carrier"])             # T = 0 everywhere
        # broadcast identity: each real row is constant along e0_dim
        assert np.allclose(e0[:n_real], e0[:n_real, :1], rtol=0.0, atol=0.0)
        assert not np.any(e0[n_real:])                # padding discipline
        assert not mask[n_real:].any() and mask[:n_real].all()
        log = est["session_log"][name]
        assert log["n_real_units"] == n_real
        assert log["n_defined_slots"] + log["n_undefined_slots"] == n_real
        assert log["n_support_trials"] == plan.NORM_ONLY_SUPPORT_TRIALS
        rate = np.asarray(log["rate"], dtype=np.float64)
        assert rate.shape == (n_real,)
        assert plan.array_digest(rate) == log["rate_sha256"]
        assert np.isfinite(rate).all() and (rate >= 0.0).all()
        # reproducible from the recorded statistic + frozen distribution
        z = (rate - mu[:n_real]) / np.maximum(sigma[:n_real], plan.NORM_ONLY_SIGMA_FLOOR)
        z = np.where(defined[:n_real], z, plan.NORM_ONLY_UNDEFINED_SLOT_Z)
        assert np.isfinite(z).all()
        assert np.allclose(z, np.asarray(e0[:n_real, 0], np.float64),
                           rtol=1e-6, atol=1e-7)
        assert float(np.abs(z).max()) <= plan.NORM_ONLY_MAX_ABS_Z
        assert plan.array_digest(e0) == log["e0_sha256"]
        assert log["e0_vs_f_labelfree"]["n_differing_rows"] > 0
        assert log["e0_vs_f_labelfree"]["identical"] is False
        z_all.append(z)
    assert min(float(z.min()) for z in z_all) < 0.0   # a real deviation spread
    assert max(float(z.max()) for z in z_all) > 0.0

    # byte-copied neural/behavior/starts/mask + the identity difference law
    name = plan.EXP1_NARROW_EXAM_SESSIONS[0]
    frozen = eval_local.load_session(plan.prepared_cache_path(), name)
    row = eval_local.load_session(cache, name)
    for key in ("neural", "behavior", "starts", "mask"):
        assert plan.array_digest(row[key]) == plan.array_digest(frozen[key])
    assert not np.array_equal(row["e0"], frozen["e0"])
    if (f_cache / "prepared_contract.json").is_file():
        frow = eval_local.load_session(f_cache, name)
        diff = norm_only.e0_difference(row["e0"], frow["e0"])
        assert diff["identical"] is False and diff["n_differing_rows"] > 0
        assert diff["max_abs_diff"] > 1e-6
        assert est["e0_difference_summary"]["min_max_abs_diff"] <= diff["max_abs_diff"]
        assert est["e0_difference_summary"]["sessions_differing"] == len(meta["sessions"])
