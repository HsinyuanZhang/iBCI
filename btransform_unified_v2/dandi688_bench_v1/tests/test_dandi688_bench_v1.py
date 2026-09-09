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
        "vstate", "z_vstate_srcbank", "f_labelfree",       # vstate-688 series
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
    # two-stage protocol constants (ADDENDUM-TWO-STAGE)
    assert tuple(plan.PROTOCOLS) == ("exp1_narrow", "exp2_full")
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
    assert set(protocol_action.choices) == {"exp1_narrow", "exp2_full"}


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
    # z0 shares the all-zero carrier; E0 zeroing is the runner's job
    assert np.array_equal(arms.apply_arm("z0", "s", carrier, mask), out)
    assert arms.arm_e0_action("z0") == "zero" and arms.arm_e0_action("f0") == "identity"


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
    assert arms.ARM_SPECS["z0"] == {"carrier": "zero", "fusion": "proj_add", "e0": "zero"}
    # vstate-688 series (user directive 2026-09-09): variant-cache bound,
    # proj_add; z-series bank swap is a runner-level cross-session operation
    assert arms.ARM_SPECS["vstate"] == {
        "carrier": "identity", "fusion": "proj_add", "e0": "identity",
    }
    assert arms.ARM_SPECS["z_vstate_srcbank"] == {
        "carrier": "identity", "fusion": "proj_add", "e0": "identity",
    }
    assert arms.ARM_SPECS["f_labelfree"] == {
        "carrier": "zero", "fusion": "proj_add", "e0": "identity",
    }
    assert plan.ARM_REQUIRED_CACHE_VARIANT == {
        "vstate": "vstate", "z_vstate_srcbank": "vstate",
        "f_labelfree": "f_labelfree",
    }
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
    # 8 arms, 3 distinct carrier transforms: identity {t4, t4_concat, vstate,
    # z_vstate_srcbank}, zero {f0, z0, f_labelfree}, shuffle {ts4}.
    assert len(plan.ARMS) == 8
    assert len(set(digests.values())) == 3
    assert digests["t4"] == plan.array_digest(carrier)
    assert digests["t4_concat"] == digests["t4"]
    assert digests["vstate"] == digests["z_vstate_srcbank"] == digests["t4"]
    assert digests["f0"] == digests["z0"] == digests["f_labelfree"]
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
        # does -- vstate/z on the vstate cache (protocol-bound), f_labelfree
        # on the f_labelfree cache.
        vstate_cache = PKG_ROOT / "results" / "cache_vstate_exp1_narrow"
        f_cache = PKG_ROOT / "results" / "cache_f_labelfree"
        for arm, cache in (("vstate", vstate_cache), ("z_vstate_srcbank", vstate_cache),
                           ("f_labelfree", f_cache)):
            if not (cache / "prepared_contract.json").is_file():
                continue
            vmeta = runner.verify_prepared_cache(cache)
            binding = runner.assert_cache_variant_for_arm(vmeta, arm, "exp1_narrow")
            assert binding == ({"arm": arm, "variant": "vstate",
                                "protocol": "exp1_narrow"}
                               if arm != "f_labelfree" else
                               {"arm": arm, "variant": "f_labelfree"})
            vrows = runner.load_rows(cache, vmeta, "exp1_narrow")
            varmed, vrecords = runner.apply_arm_to_rows(arm, vrows, protocol="exp1_narrow")
            assert all(r["verify"] == "PASSED" for r in vrecords.values())
            if arm == "z_vstate_srcbank":
                for exam in plan.EXP1_NARROW_EXAM_SESSIONS:
                    rec = vrecords[exam]
                    assert rec["z_srcbank"]["source_session"] == "sub-C_ses-CO-20150710"
                    assert rec["z_srcbank"]["exam_date_after_source"] is True
            if arm == "f_labelfree":
                assert all(not np.any(r["carrier"]) for r in varmed.values())
            first = next(n for n, r in sorted(varmed.items()) if r["protocol_role"] == "train")
            smoke = runner.smoke_forward_backward(
                varmed[first], first, batch=2, fusion=arms.arm_fusion(arm)
            )
            assert smoke["forward_finite"] and smoke["gradients_finite"]
            assert smoke["stream_parity"]["max_abs"] <= 1e-5
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
    b = float(R.mean() - R_hold.mean())
    assert got[0].tolist() == pytest.approx([a, c, m, b], rel=1e-12, abs=1e-12)

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
    # b keeps the delta_b motion-minus-hold sign: higher hold than motion
    # rates -> b < 0
    got_b = vstate.vstate_carrier_from_blocks(
        np.full((7, 1), 5.0), base_vel, np.full((3, 1), 50.0), np.zeros((3, 2)),
        np.ones(2),
    )
    assert got_b[0, 3] < 0


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

    def meta(variant=None, protocol=None):
        return {"variant": variant, "estimator": {"protocol": protocol}}

    # the five original arms are cache-agnostic
    assert runner.assert_cache_variant_for_arm(meta(), "t4", "exp1_narrow") is None
    # vstate arms bind to the vstate variant AND the active protocol
    assert runner.assert_cache_variant_for_arm(
        meta("vstate", "exp1_narrow"), "vstate", "exp1_narrow"
    ) == {"arm": "vstate", "variant": "vstate", "protocol": "exp1_narrow"}
    assert runner.assert_cache_variant_for_arm(
        meta("vstate", "exp1_narrow"), "z_vstate_srcbank", "exp1_narrow"
    ) == {"arm": "z_vstate_srcbank", "variant": "vstate", "protocol": "exp1_narrow"}
    # f_labelfree binds to its variant but carries no protocol
    assert runner.assert_cache_variant_for_arm(
        meta("f_labelfree"), "f_labelfree", "exp2_full"
    ) == {"arm": "f_labelfree", "variant": "f_labelfree"}
    # wrong variant / wrong protocol fail closed
    with pytest.raises(RuntimeError, match="variant cache"):
        runner.assert_cache_variant_for_arm(meta("u1_m10"), "vstate", "exp1_narrow")
    with pytest.raises(RuntimeError, match="variant cache"):
        runner.assert_cache_variant_for_arm(meta(), "f_labelfree", "exp1_narrow")
    with pytest.raises(RuntimeError, match="built for that protocol"):
        runner.assert_cache_variant_for_arm(meta("vstate", "exp2_full"), "vstate", "exp1_narrow")


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
    assert fmeta["variant"] == "f_labelfree"
    assert "zero side" in fmeta["estimator"]["e0_side"]
