"""Tests for the AM/IM decomposition round in scripts/run_subpop_score.py.

Covers HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md (activity-vs-identity
mask) as scored by the sub-population matched scorer:

- the pre-registered §3 reading matrix as a machine-checkable pure function:
  all four handoff rows on synthetic deltas (identity ablation / activity
  sub-sampling / joint ablation / SUSPICIOUS) plus the explicit
  out-of-matrix superior-to-D case, the |d| < 0.03 band boundaries, the
  single-landed-cell ``matrix_incomplete`` state and the no-cells state —
  every label with its own consequence, the SUSPICIOUS row never
  reportable as a finding, and the rule string recorded verbatim;
- the §5.3 ``supersedes`` threading: explicit ``--supersedes`` paths, the
  auto-derived output-root-family predecessor chain (transitive), and the
  empty first-run case; every receipt carries the top-level field;
- AM/IM terminal-receipt validation: a faithful fixture under
  ``results/aimask_v1/`` passes (status/epochs/initial-state binding/SWA
  SHA + sidecar, mask_structure, gain inherited from F.dropout, min_keep,
  the application site masking exactly one component, and
  ``mask_code_path_matches_D``); every tampered field refuses; a missing
  receipt stays ``not_landed``; the shared one-draw p stream is bound to
  the sealed R/S2 value;
- curve inclusion: AM/IM are CURVED (their eval graph is the exact parent
  graph, so the Step 0C wrapper is VALID for them — verified bitwise when
  the route module has landed) while cell W stays structurally excluded;
- the lazy ``activity_identity_cells`` loader (never touched at import
  time; a missing module is a named hard failure, not a traceback).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat as stat_mod
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "subpop_score_aimask_under_test", ROOT / "scripts/run_subpop_score.py"
)
score = importlib.util.module_from_spec(spec)
sys.modules["subpop_score_aimask_under_test"] = score
spec.loader.exec_module(score)

# the sealed shared one-p-per-forward stream of R/S2/T/G (read from the
# SHA-verified sparsification receipt; hardcoded HERE in the test only)
SEALED_SHARED_P_SEQUENCE = (
    "e62fc92ffe1ce0983d83d22dae299b2e9864112818b9a89d13dd21052b5b8b29"
)


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ArmCommonStub:
    sha256_file = staticmethod(_sha_file)


_CANONICAL_STUB = {"state_sha256": "canonical-state-sha"}
_THETA_STUB = {"authority_sha256": "theta-authority-sha"}


# ---------------------------------------------------------------------------
# the pre-registered reading matrix (handoff §3) on synthetic deltas
# ---------------------------------------------------------------------------
def test_matrix_row_identity_ablation():
    reading = score.decomposition_reading(-0.10, 0.005)  # AM << D, IM ~ D
    assert reading["matrix_status"] == "matrix_complete"
    assert reading["matrix_row"] == "IDENTITY_ABLATION_IS_THE_MECHANISM"
    assert "headline rewrite" in reading["consequence"]
    assert reading["reportable_as_finding"] is True
    assert reading["cells"]["AM"]["band"] == "MUCH_LESS_THAN_D"
    assert reading["cells"]["IM"]["approx_equal_to_d"] is True
    assert "REVOKED" in reading["mechanism_sentence_status"]


def test_matrix_row_activity_subsampling():
    reading = score.decomposition_reading(0.005, -0.10)  # AM ~ D, IM << D
    assert reading["matrix_row"] == "ACTIVITY_SUBSAMPLING_IS_THE_MECHANISM"
    assert "population story stands as written" in reading["consequence"]
    assert reading["reportable_as_finding"] is True
    assert "UNCHANGED" in reading["mechanism_sentence_status"]


def test_matrix_row_joint_ablation():
    reading = score.decomposition_reading(-0.20, -0.31)  # both << D
    assert reading["matrix_row"] == "JOINT_ABLATION_REQUIRED"
    assert "decomposition's floor" in reading["consequence"]
    assert reading["cells"]["AM"]["approx_equal_to_d"] is False
    assert reading["cells"]["IM"]["approx_equal_to_d"] is False


def test_matrix_row_suspicious_is_never_a_finding():
    reading = score.decomposition_reading(0.001, -0.002)  # both ~ D
    # the label text itself must carry the do-not-report instruction
    assert reading["matrix_row"] == (
        "SUSPICIOUS__INVESTIGATE_MASK_AND_EVAL_PATH__DO_NOT_REPORT"
    )
    assert reading["matrix_row"] == score.AIMASK_LABELS["suspicious"]
    assert reading["reportable_as_finding"] is False
    assert "Do not report as a finding" in reading["consequence"]
    assert "investigate the mask draw" in reading["consequence"]
    assert "NO CHANGE PERMITTED" in reading["mechanism_sentence_status"]


@pytest.mark.parametrize("am_delta,im_delta", [
    (+0.05, -0.10),   # AM superior to D
    (0.001, +0.04),   # IM superior to D
    (+0.03, +0.20),   # both superior (boundary inclusive)
])
def test_matrix_out_of_the_table_superior_case(am_delta, im_delta):
    reading = score.decomposition_reading(am_delta, im_delta)
    assert reading["matrix_row"] == (
        "UNANTICIPATED__SUPERIOR_TO_D__REPORT_AS_MEASURED_NO_CLAIM"
    )
    assert "report the number as measured" in reading["consequence"]
    assert reading["reportable_as_finding"] is False
    assert "NO claim" in reading["consequence"]


def test_matrix_band_boundaries_are_the_t_minus_d_band():
    band = score.aimask_cell_band
    assert band(None) is None
    assert band(0.0) == "APPROX_EQUAL_TO_D"
    assert band(0.02999) == "APPROX_EQUAL_TO_D"
    assert band(-0.02999) == "APPROX_EQUAL_TO_D"
    assert band(0.03) == "SUPERIOR_TO_D"        # boundary inclusive
    assert band(-0.03) == "MUCH_LESS_THAN_D"    # boundary inclusive
    assert band(0.25) == "SUPERIOR_TO_D"
    assert band(-0.25) == "MUCH_LESS_THAN_D"


@pytest.mark.parametrize("am_delta,im_delta", [(0.004, None), (None, -0.11)])
def test_matrix_incomplete_when_only_one_cell_has_landed(am_delta, im_delta):
    reading = score.decomposition_reading(am_delta, im_delta)
    assert reading["matrix_status"] == "matrix_incomplete"
    assert reading["matrix_row"] is None and reading["consequence"] is None
    assert len(reading["cells"]) == 1
    present = next(iter(reading["cells"]))
    assert reading["cells"][present]["delta_external_governing_mean"] == (
        am_delta if am_delta is not None else im_delta
    )
    assert "needs BOTH cells" in reading["matrix_incomplete_note"]
    assert "must not be read against the matrix" in reading["matrix_incomplete_note"]


def test_matrix_absent_when_no_am_im_cell_is_scored():
    reading = score.decomposition_reading(None, None)
    assert reading["matrix_status"] == "no_am_im_cells_scored"
    assert reading["cells"] == {} and reading["matrix_row"] is None


def test_matrix_rule_string_is_recorded_verbatim():
    reading = score.decomposition_reading(-0.10, 0.005)
    assert reading["rule"] == score.AIMASK_MATRIX_RULE
    for fragment in ("|d| < 0.03 -> APPROX_EQUAL_TO_D",
                     "IDENTITY_ABLATION_IS_THE_MECHANISM",
                     "ACTIVITY_SUBSAMPLING_IS_THE_MECHANISM",
                     "JOINT_ABLATION_REQUIRED",
                     "SUSPICIOUS__INVESTIGATE_MASK_AND_EVAL_PATH__DO_NOT_REPORT",
                     "UNANTICIPATED__SUPERIOR_TO_D__REPORT_AS_MEASURED_NO_CLAIM",
                     "ONLY when BOTH AM and IM have landed"):
        assert fragment in reading["rule"], fragment
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    # the rule is also pre-registered in the plan block the receipt carries
    assert '"AM_IM_decomposition_matrix": AIMASK_MATRIX_RULE' in source
    # no collapsed labels: every row has its own consequence, mechanism
    # sentence and reportability, and they are all distinct
    labels = list(score.AIMASK_CONSEQUENCES)
    assert len(labels) == 5 and len(set(labels)) == 5
    for label in labels:
        assert score.AIMASK_CONSEQUENCES[label]
        assert label in score.AIMASK_REPORTABLE_AS_FINDING
        assert label in score.AIMASK_MECHANISM_SENTENCE
    assert score.AIMASK_REPORTABLE_AS_FINDING[score.AIMASK_LABELS["suspicious"]] is False


def test_matrix_framing_note_carries_the_handoff_section_6_constraints():
    reading = score.decomposition_reading(0.0, 0.0)
    assert "THIS ROUND CAN REVOKE IT" in reading["framing_note"]
    assert "carrier remains necessary" in reading["framing_note"]


# ---------------------------------------------------------------------------
# the supersedes threading (handoff §5.3)
# ---------------------------------------------------------------------------
def _write_prior_receipt(root: Path, name: str, supersedes=None) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / score.RECEIPT_BASENAME
    payload = {"schema": "tfpd_subpop_score_v1", "status": "SUBPOP_SCORED"}
    if supersedes is not None:
        payload["supersedes"] = supersedes
    receipt.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return receipt


def test_supersedes_first_run_is_an_empty_list(tmp_path):
    block = score.resolve_supersedes(tmp_path / "results/subpop_score_v1", None)
    assert block["supersedes"] == []
    assert block["supersedes_source"] == "none_first_run"
    assert "first run" in block["supersedes_note"]
    # an unrelated explicit root is also a first run (no family predecessor)
    other = score.resolve_supersedes(tmp_path / "results/somewhere_else", None)
    assert other["supersedes"] == [] and other["supersedes_source"] == "none_first_run"


def test_supersedes_auto_derives_the_family_predecessor(tmp_path):
    root = tmp_path / "results"
    v1 = _write_prior_receipt(root, "subpop_score_v1")
    block = score.resolve_supersedes(root / "subpop_score_v1_r1", None)
    assert block["supersedes"] == [str(v1)]
    assert block["supersedes_source"] == "auto_derived_from_output_root_family"
    assert "cumulative vs corrective" in block["supersedes_note"]


def test_supersedes_auto_chain_is_transitive_and_newest_first(tmp_path):
    root = tmp_path / "results"
    v1 = _write_prior_receipt(root, "subpop_score_v1")
    r1 = _write_prior_receipt(root, "subpop_score_v1_r1", supersedes=[str(v1)])
    r2 = _write_prior_receipt(root, "subpop_score_v1_r2",
                              supersedes=[str(r1), str(v1)])
    block = score.resolve_supersedes(root / "subpop_score_v1_r3", None)
    assert block["supersedes"] == [str(r2), str(r1), str(v1)]


def test_supersedes_auto_chain_stops_at_a_missing_predecessor(tmp_path):
    root = tmp_path / "results"
    _write_prior_receipt(root, "subpop_score_v1")
    (root / "subpop_score_v1_r1").mkdir()  # directory present, receipt absent
    block = score.resolve_supersedes(root / "subpop_score_v1_r2", None)
    assert block["supersedes"] == []
    assert block["supersedes_source"] == "none_first_run"


def test_supersedes_explicit_paths_override_auto_derivation(tmp_path):
    root = tmp_path / "results"
    v1 = _write_prior_receipt(root, "subpop_score_v1")
    elsewhere = _write_prior_receipt(root / "other_lane", "score_v9")
    block = score.resolve_supersedes(
        root / "subpop_score_v1_r1", [str(elsewhere)]
    )
    assert block["supersedes"] == [str(elsewhere)]  # explicit wins over v1
    assert block["supersedes_source"] == "explicit_cli"
    with pytest.raises(SystemExit, match="not found"):
        score.resolve_supersedes(root / "subpop_score_v1_r1",
                                 [str(root / "nope.json")])


def test_supersedes_receipt_fields_are_top_level_and_cli_threaded():
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    for literal in ('"supersedes": supersedes_block["supersedes"]',
                    '"supersedes_note": supersedes_block["supersedes_note"]',
                    '"supersedes_source": supersedes_block["supersedes_source"]',
                    '"--supersedes"', "resolve_supersedes(out_dir, args.supersedes)"):
        assert literal in source, literal
    # end-to-end: a dry-run threads an explicit --supersedes into the plan
    sealed = ROOT / "results/subpop_score_v1_r2/subpop_score_receipt.json"
    if not sealed.is_file():
        pytest.skip("sealed subpop_score_v1_r2 receipt not present")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_subpop_score.py"),
         "--device", "cpu", "--dry-run", "--supersedes", str(sealed)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["supersedes"] == [str(sealed)]
    assert payload["supersedes_source"] == "explicit_cli"
    assert payload["supersedes_note"]


# ---------------------------------------------------------------------------
# faithful AM/IM terminal-receipt fixtures under results/aimask_v1/
# ---------------------------------------------------------------------------
def _aimask_integrity(cell: str) -> dict:
    masked, untouched = (
        ("activity", "identity") if cell == "AM" else ("identity", "activity")
    )
    return {
        "cell": cell,
        "num_heads": 2,
        "mask_structure": score.AIMASK_MASK_STRUCTURE[cell],
        "min_keep": "none",
        "gain": "1/(1-p) inherited from F.dropout",
        "application_site": (
            f"src = {masked} * mask + {untouched} (the whole-unit F.dropout "
            f"mask multiplies the {masked} component only; the {untouched} "
            "component is untouched; the mask tensor is the SAME draw and the "
            "SAME code path as cell D's)"
        ),
        "mask_code_path_matches_D": True,
        "p_distribution_and_clamp_policy": {"clamp": "none", "seed": 42},
        "generator_namespaces": {"unit_mask": "torch.Generator CPU seeded 42003"},
        "eval_mask_policy": (
            "perturbation disabled; the eval path is the literal parent sum "
            "activity + identity, bitwise equal to the parent path"
        ),
        "dropout_structure": {"dynamic_dropout": False, "tf_drop_rate": 0.1},
        "behavior_scaling_convention": (
            "unscaled standardized behavior (exact Arm A replica)"
        ),
    }


def _aimask_diagnostics() -> list[dict]:
    return [
        {"epoch": 47, "perturbation_summary": {
            "n_forwards": 10, "p_raw_min": 0.01, "p_raw_median": 0.49,
            "p_raw_max": 0.99, "kept_fraction_mean": 0.5,
            "surviving_units_min": 0, "surviving_units_q05": 3,
            "surviving_units_median": 61, "surviving_units_mean": 60.2,
            "surviving_units_q95": 118, "surviving_units_max": 143,
            "min_keep_trigger_rate": 0.0, "clamp_trigger_rate": 0.0,
            "mask_code_path_matches_D": True,
        }},
    ]


def _aimask_fixture(tmp_path: Path, cell: str = "AM", tamper: str | None = None,
                    under_root: str | None = None):
    """Faithful landed-AM/IM fixture: 0444 receipt + sidecar + sealed SWA."""
    spec_entry = score.CELL_SPECS[cell]
    root_name = under_root or spec_entry["cell_root"]
    directory = tmp_path / root_name / spec_entry["directory"]
    directory.mkdir(parents=True)
    swa = directory / "swa_final4.pt"
    swa.write_bytes(b"aimask-swa-bytes")
    swa_sha = _sha_file(swa)
    if tamper == "swa_bytes":
        swa.write_bytes(b"tampered")
    integrity = _aimask_integrity(cell)
    payload = {
        "schema": "tfpd_activity_identity_cell_v1",
        "status": "CELL_TERMINAL",
        "cell": cell,
        "cell_name": spec_entry["directory"],
        "smoke": False,
        "max_train_steps": None,
        "epochs_run": 48,
        "invariant_failures": [],
        "source_closure": {"launch_final_closure_equal": True},
        "initial_state": {
            "state_dict_sha256": "canonical-state-sha",
            "artifact_sha256": _sha_file(score.CANONICAL_INITIAL_STATE),
            "strict_load": True,
        },
        "data_contract": {
            "behavior_normalizer_semantic_sha256":
                "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391",
        },
        "integrity": integrity,
        "p_sequence_sha256": SEALED_SHARED_P_SEQUENCE,
        "p_stream_total_draws": 48 * 33925,
        "diagnostics_per_epoch": _aimask_diagnostics(),
        "swa": {"sha256": swa_sha, "window_epochs": [44, 45, 46, 47]},
    }
    if tamper == "status":
        payload["status"] = "CELL_FAILED"
    elif tamper == "smoke_flag":
        payload["smoke"] = True
    elif tamper == "epochs":
        payload["epochs_run"] = 5
    elif tamper == "closure":
        payload["source_closure"]["launch_final_closure_equal"] = False
    elif tamper == "invariants":
        payload["invariant_failures"] = [17]
    elif tamper == "initial_state":
        payload["initial_state"]["state_dict_sha256"] = "0" * 64
    elif tamper == "swa_sha":
        payload["swa"]["sha256"] = "0" * 64
    elif tamper == "num_heads":
        payload["integrity"]["num_heads"] = 64
    elif tamper == "mask_structure":  # swap to the OTHER cell's structure
        payload["integrity"]["mask_structure"] = score.AIMASK_MASK_STRUCTURE[
            "IM" if cell == "AM" else "AM"
        ]
    elif tamper == "gain":
        payload["integrity"]["gain"] = "fixed 1.25 rescaling of survivors"
    elif tamper == "gain_rule":  # the landed runner's field name, tampered
        payload["integrity"]["gain_rule"] = "fixed 1.25 rescaling of survivors"
        del payload["integrity"]["gain"]
    elif tamper == "min_keep":
        payload["integrity"]["min_keep"] = "elementwise clamp"
    elif tamper == "application_site":  # swap to the OTHER cell's site
        other = _aimask_integrity("IM" if cell == "AM" else "AM")
        payload["integrity"]["application_site"] = other["application_site"]
    elif tamper == "mask_code_path":
        payload["integrity"]["mask_code_path_matches_D"] = False
    elif tamper == "mask_code_path_missing":
        del payload["integrity"]["mask_code_path_matches_D"]
    body = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    receipt = directory / "terminal_receipt.json"
    receipt.write_text(body)
    os.chmod(receipt, stat_mod.S_IRUSR | stat_mod.S_IRGRP | stat_mod.S_IROTH)
    sidecar = directory / "terminal_receipt.json.sha256"
    sidecar_text = _sha_file(receipt) + "  terminal_receipt.json\n"
    if tamper == "sidecar":
        sidecar_text = "0" * 64 + "  terminal_receipt.json\n"
    sidecar.write_text(sidecar_text)
    os.chmod(sidecar, stat_mod.S_IRUSR | stat_mod.S_IRGRP | stat_mod.S_IROTH)
    if tamper == "mode":
        os.chmod(receipt, 0o644)
    return receipt


@pytest.mark.parametrize("cell", ["AM", "IM"])
def test_validate_aimask_terminal_passes_on_faithful_fixture(tmp_path, cell):
    _aimask_fixture(tmp_path, cell=cell)
    record = score.validate_cell_terminal(
        cell, _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
    )
    assert record["cell"] == cell
    assert record["cell_name"] == score.CELL_SPECS[cell]["directory"]
    assert record["terminal_receipt"].startswith(
        str(tmp_path / "results/aimask_v1")
    )
    assert record["mode_0444"] and record["non_symlink"]
    assert record["launch_final_closure_equal"] and record["epochs_run"] == 48
    assert record["initial_state_reconciled"]
    assert record["normalizer_reconciled_f062506c"]
    assert record["num_heads"] == 2 and record["p_draws_per_step"] == 1
    assert record["p_sequence_sha256"] == SEALED_SHARED_P_SEQUENCE
    # the cell-law fields are carried in the perturbation-law record
    law = record["perturbation_law"]
    for field in ("mask_structure", "min_keep", "gain", "application_site",
                  "mask_code_path_matches_D"):
        assert field in law, field
    assert law["mask_structure"] == score.AIMASK_MASK_STRUCTURE[cell]
    assert law["mask_code_path_matches_D"] is True
    # no session theta authority is bound for these cells (stated, not absent)
    assert "no separate session authority" in str(
        record["theta_authority_reconciled"]
    )
    # handoff §5.1: the realized perturbation statistics are recorded
    stats = record["realized_perturbation_statistics"]
    assert stats["available"] is True and stats["n_epochs"] == 1
    assert stats["final_epoch"]["p_raw_median"] == 0.49
    assert stats["final_epoch"]["surviving_units_mean"] == 60.2
    assert stats["final_epoch"]["mask_code_path_matches_D"] is True


@pytest.mark.parametrize("cell", ["AM", "IM"])
def test_validate_aimask_missing_receipt_means_not_landed(tmp_path, cell):
    record = score.validate_cell_terminal(
        cell, _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
    )
    assert record is None  # recorded as not_landed and omitted, never fatal


def test_aimask_cells_are_looked_up_under_their_own_root(tmp_path):
    # the same directory under the OLD root is invisible to the AM/IM specs
    _aimask_fixture(tmp_path, cell="AM", under_root="results/subpop_v1")
    assert score.validate_cell_terminal(
        "AM", _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
    ) is None


@pytest.mark.parametrize("cell", ["AM", "IM"])
@pytest.mark.parametrize(
    "tamper",
    ["status", "smoke_flag", "epochs", "closure", "invariants",
     "initial_state", "sidecar", "mode", "swa_sha", "swa_bytes", "num_heads",
     "mask_structure", "gain", "gain_rule", "min_keep", "application_site",
     "mask_code_path", "mask_code_path_missing"],
)
def test_validate_aimask_terminal_refuses_each_tampering(tmp_path, cell, tamper):
    _aimask_fixture(tmp_path, cell=cell, tamper=tamper)
    with pytest.raises(SystemExit):
        score.validate_cell_terminal(
            cell, _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
        )


@pytest.mark.parametrize("cell", ["AM", "IM"])
def test_validate_accepts_the_landed_runner_field_names(tmp_path, cell):
    """The landed runner records the F.dropout gain under ``gain_rule`` and
    the clamp policy under ``p_law`` — the scorer accepts both namings."""
    receipt = _aimask_fixture(tmp_path, cell=cell)
    payload = json.loads(receipt.read_text())
    payload["integrity"]["gain_rule"] = payload["integrity"].pop("gain")
    payload["integrity"]["p_law"] = payload["integrity"].pop(
        "p_distribution_and_clamp_policy"
    )
    os.chmod(receipt, stat_mod.S_IRUSR | stat_mod.S_IWUSR)
    receipt.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    sidecar = Path(str(receipt) + ".sha256")
    os.chmod(sidecar, stat_mod.S_IRUSR | stat_mod.S_IWUSR)
    sidecar.write_text(_sha_file(receipt) + "  terminal_receipt.json\n")
    os.chmod(receipt, stat_mod.S_IRUSR | stat_mod.S_IRGRP | stat_mod.S_IROTH)
    os.chmod(sidecar, stat_mod.S_IRUSR | stat_mod.S_IRGRP | stat_mod.S_IROTH)
    record = score.validate_cell_terminal(
        cell, _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
    )
    assert record["perturbation_law"]["gain_rule"].startswith("1/(1-p)")
    assert record["perturbation_law"]["p_law"]["clamp"] == "none"


def test_realized_perturbation_statistics_aimask_series_tolerates_keys():
    stats = score.realized_perturbation_statistics("IM", [
        {"epoch": 46, "mask_summary": {"kept_fraction_mean": 0.51}},
        {"epoch": 47, "perturbation_summary": {
            "n_forwards": 8, "p_raw_max": 0.98, "clamp_trigger_rate": 0.0}},
    ])
    assert stats["available"] is True and stats["n_epochs"] == 2
    assert stats["per_epoch_series"][0]["kept_fraction_mean"] == 0.51
    assert stats["per_epoch_series"][1]["p_raw_max"] == 0.98
    assert "mask_code_path_matches_D" not in stats["per_epoch_series"][1]
    empty = score.realized_perturbation_statistics("AM", [])
    assert empty["available"] is False


# ---------------------------------------------------------------------------
# the shared one-draw p stream, bound to the sealed R/S2 value
# ---------------------------------------------------------------------------
def test_shared_p_sequence_is_loaded_from_the_sealed_receipt():
    refs = score.load_governing_references(_sha_file)
    shared = refs["shared_one_draw_p_sequence"]
    assert shared["value"] == SEALED_SHARED_P_SEQUENCE
    assert shared["bound_from"] == ["R", "S2"]
    assert shared["receipt_sha256"] == score.SPARSIFY_SCORE_RECEIPT_SHA


def _record(sha, draws=1):
    return {"p_sequence_sha256": sha, "p_draws_per_step": draws}


def test_p_stream_binding_extended_to_aimask():
    good = {
        "T": _record(SEALED_SHARED_P_SEQUENCE),
        "G": _record(SEALED_SHARED_P_SEQUENCE),
        "AM": _record(SEALED_SHARED_P_SEQUENCE),
        "IM": _record(SEALED_SHARED_P_SEQUENCE),
        "C": _record("c" * 64, draws=2),
        "W": {"p_sequence_sha256": "n/a", "p_draws_per_step": 0},
    }
    score.verify_p_stream_binding(good, shared_p_sequence=SEALED_SHARED_P_SEQUENCE)
    score.verify_p_stream_binding({"AM": good["AM"]})  # alone, no sealed value
    score.verify_p_stream_binding({})                  # nothing landed
    # AM off-stream: the shared-stream binding refuses
    with pytest.raises(SystemExit, match="sealed shared"):
        score.verify_p_stream_binding(
            {"AM": _record("0" * 64)}, shared_p_sequence=SEALED_SHARED_P_SEQUENCE
        )
    # AM vs IM disagreement: the one-draw equality refuses
    with pytest.raises(SystemExit, match="one-draw-per-step cells disagree"):
        score.verify_p_stream_binding(
            {"AM": _record("a" * 64), "IM": _record("b" * 64)}
        )
    # the classic T/G message survives the generalization
    with pytest.raises(SystemExit, match="T/G p-stream SHA inequality"):
        score.verify_p_stream_binding(
            {"T": _record("a" * 64), "G": _record("b" * 64)}
        )


# ---------------------------------------------------------------------------
# curve inclusion: AM/IM curved (the wrapper is VALID), W still excluded
# ---------------------------------------------------------------------------
def test_curve_cells_for_curves_am_im_and_excludes_only_w():
    curve_cells, excluded = score.curve_cells_for(["AM", "IM", "W", "T"])
    assert curve_cells == ["AM", "IM", "T"]
    assert set(excluded) == {"W"}
    assert score.CURVE_EXCLUDED_CELLS == {"W": score.CURVE_EXCLUDED_CELLS["W"]}
    # AM/IM never hit the wrong-graph guard; W always does
    score.refuse_cell_curve("AM")
    score.refuse_cell_curve("IM")
    with pytest.raises(SystemExit, match="cell W curve refused"):
        score.refuse_cell_curve("W")
    assert "train-mode-only" in score.AIMASK_CURVE_VALIDITY_NOTE
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    assert '"aimask_curve_validity": AIMASK_CURVE_VALIDITY_NOTE' in source


def _step0c_module():
    return sys.modules.get("tfpd_subpop_step0c_machinery") or score._load_module(
        "tfpd_subpop_step0c_machinery", ROOT / "scripts/run_subpop_step0c.py"
    )


@pytest.mark.skipif(
    not score.ACTIVITY_IDENTITY_CELLS_PATH.is_file(),
    reason="src/tfpd_lane/activity_identity_cells.py has not landed yet",
)
@pytest.mark.parametrize("cell", ["AM", "IM"])
def test_step0c_wrapper_is_valid_for_am_im_because_eval_is_the_parent_sum(cell):
    """AM/IM are the OPPOSITE of W: their mask is train-mode-only, so the
    loaded graph is the exact parent graph and the 0C wrapper reproduces the
    cell's own eval forward bitwise (making the fraction-0 point an exact
    cross-check and the curve well-posed)."""
    step0c = _step0c_module()
    model = score.build_cell_model(None, None, cell).eval()
    # the AM/IM graph adds NO parameters: the canonical Arm A state loads
    # strictly, exactly as for T/C/G
    if score.CANONICAL_INITIAL_STATE.is_file():
        payload = torch.load(
            score.CANONICAL_INITIAL_STATE, map_location="cpu", weights_only=False
        )
        state = {(k[len("model."):] if k.startswith("model.") else k): v
                 for k, v in payload["state_dict"].items()}
        model.load_state_dict(state, strict=True)
    wrapper = step0c.SubpopEvalModel(model, min_keep=step0c.MIN_KEEP).eval()
    torch.manual_seed(0)
    neural = torch.randn(2, 50, 12)
    calib = torch.randn(2, 30, 100, 12)
    side = torch.randn(2, 12, 4)
    with torch.no_grad():
        mine, identity = model(neural, calib_trials=calib, side_features=side)
        wrapped, identity_w = wrapper(neural, calib_trials=calib, side_features=side)
        forward = step0c.curve_forward(
            wrapper, "zero_nogain", 0.25, 0, "zero_nogain_f0.25"
        )
        curve, _ = forward(neural, calib, side)
    assert torch.equal(mine, wrapped)      # the wrapper IS the parent graph
    assert torch.equal(identity, identity_w)
    assert not torch.equal(curve, mine)    # the intervention is live
    assert score.perturbation_disabled(model)


def test_activity_identity_loader_is_lazy_and_fails_named():
    """The module is only touched when a cell lands: building an AM/IM model
    resolves it lazily, and a not-yet-landed module is a named hard failure
    (never an import traceback, never a silent wrong-graph score)."""
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    assert "build_activity_identity_model" in source
    assert "ACTIVITY_IDENTITY_CELLS_PATH" in source
    if score.ACTIVITY_IDENTITY_CELLS_PATH.is_file():
        module = score._activity_identity_module()
        assert hasattr(module, "build_activity_identity_model")
        model = score.build_cell_model(None, None, "AM")
        assert model is not None
    else:
        with pytest.raises(SystemExit, match="has not landed yet"):
            score._activity_identity_module()
        with pytest.raises(SystemExit, match="has not landed yet"):
            score.build_cell_model(None, None, "IM")


# ---------------------------------------------------------------------------
# dry-run: AM/IM are not_landed (their own root) until the operator launches
# ---------------------------------------------------------------------------
def test_dry_run_reports_aimask_not_landed_under_their_own_root():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_subpop_score.py"),
         "--device", "cpu", "--dry-run", "--cells", "AM", "IM"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["cells_requested"] == ["AM", "IM"]
    # the decomposition plan is pre-registered in the dry-run, rule verbatim
    # (the dry-run prints the plan flat, merged with the provenance blocks)
    aimask = payload["aimask_decomposition"]
    assert aimask["cell_root"] == "results/aimask_v1"
    assert aimask["rule"] == score.AIMASK_MATRIX_RULE
    assert "native_only" not in aimask["curve_validity"]
    # AM/IM are never in the native-only set (the wrapper is valid for them)
    assert set(payload["curve"]["native_only_cells"]) == {"W"}
    for cell in ("AM", "IM"):
        if cell in payload["cells_not_landed"]:
            assert f"results/aimask_v1/{score.CELL_SPECS[cell]['directory']}" in (
                payload["not_landed"][cell]["note"]
            )
        else:
            assert payload["landed_cell_provenance"][cell]["swa_sha256"]
            assert payload["landed_cell_provenance"][cell]["epochs_run"] == 48


def test_default_roster_and_supersedes_in_a_bare_dry_run():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_subpop_score.py"),
         "--device", "cpu", "--dry-run"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["cells_requested"] == ["T", "C", "G", "W", "AM", "IM"]
    assert set(payload["cells_landed"]) | set(payload["cells_not_landed"]) == {
        "T", "C", "G", "W", "AM", "IM"}
    # the default root has no family predecessor slot: an explicit first run
    assert payload["supersedes"] == []
    assert payload["supersedes_source"] == "none_first_run"
    assert payload["supersedes_note"]
