from __future__ import annotations

from scripts import score_cebra_pseudosession_sua as subject


def test_result_paths_are_cell_and_domain_specific() -> None:
    assert subject.result_path("t4", 42, "within_subject") != subject.result_path(
        "t4", 42, "external_subject_M"
    )
    assert subject.result_path("t4", 42, "within_subject") != subject.result_path(
        "z4", 42, "within_subject"
    )


def test_a2_measurement_surface_remains_live() -> None:
    payload = subject.validate_a2_measurement_surface()
    assert payload["formal_subc_test_nwb_opened"] is False


def test_dry_path_never_requires_source_run() -> None:
    path = subject.result_path("t4", 42, "within_subject")
    assert path.name == "within_subject_mix_t4_s42.json"
