"""Synthetic contracts for the guarded M1 family/public comparison."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tfpd_exploration.src.family_runtime_v1 import m1_family_spint_comparison as comparison
from tfpd_exploration.src.family_runtime_v1.complete_m1_family_source import array_sha, sha


def _proof(audit):
    source = {"files": {}, "source": "frozen"}
    arms = {
        arm: {"selected": audit["selected"][arm], "scored_count": 31252,
              "public_calls": 112985, "initial_current_predictions": 3,
              "max_abs_error": 1e-6}
        for arm in ("flat", "route")
    }
    return {"status": "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY", "outer_query_opened": False,
            "parameter_updates": 0, "batch": 1, "pre_artifact": audit,
            "post_artifact": audit, "pre_source": source, "post_source": source,
            "arms": arms}


def test_require_proof_requires_frozen_sha_counts_and_pre_post_authority(tmp_path):
    audit = {"selected": {"flat": {"epoch": 7}, "route": {"epoch": 11}}}
    path = tmp_path / "proof.json"; path.write_text(json.dumps(_proof(audit)))
    assert comparison.require_proof(path, sha(path), audit)["arms"]["flat"]["public_calls"] == 112985
    variants = []
    bad = _proof(audit); bad["arms"]["route"]["scored_count"] = 31251; variants.append(bad)
    bad = _proof(audit); bad["pre_source"] = {"files": {}, "source": "different"}; variants.append(bad)
    bad = _proof(audit); bad["post_artifact"] = {"selected": {}}; variants.append(bad)
    for index, body in enumerate(variants):
        changed = tmp_path / f"bad-{index}.json"; changed.write_text(json.dumps(body))
        with pytest.raises(RuntimeError):
            comparison.require_proof(changed, sha(changed), audit)
    with pytest.raises(RuntimeError, match="SHA drift"):
        comparison.require_proof(path, "0" * 64, audit)


def _source_fixture(tmp_path):
    cache, provenance = tmp_path / "cache.npz", tmp_path / "provenance.npz"
    cache_items, provenance_items, raw_receipt, roster_receipt = {}, {}, {}, {}
    for number, name in enumerate(comparison.LANES[:3]):
        raw = np.zeros((4200, 64), np.float32)
        raw[99:] = number + np.arange((4200 - 99) * 64, dtype=np.float32).reshape(-1, 64) / 1000
        roster = np.arange(number * 64, (number + 1) * 64, dtype=np.int64)
        cache_items[f"raw_neural/{name}"] = raw
        cache_items[f"bank_e0/{name}"] = np.full((64, 100), number, np.float32)
        cache_items[f"bank_t/{name}"] = np.full((64, 4), number, np.float32)
        cache_items[f"bank_unit_mask/{name}"] = np.ones(64, bool)
        provenance_items[f"nwb_unit_ids_in_rate_column_order/{name}"] = roster
        raw_receipt[name] = {"raw_neural_sha256": array_sha(raw)}
        roster_receipt[name] = {"nwb_unit_ids_sha256": array_sha(roster)}
    np.savez(cache, **cache_items); np.savez(provenance, **provenance_items)
    return {"cache": str(cache), "provenance": str(provenance),
            "cache_receipt": {"arrays": raw_receipt},
            "provenance_receipt": {"rows": roster_receipt}}


def test_load_source_lanes_uses_three_banks_and_a_separate_fourth_offset(tmp_path):
    source = _source_fixture(tmp_path)
    bank, values = comparison.load_source_lanes(source, 3)
    assert bank.session_ids == comparison.LANES
    assert values.shape == (3, 4, 64) and values.dtype == np.float32 and values.flags.c_contiguous
    assert np.array_equal(bank.E0[0].numpy(), bank.E0[3].numpy())
    # The fourth row deliberately repeats bank 0 but uses the independent +4096 raw segment.
    assert not np.array_equal(values[:, 0], values[:, 3])
    assert values[0, 3, 0] == pytest.approx(262.144)


@pytest.mark.parametrize("kind", ("roster", "startup"))
def test_load_source_lanes_rejects_roster_and_raw_startup_hash_or_content_drift(tmp_path, kind):
    source = _source_fixture(tmp_path)
    if kind == "roster":
        source["provenance_receipt"]["rows"][comparison.LANES[0]]["nwb_unit_ids_sha256"] = "0" * 64
    else:
        with np.load(source["cache"], allow_pickle=False) as saved:
            items = {key: saved[key] for key in saved.files}
        items[f"raw_neural/{comparison.LANES[0]}"][0, 0] = 1
        np.savez(source["cache"], **items)
    with pytest.raises(RuntimeError):
        comparison.load_source_lanes(source, 2)


def test_stats_requires_finite_positive_samples_and_quantiles_are_exact():
    result = comparison.stats(np.array([1., 2., 3., 4.]))
    assert result == {"mean_ms": 2.5, "p50_ms": 2.5, "p95_ms": pytest.approx(3.85),
                      "p99_ms": pytest.approx(3.97), "max_ms": 4.}
    for values in ([], [0.], [-1.], [np.nan], [np.inf]):
        with pytest.raises(RuntimeError, match="timing samples"):
            comparison.stats(values)


def test_assert_public_requires_owning_finite_contiguous_fp32_b4x16():
    value = np.zeros((4, 16), np.float32); comparison.assert_public("ok", value)
    alias = np.zeros((5, 16), np.float32)[1:]
    nonfinite = value.copy(); nonfinite[0, 0] = np.nan
    for invalid in (alias, nonfinite, value.astype(np.float64), np.zeros((1, 16), np.float32)):
        with pytest.raises(RuntimeError, match="owning native FP32"):
            comparison.assert_public("bad", invalid)


def test_main_no_go_gate_precedes_real_paths_audits_and_model_imports(tmp_path, monkeypatch):
    output = tmp_path / "not-created.json"
    args = SimpleNamespace(output=output, container_reference=False, image_digest="not-image",
                           calls=2048, warmup=128, threads=1, run_root=tmp_path / "not-real",
                           proof=tmp_path / "not-real-proof", proof_sha256="0" * 64)
    monkeypatch.delenv("M1_FAMILY_SPINT_COMPARISON_GO", raising=False)
    monkeypatch.setattr(comparison, "sha", lambda _: pytest.fail("gate read a real path"))
    monkeypatch.setattr(comparison, "artifact_audit", lambda _: pytest.fail("gate audited artifacts"))
    with pytest.raises(RuntimeError, match="comparison gate"):
        comparison.main(args)
    assert not output.exists()
