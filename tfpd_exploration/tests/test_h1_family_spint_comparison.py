"""Synthetic-only contracts for the H1 original/family timing guard."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tfpd_exploration.src.family_runtime_v1 import h1_family_spint_comparison as comparison
from tfpd_exploration.src.family_runtime_v1.complete_h1_family_source import sha


def _proof(audit, source):
    return {
        "status": "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY", "parameter_updates": 0, "batch": 1,
        "pre_artifact": audit, "post_artifact": audit, "pre_source": source, "post_source": source,
        "arms": {arm: {"selected": audit["selected"][arm], "scored_count": 20325,
                       "public_calls": 20920, "max_archive_abs_error": 1e-6,
                       "max_native_subset_abs_error": 1e-6}
                 for arm in ("flat", "route")},
    }


def test_require_proof_binds_sha_complete_counts_and_all_authority(tmp_path):
    audit = {"selected": {"flat": {"epoch": 5}, "route": {"epoch": 9}}}
    source = {"files": {}, "cache": "known-frozen"}
    path = tmp_path / "proof.json"; path.write_text(json.dumps(_proof(audit, source)))
    result = comparison.require_proof(path, sha(path), audit, source)
    assert result["arms"]["flat"]["scored_count"] == 20325
    variants = []
    bad = _proof(audit, source); bad["arms"]["flat"]["public_calls"] = 20919; variants.append(bad)
    bad = _proof(audit, source); bad["arms"]["route"]["max_native_subset_abs_error"] = 2e-5; variants.append(bad)
    bad = _proof(audit, source); bad["arms"]["flat"]["max_archive_abs_error"] = float("nan"); variants.append(bad)
    bad = _proof(audit, source); bad["arms"]["route"]["max_native_subset_abs_error"] = -1e-9; variants.append(bad)
    bad = _proof(audit, source); bad["post_source"] = {"files": {}, "cache": "changed"}; variants.append(bad)
    bad = _proof(audit, source); bad["pre_artifact"] = {"selected": {}}; variants.append(bad)
    for index, value in enumerate(variants):
        candidate = tmp_path / f"bad-{index}.json"; candidate.write_text(json.dumps(value))
        with pytest.raises(RuntimeError):
            comparison.require_proof(candidate, sha(candidate), audit, source)
    with pytest.raises(RuntimeError, match="SHA drift"):
        comparison.require_proof(path, "0" * 64, audit, source)


def test_assert_public_rejects_alias_nonfinite_and_wrong_h1_contract():
    ok = np.zeros((1, 7), np.float32); comparison.assert_public("ok", ok)
    alias = np.zeros((2, 7), np.float32)[1:]
    nonfinite = ok.copy(); nonfinite[0, 0] = np.nan
    for value in (alias, nonfinite, ok.astype(np.float64), np.zeros((1, 8), np.float32)):
        with pytest.raises(RuntimeError, match="owning native FP32"):
            comparison.assert_public("bad", value)


def test_stats_records_full_quantiles_and_rejects_invalid_samples():
    actual = comparison.stats([1., 2., 3., 4.])
    assert actual == {"mean_ms": 2.5, "p50_ms": 2.5, "p95_ms": pytest.approx(3.85),
                      "p99_ms": pytest.approx(3.97), "max_ms": 4.}
    for values in ([], [0.], [-1.], [np.nan], [np.inf], [[1., 2.]]):
        with pytest.raises(RuntimeError, match="timing samples"):
            comparison.stats(values)


def test_fixed_train_row_cannot_accidentally_use_shorter_minival_sentinel():
    train = np.full((2177, comparison.UNITS), 3, np.float32)
    minival = np.full((1503, comparison.UNITS), 9, np.float32)
    cache = {"train": {comparison.SESSION: {"neural": train, "bank": {}}},
             "minival": {comparison.SESSION: {"neural": minival, "bank": {}}}}
    row, values = comparison.fixed_train_row(cache, 2177)
    assert row is cache["train"][comparison.SESSION]
    assert values.shape == (2177, comparison.UNITS) and values[0, 0] == 3 and values.flags.c_contiguous
    with pytest.raises(RuntimeError, match="cardinality"):
        comparison.fixed_train_row(cache, 2178)


def test_no_go_precedes_formal_audit_image_hash_cache_and_model_paths(tmp_path, monkeypatch):
    output = tmp_path / "not-created.json"
    args = SimpleNamespace(output=output, container_reference=False, image_digest="wrong", calls=2048,
                           warmup=128, threads=1, formal=tmp_path / "not-formal",
                           proof=tmp_path / "not-proof", proof_sha256="0" * 64)
    monkeypatch.delenv("H1_FAMILY_SPINT_COMPARISON_GO", raising=False)
    monkeypatch.setattr(comparison, "artifact_audit", lambda _: pytest.fail("gate audited formal"))
    monkeypatch.setattr(comparison, "sha", lambda _: pytest.fail("gate hashed image/cache"))
    with pytest.raises(RuntimeError, match="comparison gate"):
        comparison.main(args)
    assert not output.exists()


def test_static_original_topology_constants_are_explicit():
    assert comparison.IMAGE.endswith("f719c4228c345f9a1d6aa7c1e10d63ad7b9aa1f551dc95272b0b7a612d61fac6")
    assert (comparison.W, comparison.UNITS, comparison.OUT, comparison.SCALE) == (700, 176, 7, 20.0)
    assert comparison.EXPECTED[comparison.PAYLOAD] == "20a1d41a1d82a8037579caa2e4454f56817e02f021dec2132798c7fc57849298"
    assert comparison.SESSION in str(comparison.ORIGINAL_TAG)
    # Released H1 config uses the entire suffix after _ses- as the mapping key.
    assert comparison.ORIGINAL_TAG.stem.split('_ses-')[-1] == comparison.SESSION.removeprefix('ses-')


def test_image_python_snapshot_binds_every_model_source_and_detects_any_drift(tmp_path, monkeypatch):
    models = tmp_path / "models"; components = models / "components"; components.mkdir(parents=True)
    spint = components / "spint.py"; extra = models / "unimported_helper.py"
    spint.write_text("spint = 1\n"); extra.write_text("helper = 1\n")
    proof = tmp_path / "proof.json"; proof.write_text("{}")
    monkeypatch.setattr(comparison, "MODELS_ROOT", models)
    monkeypatch.setattr(comparison, "EXPECTED", {spint: sha(spint)})
    files = comparison.image_python_files(models)
    assert files == [spint, extra]
    bound = comparison.immutable_snapshot(proof)
    assert str(extra) in bound and str(comparison.Path(__file__).resolve()) not in bound
    extra.write_text("helper = 2\n")
    with pytest.raises(RuntimeError, match="post-replay"):
        comparison.require_same_files(bound)


def test_image_python_topology_refuses_empty_tree(tmp_path):
    with pytest.raises(RuntimeError, match="topology"):
        comparison.image_python_files(tmp_path)
