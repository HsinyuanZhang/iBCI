"""Fast contracts around training authority, pairing, and encoder handoff."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dandi688_bench_v2 import protocol
from dandi688_bench_v2.common import RECIPE, SCHEMA, PairedSampler, digest, sha256
from dandi688_bench_v2.model import B3SIdentityEncoder
from dandi688_bench_v2.training import load_encoder, load_trained_model


class _Record:
    def __init__(self, session_id: str, endpoints: np.ndarray) -> None:
        self.session_id, self.query_indices = session_id, endpoints


def test_paired_sampler_is_representation_independent_and_exactly_balanced() -> None:
    names = protocol.TRAIN_SESSIONS[:2]
    sua = [_Record(name, np.arange(49, 65, dtype=np.int64)) for name in names]
    pmua = [_Record(name, np.arange(49, 65, dtype=np.int64)) for name in names]
    left, right = PairedSampler(sua, 42, batch=2, updates_per_segment=5), PairedSampler(pmua, 42, batch=2, updates_per_segment=5)
    got_left = [(record.session_id, endpoints.copy()) for record, endpoints in left.segment()]
    got_right = [(record.session_id, endpoints.copy()) for record, endpoints in right.segment()]
    assert [(name, rows.tolist()) for name, rows in got_left] == [(name, rows.tolist()) for name, rows in got_right]
    counts = left.receipt()["segment_session_batch_counts"][0]
    assert max(counts.values()) - min(counts.values()) <= 1


def _write_encoder(path: Path, *, representation: str = "sua", status: str = "SMOKE",
                   schema: str | None = None) -> dict:
    encoder = B3SIdentityEncoder(side_dim=4, seed=42)
    receipt = {"schema": SCHEMA + "_encoder" if schema is None else schema, "protocol": protocol.protocol_dict(),
               "representation": representation, "status": status, "source_sessions": list(protocol.TRAIN_SESSIONS),
               "source_stats_sha256": "stats", "global_step": 2, "encoder_seed": 42,
               "selection": "fixed_final_ema_source_only", "recipe": RECIPE, "code_hashes": {},
               "encoder_line": RECIPE["encoder_line"],
               "encoder_carrier_fusion": RECIPE["encoder_carrier_fusion"],
               "encoder_film": RECIPE["encoder_film"],
               "final_sessions_opened": 0}
    torch.save({"encoder_state": encoder.state_dict(), "receipt_binding": digest(receipt)}, path)
    receipt["checkpoint_sha256"] = sha256(path)
    path.with_suffix(".json").write_text(json.dumps(receipt))
    return {"sha256": "stats"}


def test_load_encoder_rejects_smoke_for_formal_and_representation_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "encoder.pt"
    stats = _write_encoder(path)
    with pytest.raises(ValueError, match="FORMAL"):
        load_encoder(path, representation="sua", stats=stats)
    assert isinstance(load_encoder(path, representation="sua", stats=stats, allow_smoke=True), B3SIdentityEncoder)
    with pytest.raises(ValueError, match="representation"):
        load_encoder(path, representation="pmua", stats=stats, allow_smoke=True)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_encoder(path, representation="sua", stats=stats, allow_smoke=True)


def test_loaders_reject_old_film_encoder_and_checkpoint_schemas_even_for_smoke(tmp_path: Path) -> None:
    old_encoder = tmp_path / "old_encoder.pt"
    stats = _write_encoder(old_encoder, schema="dandi688_v2_2015_move_t4_learned_encoder")
    with pytest.raises(ValueError, match="schema"):
        load_encoder(old_encoder, representation="sua", stats=stats, allow_smoke=True)

    old_checkpoint = tmp_path / "old_checkpoint.pt"
    torch.save({"schema": "dandi688_v2_2015_move_t4_learned_checkpoint", "stage": "train"}, old_checkpoint)
    with pytest.raises(ValueError, match="decoder-training checkpoint"):
        load_trained_model(old_checkpoint, allow_smoke=True)


def test_recipe_and_encoder_receipt_declare_b3s_concat_without_film(tmp_path: Path) -> None:
    assert RECIPE["identity_encoder"] == "B3S"
    assert RECIPE["encoder_line"] == "concat"
    assert RECIPE["encoder_carrier_fusion"] == "post_pool_input_concat"
    assert RECIPE["encoder_film"] is False
    path = tmp_path / "encoder.pt"
    stats = _write_encoder(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    assert receipt["encoder_line"] == "concat"
    assert isinstance(load_encoder(path, representation="sua", stats=stats, allow_smoke=True), B3SIdentityEncoder)
    receipt.pop("encoder_line")
    path.with_suffix(".json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="concat-line"):
        load_encoder(path, representation="sua", stats=stats, allow_smoke=True)
