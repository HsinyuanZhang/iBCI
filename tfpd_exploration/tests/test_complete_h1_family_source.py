import json
import numpy as np
import pytest
from tfpd_exploration.src.family_runtime_v1 import complete_h1_family_source as proof


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_complete(formal):
    identities = {str(i): {"sampler_sha256": str(i), "keep_sha256": str(i)} for i in range(1, 13)}
    final = {"status": "COMPLETE", "complete": {}}
    freeze = {"selected": {}, "endpoints": {}}
    for arm in proof.ARMS:
        ready = {"shared_sha256": "shared", "identities": identities}
        write_json(formal / "barrier" / (arm + ".ready.json"), ready)
        rows = []
        for epoch in range(1, 13):
            checkpoint = formal / "checkpoints" / f"{arm}_epoch_{epoch:03d}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{arm}:{epoch}".encode())
            selection = {"r2_concat_float64": .5 if epoch in (3, 7) else .1, "finite": True, "n_bins": 2908}
            row = {"epoch": epoch, "checkpoint": str(checkpoint), "checkpoint_sha256": proof.sha(checkpoint), "selection": selection}
            rows.append(row)
            write_json(formal / "workers" / f"{arm}_epoch_{epoch:03d}_metrics.json", row)
        complete = {"status": "EPOCHS_COMPLETE_AWAITING_SUPERVISOR_FREEZE", "arm": arm, **ready,
                    "epochs": rows, "selected_epoch": 3, "selected_ema_r2_float64": .5}
        write_json(formal / "workers" / (arm + "_complete.json"), complete)
        worker = {"status": "COMPLETE_POST_FREEZE", "arm": arm, "reports": {}}
        for label, slot, row in (("selected", "selected", rows[2]), ("epoch12", "endpoints", rows[-1])):
            freeze[slot][arm] = {k: row[k] for k in ("epoch", "checkpoint", "checkpoint_sha256")}
            freeze[slot][arm]["ema_r2_float64"] = row["selection"]["r2_concat_float64"]
            export = formal / "exports" / f"{arm}_{label}_plain_ema.pt"
            archive = formal / "exports" / f"{arm}_{label}_complete_native_float64.npz"
            export.parent.mkdir(parents=True, exist_ok=True)
            export.write_bytes(b"fake-export")
            archive.write_bytes(b"fake-archive")
            worker["reports"][label] = {"epoch": row["epoch"], "checkpoint_sha256": row["checkpoint_sha256"],
                "plain_ema_sha256": proof.sha(export), "complete_archive_sha256": proof.sha(archive),
                "selection_reproduced": row["selection"], "complete": {"n_bins": proof.COUNT}}
        write_json(formal / "workers" / (arm + "_final.json"), worker)
        final["complete"][arm] = worker
    final["selection_freeze"] = freeze["selected"]
    write_json(formal / "final.json", final)
    write_json(formal / "selection_freeze.json", freeze)
    write_json(formal / "input_authority.json", {"fixture": True})
    return final


def test_all24_artifact_audit_and_nonselected_tamper_rejected(tmp_path):
    fake_complete(tmp_path)
    audit = proof.artifact_audit(tmp_path)
    assert audit["selected"]["flat"]["epoch"] == 3
    assert len([p for p in audit["files"] if "/checkpoints/" in p]) == 24
    (tmp_path / "checkpoints/route_epoch_010.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA drift"):
        proof.artifact_audit(tmp_path)


def test_incomplete_before_source_import_or_load(tmp_path, monkeypatch):
    write_json(tmp_path / "final.json", {"status": "TRAINING"})
    monkeypatch.setattr(proof, "code_source_audit", lambda *a: pytest.fail("must not load source"))
    with pytest.raises(RuntimeError, match="incomplete"):
        proof.run(tmp_path, tmp_path / "out")


def test_post_hashes_are_fresh(tmp_path):
    p = tmp_path / "input"
    p.write_bytes(b"first")
    files = {str(p): proof.sha(p)}
    assert proof.require_same_files(files) == files
    p.write_bytes(b"second")
    with pytest.raises(RuntimeError, match="post-replay"):
        proof.require_same_files(files)


def test_earliest_finite_tie_rule():
    rows = [{"epoch": i, "selection": {"r2_concat_float64": .5 if i in (3, 7) else .1}} for i in range(1, 13)]
    assert proof.earliest(rows)["epoch"] == 3
    rows[0]["selection"]["r2_concat_float64"] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        proof.earliest(rows)


def test_archive_shape_precision_and_metadata_guards(tmp_path):
    n = proof.COUNT
    arrays = {"prediction": np.zeros((n, 7), dtype=np.float64), "target": np.ones((n, 7), dtype=np.float64),
              "end": np.arange(n, dtype=np.int64), "session_id": np.asarray(["s"] * n)}
    path = tmp_path / "native.npz"
    np.savez(path, **arrays)
    assert proof.validate_archive(path)["prediction"].shape == (n, 7)
    arrays["prediction"] = arrays["prediction"].astype(np.float32)
    np.savez(path, **arrays)
    with pytest.raises(RuntimeError, match="shape/dtype"):
        proof.validate_archive(path)
    cache = {"minival": {f"s{i:02d}": {"velocity": np.ones((3, 7), dtype=np.float32),
                                     "eval_mask": np.asarray([True, False, True])} for i in range(13)}}
    a = {"target": np.ones((26, 7)), "end": np.tile([0, 2], 13),
         "session_id": np.repeat(sorted(cache["minival"]), 2)}
    proof.check_metadata(a, cache)
    a["target"][0, 0] = 99
    with pytest.raises(RuntimeError, match="target"):
        proof.check_metadata(a, cache)
