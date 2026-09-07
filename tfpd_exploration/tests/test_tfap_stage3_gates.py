"""Gate tests for the third-party debug ticket on run_tfap_stage3.py.

- mechanism-gate truth table: the gate is a pure conjunction of the absolute
  mean threshold and the positive-session count; a positive bootstrap CI lower
  bound can NEVER rescue mean < +0.03, and a large mean can NEVER rescue
  n_positive < 10;
- receipt text matches the implementation (no OR clause; descriptive-only flag);
- preflight_integrity: raises on a non-FINETUNE_TERMINAL stage-2 receipt and on
  a sidecar SHA mismatch, using temp fixtures only — no CUDA, no NWB, no
  datamodule construction (verified structurally: the function body never
  constructs a datamodule and the CLI reaches it before the datamodule code);
- --dry-run returns 0 with DRY_RUN__NO_NWB_OPENED and never runs preflight.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "tfap_stage3_gates_under_test", ROOT / "scripts/run_tfap_stage3.py"
)
stage3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage3)

THRESHOLD = stage3.GATE_THRESHOLD
N_POS = stage3.N_POSITIVE_REQUIRED


def test_mechanism_gate_truth_table_ci_cannot_rescue():
    f = stage3.evaluate_mechanism_gate
    # pass region: both axes clear
    assert f(0.05, 12) is True
    assert f(THRESHOLD, N_POS) is True  # exact boundary is inclusive
    # mean below threshold: False regardless of n_positive (CI escape removed)
    assert f(THRESHOLD - 1e-9, 15) is False
    assert f(0.029, 15) is False
    assert f(-0.5, 15) is False
    # n_positive below requirement: False regardless of mean
    assert f(0.5, N_POS - 1) is False
    assert f(0.5, 0) is False
    assert f(THRESHOLD, 9) is False


def test_receipt_rule_text_matches_implementation():
    source = (ROOT / "scripts/run_tfap_stage3.py").read_text()
    gate_block = source[source.index("def evaluate_mechanism_gate"):]
    gate_block = gate_block[:gate_block.index("def ", 4)]  # next top-level def
    return_line = [l for l in gate_block.splitlines() if l.strip().startswith("return")][0]
    assert " or " not in return_line.lower()
    assert return_line.strip() == (
        "return bool(mean >= GATE_THRESHOLD and n_positive >= N_POSITIVE_REQUIRED)"
    )
    rule_line = [l for l in source.splitlines() if '"rule": "P-T4 - P-Z4' in l][0]
    assert "OR" not in rule_line and "bootstrap" not in rule_line
    assert '"bootstrap_95_interval_is_descriptive_only": True' in source


def _arm_common_stub(tmp_root: Path, swa_sha: str, sidecar_sha: str | None):
    """Minimal arm_common stand-in: real sha256_file, everything else unused."""
    import hashlib

    class Stub:
        @staticmethod
        def sha256_file(path):
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()

    return Stub()


@pytest.fixture()
def finetune_fixture(tmp_path):
    """Stage-2-like tree: SWA artifacts + sidecars + terminal receipts."""
    import hashlib

    root = tmp_path
    models = {}
    for arm, model_key in (("pt4", "P-T4_finetuned_swa"), ("pz4", "P-Z4_finetuned_swa")):
        arm_dir = root / f"results/tfap_stage2_v1/finetune_{arm}"
        arm_dir.mkdir(parents=True)
        blob = f"weights-{arm}".encode()
        swa_sha = hashlib.sha256(blob).hexdigest()
        (arm_dir / "swa_final4.pt").write_bytes(blob)
        (arm_dir / "swa_final4.pt.sha256").write_text(swa_sha + "  swa_final4.pt\n")
        (arm_dir / "terminal_receipt.json").write_text(json.dumps({
            "status": "FINETUNE_TERMINAL", "epochs_run": 48,
            "swa": {"sha256": swa_sha},
        }))
        models[model_key] = arm_dir / "swa_final4.pt"
    return root, models


def test_preflight_integrity_passes_and_returns_sha_map(finetune_fixture):
    root, models = finetune_fixture
    stub = _arm_common_stub(root, "", None)
    shas = stage3.preflight_integrity(models, root, stub)
    assert set(shas) == set(models)
    import hashlib
    assert shas["P-T4_finetuned_swa"] == hashlib.sha256(b"weights-pt4").hexdigest()
    assert shas["P-Z4_finetuned_swa"] == hashlib.sha256(b"weights-pz4").hexdigest()


def test_preflight_integrity_raises_on_non_terminal_receipt(finetune_fixture):
    root, models = finetune_fixture
    receipt = root / "results/tfap_stage2_v1/finetune_pz4/terminal_receipt.json"
    payload = json.loads(receipt.read_text())
    payload["status"] = "FINETUNE_FAILED"
    receipt.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="not terminal"):
        stage3.preflight_integrity(models, root, _arm_common_stub(root, "", None))


def test_preflight_integrity_raises_on_sha_mismatch(finetune_fixture):
    root, models = finetune_fixture
    swa = root / "results/tfap_stage2_v1/finetune_pt4/swa_final4.pt"
    swa.write_bytes(b"tampered-bytes")  # sidecar now disagrees
    with pytest.raises(SystemExit, match="SHA mismatch"):
        stage3.preflight_integrity(models, root, _arm_common_stub(root, "", None))


def test_preflight_integrity_opens_no_datamodule_or_nwb():
    source = (ROOT / "scripts/run_tfap_stage3.py").read_text()
    body = source[source.index("def preflight_integrity"):source.index("def evaluate_mechanism_gate")]
    for forbidden in ("Dandi688", "DataModule", "NWBHDF5IO", "setup("):
        assert forbidden not in body
    # and the CLI calls it before any datamodule construction
    main_body = source[source.index("def main() -> int:"):]
    preflight_pos = main_body.index("preflight_integrity(MODELS, ROOT, arm_common)")
    datamodule_pos = main_body.index("Dandi688MultiSessionDataModule(")
    assert preflight_pos < datamodule_pos


def test_cli_dry_run_returns_zero_without_preflight():
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_tfap_stage3.py"), "--dry-run"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert payload["status"] == "DRY_RUN__NO_NWB_OPENED"
    assert payload["authorized"] is False
