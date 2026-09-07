import json
import subprocess
import sys
import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tfpd_exploration/src/family_runtime_v1/aggregate_m2_family_long.py"

spec = importlib.util.spec_from_file_location("aggregate_m2_family_long", SCRIPT)
aggregate_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(aggregate_module)


def test_m2_long_archive_summary_is_strict_and_reproducible(tmp_path):
    output = tmp_path / "summary.json"
    subprocess.run([sys.executable, str(SCRIPT), "--output", str(output)], cwd=ROOT, check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["status"] == "PASS_ARCHIVE_ONLY_SEALED_AGGREGATION"
    assert result["protocol"] == {"batch": 7, "warmup": 128, "steady_calls_per_receipt": 2048, "threads": {"T1": 1, "T2": 2}}
    assert len(result["receipt_set"]) == 6
    assert result["aggregation_seal"]["aggregator_pre_sha256"] == result["aggregation_seal"]["aggregator_post_sha256"]
    assert all(item["pre_sha256"] == item["post_sha256"] for item in result["receipt_set"])
    for thread in ("T1", "T2"):
        assert result["steady_public_call_ms_by_threads"][thread]["repetitions"] == 3
        for arm in ("FLAT", "ROUTE"):
            for baseline in ("ORIGINALdefault", "ORIGINALdeclared"):
                triplet = result["paired_p95_ratio_by_threads"][thread][arm][f"p95_ratio_to_{baseline}"]
                assert triplet["min"] <= triplet["median"] <= triplet["max"]
        assert len(result["per_repeat_p95_by_threads"][thread]) == 3
    rerun = subprocess.run([sys.executable, str(SCRIPT), "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert rerun.returncode != 0
    assert "refusing to overwrite" in rerun.stderr


def test_m2_long_archive_receipt_corruptions_are_rejected():
    path = aggregate_module.default_receipts()[0]
    receipt = json.loads(path.read_text())
    corruptions = []
    changed_post = copy.deepcopy(receipt)
    changed_post["immutable_hashes_post"]["spint_module"] = "mutated"
    corruptions.append(changed_post)
    negative_oracle = copy.deepcopy(receipt)
    negative_oracle["max_public_vs_full_model_oracle_abs_error"]["FLAT"] = -1.0
    corruptions.append(negative_oracle)
    wrong_thread = copy.deepcopy(receipt)
    wrong_thread["threads"] = 2
    corruptions.append(wrong_thread)
    missing_authority = copy.deepcopy(receipt)
    missing_authority["authority_pre"] = {}
    missing_authority["authority_post"] = {}
    corruptions.append(missing_authority)
    wrong_image = copy.deepcopy(receipt)
    wrong_image["historical_spint_image"] = "sha256:not-the-frozen-image"
    corruptions.append(wrong_image)
    for corrupted in corruptions:
        try:
            aggregate_module._require(corrupted, path, 1, receipt)
        except ValueError:
            continue
        raise AssertionError("corrupted receipt was accepted")
