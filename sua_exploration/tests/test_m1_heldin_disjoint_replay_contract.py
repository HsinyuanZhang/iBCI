"""Static fail-closed contract checks for the M1 held-in replay correction scripts."""
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_SCRIPT = ROOT / "sua_exploration/scripts/write_m1_heldin_disjoint_replay_correction_provenance.py"


def _load_provenance_module():
    spec = importlib.util.spec_from_file_location("m1_heldin_provenance", PROVENANCE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_heldin_query_start_and_never_reads_contaminated_aggregate() -> None:
    text = (ROOT / "sua_exploration/scripts/run_m1_heldin_disjoint_replay_correction_one_arm.sh").read_text()
    assert "data.heldin_query_start_trial=10" in text
    assert "train=false" in text and "test=true" in text
    assert "aggregate_m1.json" not in text


def test_aggregate_requires_an_explicit_immutable_seal_list() -> None:
    text = (ROOT / "sua_exploration/scripts/aggregate_m1_heldin_disjoint_replay_correction.py").read_text()
    assert 'parser.add_argument("--seal-list", type=Path, required=True' in text
    assert "explicit seal SHA" in text
    assert "glob(" not in text


def test_provenance_runtime_data_whitelist_contains_exactly_intended_keys() -> None:
    module = _load_provenance_module()
    whitelist = module.RUNTIME_DATA_WHITELIST_OVERRIDES
    assert set(whitelist) == {
        "query_start_trial",
        "heldin_query_start_trial",
        "allow_empty_heldout_query",
    }
    assert whitelist["query_start_trial"] == 0
    assert whitelist["heldin_query_start_trial"] == module.M
    assert whitelist["allow_empty_heldout_query"] is False
