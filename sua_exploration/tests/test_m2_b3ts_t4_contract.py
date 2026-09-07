import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[2]

def test_b3ts_m24_configs_and_source_gate_are_frozen():
    c=(ROOT/'streaming_calibration_exp/configs/experiment/b3ts_t4_m2_m24_loso_internal.yaml').read_text()
    assert 'streaming_b3ts_t4' in c and 'calibration_n_trials: 24' in c and 'loso_fold: 1' in c
    assert 'include_heldout_in_fit: false' in c and 'include_heldout_in_test: false' in c
    r=(ROOT/'sua_exploration/scripts/run_m2_m24_b3ts_t4_source_one_arm.sh').read_text()
    assert 'data.loso_fold=1' in r and 'data.query_start_trial=0' in r and 'data.include_heldout_in_fit=false' in r
    a=(ROOT/'sua_exploration/scripts/aggregate_m2_m24_b3ts_t4_source.py').read_text()
    assert "'threshold':-0.03" in a and 'formal_heldout_evaluated' in a and 'B3TS' in a

def test_b3ts_feasibility_receipt_records_no_heldout_and_t4_only_budget():
    text=(ROOT/'sua_exploration/results/m2_m24_b3ts_t4_v1/feasibility.json').read_text()
    assert '"formal_heldout_evaluated": false' in text
    assert 'trial-level target-direction T4 labels only' in text


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate_module(tmp_path: Path):
    script = ROOT / 'sua_exploration/scripts/aggregate_m2_m24_b3ts_t4_heldout.py'
    spec = importlib.util.spec_from_file_location('b3ts_heldout_aggregate_test', script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.R = tmp_path
    receipt = tmp_path / 'sua_exploration/results/m2_m24_b3ts_t4_v1/protocol_stage0_and_conditional_expansion.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"frozen": true}\n')
    module.PROTOCOL_SHA = _sha(receipt)
    return module


def _fake_arm_tree(tmp_path: Path):
    module = _aggregate_module(tmp_path)
    screen = module.X
    nwb_dir = tmp_path / 'SPINT-main/data/000953'
    nwb_dir.mkdir(parents=True)
    for session in sorted(module.S):
        (nwb_dir / f'sub-MonkeyN-held-out-calib_{session}_behavior+ecephys.nwb').write_bytes(session.encode())
    nwb_binding = module.current_nwb_binding()
    checkpoints = {}
    for group in module.GROUP:
        checkpoint = tmp_path / f'{group}.ckpt'
        checkpoint.write_bytes(f'{group}-checkpoint'.encode())
        checkpoints[group] = {'path': str(checkpoint.resolve()), 'sha256': _sha(checkpoint)}
    source = tmp_path / f'sua_exploration/results/{screen}/aggregate_source.json'
    source.write_text(json.dumps({'gate': {'pass': True}, 'arms': {g: {'checkpoint': v} for g, v in checkpoints.items()}}))
    source_sha = _sha(source)
    artifacts = {}
    for group, (variant, side) in module.GROUP.items():
        artifact = tmp_path / f'streaming_calibration_exp/outputs/streaming_calibration/{screen}_heldout_{group}_m2_f1_s42_fake'
        artifact.mkdir(parents=True)
        audit = {
            session: {
                'support_trials': 24, 'query_start_trial': 24, 'window_size': 50,
                'full_window_disjoint': True, 'raw_query_start_bin': 100,
                'minimum_window_start_padded_bin': 149, 'eligible_windows': 7, 'query_trials': 3,
            }
            for session in module.S
        }
        (artifact / 'split_manifest.json').write_text(json.dumps({'heldout_query_window_audit': audit}))
        config = {
            'seed': 42, 'train': False, 'test': True, 'ckpt_path': checkpoints[group]['path'],
            'data': {'task': 'm2', 'loso_fold': 1, 'side_feature_group': side, 'calibration_n_trials': 24,
                     'random_calibration': False, 'include_heldout_in_fit': False,
                     'include_heldout_in_test': True, 'query_start_trial': 24},
            'model': {'variant': variant},
        }
        (artifact / 'resolved_config.yaml').write_text(yaml.safe_dump(config))
        with (artifact / 'metrics_per_session.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['split', 'session', 'R2_variance_weighted'])
            writer.writeheader()
            for index, session in enumerate(sorted(module.S)):
                writer.writerow({'split': 'test_heldout', 'session': session, 'R2_variance_weighted': .2 + index / 100})
        provenance = {
            'formal_heldout_evaluated': False, 'local_heldout_calib_evaluated': True,
            'hidden_evalai_evaluated': False, 'group': group,
            'frozen_protocol_receipt_sha256': module.PROTOCOL_SHA,
            'source_aggregate_sha256': source_sha, 'source_checkpoint': checkpoints[group],
            'six_calibration_nwbs': nwb_binding, 'query_contract': audit,
            'label_budget': module.LABEL_BUDGET,
            'no_backward_optimizer_or_checkpoint_selection_on_heldout': True,
        }
        (artifact / 'heldout_b3ts_t4_provenance.json').write_text(json.dumps(provenance))
        artifacts[group] = artifact
    return module, source, checkpoints, artifacts


@pytest.mark.parametrize(
    ('mutation', 'error'),
    [
        ('formal_scope', 'scope'),
        ('config_drift', 'heldout config contract'),
        ('checkpoint_drift', 'source checkpoint bytes'),
        ('window_drift', 'window contract'),
    ],
)
def test_b3ts_heldout_aggregate_rejects_dynamic_artifact_drift(tmp_path, mutation, error):
    module, source, checkpoints, artifacts = _fake_arm_tree(tmp_path)
    artifact = artifacts['t4']
    if mutation == 'formal_scope':
        path = artifact / 'heldout_b3ts_t4_provenance.json'
        payload = json.loads(path.read_text()); payload['formal_heldout_evaluated'] = True; path.write_text(json.dumps(payload))
    elif mutation == 'config_drift':
        path = artifact / 'resolved_config.yaml'
        payload = yaml.safe_load(path.read_text()); payload['model']['variant'] = 'B3TS'; path.write_text(yaml.safe_dump(payload))
    elif mutation == 'checkpoint_drift':
        Path(checkpoints['t4']['path']).write_bytes(b'changed-after-source-receipt')
    else:
        path = artifact / 'split_manifest.json'
        payload = json.loads(path.read_text())
        payload['heldout_query_window_audit'][sorted(module.S)[0]]['window_size'] = 49
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=error):
        module.arm('t4', json.loads(source.read_text())['arms']['t4'], _sha(source))
