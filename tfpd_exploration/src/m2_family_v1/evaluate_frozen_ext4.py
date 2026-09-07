"""One fixed post-selection native ext4 development comparison, not selection.

Only the original completed24 selected FLAT/ROUTE exports are eligible.
No cold-phase checkpoint, new fit, cache build, official endpoint, or alternate
epoch is evaluated. Existing support33 banks and historical scores are bound.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from .decoder import make_paired_decoders
from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_same_query_comparator_v1 import core
from tfpd_exploration.src.family_runtime_v1.complete_m2_family_source import require_final, validate_arrays

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / 'tfpd_exploration/results/m2/family_v1'
PROSPECTIVE = FAMILY / 'frozen24_selected_ext4_prospective_v1'
OUT = FAMILY / 'frozen24_selected_ext4_native_v1'
BASELINE = FAMILY / 'ext4_e8_spint_dev_pooled_addendum_v1.json'
BASELINE_SHA = 'a10963c3b87081fa7f401eeed1ab9c6829c613e479abe835acc0ff43405649e0'
FINAL_SHA = '3be4a096f98bd9f77726094501271e2836e37c6f61c06ce136ce72505488cf47'
PROTOCOL = {'schema': 'm2_frozen24_selected_ext4_native_v1', 'arms': ['FLAT', 'ROUTE'],
    'picks': {'FLAT': 2, 'ROUTE': 6}, 'selection': 'original all24 source-minival EMA picks, already frozen; never reselect',
    'surface': 'existing local ext4 development2069, support33 from held-out-calib; fixed disjoint post-support windows',
    'architecture_or_weight_change': False, 'calibration_fits': 0, 'batch': 8, 'native_divisor': 5,
    'reported': 'all2069 pooled and equal-four-session native R2, every session, delta vs bound historical e8/SPINT',
    'qualification': 'development comparison with historical external exposure, not unbiased heldout or formal noninferiority',
    'forbidden': ['cold-phase candidates', 'alternative epochs', 'selection or promotion', 'official/hidden', 'cache rebuild'],
    'device': 'physical GPU0', 'hard_budget_seconds': 180}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def atomic_json(p, value):
    if p.exists():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=p.parent, mode='w', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, p)

def verify_geometry(raw, starts, target, mapping, sealed, expected):
    validate_arrays(raw, starts, target, expected)
    if (core.array_sha256(starts) != sealed['ordered_window_starts_sha256']
            or core.array_sha256(target) != sealed['target_sha256']
            or mapping['support_horizon'] != 33 or len(mapping['support_trial_ids']) != 33
            or mapping['apply_disjoint_on_this_query_file'] is not True
            or mapping['eligible_starts_padded'] != starts.tolist()
            or mapping['query_window_audit']['full_window_disjoint'] is not True
            or int(starts.min()) < mapping['support_boundary_padded_bin']):
        raise RuntimeError('fixed ext4 target/start/support/disjoint contract drift')

def authority():
    receipt, frozen = require_final()
    if frozen['receipt_sha256'] != FINAL_SHA or frozen['complete_training_audit']['picks'] != PROTOCOL['picks']:
        raise RuntimeError('original completed24 fixed picks required')
    if sha(BASELINE) != BASELINE_SHA:
        raise RuntimeError('historical ext4 baseline drift')
    baseline = json.loads(BASELINE.read_text())
    if baseline['sessions'] != list(plan.EXT4_SESSIONS) or baseline['status'] != 'DEVELOPMENT_COMPARISON_NOT_UNBIASED_HELDOUT':
        raise RuntimeError('exact ext4 development baseline required')
    paths = [Path(__file__), BASELINE, Path(data.__file__), Path(plan.__file__), Path(core.__file__)]
    paths.append(ROOT / 'tfpd_exploration/src/family_runtime_v1/complete_m2_family_source.py')
    rows = {}
    for session, sealed in zip(plan.EXT4_SESSIONS, baseline['rows'], strict=True):
        if sealed['session'] != session:
            raise RuntimeError('sealed session order drift')
        directory = data.cache_root() / 'ext4' / session
        names = ('X_store.npy', 'target_store.npy', 'eligible_starts.npy', 'e0_u.pt', 'T.npy',
                 'calib_activity.npy', 'mapping.json', 'provenance.json', 'extra.json')
        if any(not (directory / name).is_file() for name in names):
            raise RuntimeError('existing compact cache required; never build')
        paths.extend(directory / name for name in names)
        raw = np.load(directory / 'X_store.npy', mmap_mode='r')
        target = np.load(directory / 'target_store.npy', mmap_mode='r')
        starts = np.load(directory / 'eligible_starts.npy')
        mapping = json.loads((directory / 'mapping.json').read_text())
        verify_geometry(raw, starts, target, mapping, sealed, plan.EXT4_EXPECTED_WINDOWS[session])
        rows[session] = {'windows': len(starts), 'raw_bins_after49pad': len(raw) - 49,
                         'mapping': mapping, 'provenance': json.loads((directory / 'provenance.json').read_text())}
    if sum(r['windows'] for r in rows.values()) != 2069:
        raise RuntimeError('fixed2069 surface required')
    return {'protocol': PROTOCOL, 'original_finalizer': frozen, 'baseline': baseline,
            'files_sha256': {str(p): sha(p) for p in paths}, 'rows': rows}

def prepare():
    bound = authority()
    atomic_json(PROSPECTIVE / 'preflight.json', {'status': 'PREPARED_NO_MODEL_FORWARD', 'authority': bound,
        'arrays_opened_for_identity_validation': True, 'quality_computed': False, 'new_fit': False})
    print(json.dumps({'status': 'PREPARED_NO_MODEL_FORWARD', 'windows': 2069, 'arms': PROTOCOL['picks']}))

def run(authorization):
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.environ.get('M2_FROZEN_EXT4_GO') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or not torch.cuda.is_available():
        raise RuntimeError('explicit review GO and planned physical GPU0 required')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    pre = authority()
    auth = json.loads(authorization.read_text())
    if (auth.get('status') != 'ROOT_REVIEW_GO' or auth.get('authority') != pre
            or auth.get('preflight_sha256') != sha(PROSPECTIVE / 'preflight.json')):
        raise RuntimeError('exact frozen prospective authority required')
    OUT.mkdir(parents=True)
    atomic_json(OUT / 'input_authority.json', {'authority': pre, 'authorization_sha256': sha(authorization)})
    started = time.monotonic(); results, arrays_all = {}, {}
    torch.cuda.reset_peak_memory_stats()
    for arm in ('FLAT', 'ROUTE'):
        record = pre['original_finalizer']['selected'][arm]
        pair = make_paired_decoders(42); model = pair[0 if arm == 'FLAT' else 1]; del pair
        state = torch.load(record['path'], map_location='cpu', weights_only=True)
        if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):
            raise RuntimeError('selected plain EMA finite FP32 drift')
        model.load_state_dict(state, strict=True); model.to('cuda:0').eval()
        parts = {k: [] for k in ('prediction', 'target', 'start', 'session')}
        rows, direct_error = [], 0.
        with torch.no_grad():
            for session in plan.EXT4_SESSIONS:
                bank = data.load_session_bank('ext4', session, device='cuda:0')
                starts, raw, target = bank.eligible_starts, bank.X_store, bank.target_store
                prediction = []
                for off in range(0, len(starts), 8):
                    if time.monotonic() - started > 180:
                        raise TimeoutError('fixed native replay budget')
                    x = torch.from_numpy(np.stack([raw[i:i + 50] for i in starts[off:off + 8]])).to('cuda:0')
                    value = (model.forward_last(x, bank) / 5).cpu().numpy()
                    if off == 0:
                        singleton = (model.forward_last(x[:1], bank) / 5).cpu().numpy()
                        np.testing.assert_allclose(value[:1], singleton, atol=1e-5, rtol=1e-5)
                        direct_error = max(direct_error, float(np.abs(value[:1] - singleton).max()))
                    prediction.append(value.astype(np.float64))
                p = np.concatenate(prediction)
                if p.shape != target.shape or not np.isfinite(p).all():
                    raise RuntimeError('finite complete native prediction required')
                score = core.variance_weighted_r2(target, p)
                baseline = next(r for r in pre['baseline']['rows'] if r['session'] == session)
                rows.append({'session': session, 'windows': len(starts), 'r2': score,
                             'delta_vs_spint': score - baseline['spint_r2'], 'delta_vs_e8': score - baseline['e8_r2']})
                for key, value in (('prediction', p), ('target', np.asarray(target, dtype=np.float64)),
                                   ('start', starts), ('session', np.asarray([session] * len(starts)))):
                    parts[key].append(value)
        arrays = {k: np.concatenate(v) for k, v in parts.items()}
        pooled = core.variance_weighted_r2(arrays['target'], arrays['prediction'])
        equal = float(np.mean([r['r2'] for r in rows]))
        results[arm] = {'selected': record, 'rows': rows, 'pooled_r2': pooled, 'equal_session_r2': equal,
            'pooled_delta_vs_spint': pooled - pre['baseline']['spint_pooled_r2'],
            'equal_delta_vs_spint': equal - pre['baseline']['spint_equal_session_r2'],
            'pooled_delta_vs_e8': pooled - pre['baseline']['e8_pooled_r2'],
            'equal_delta_vs_e8': equal - pre['baseline']['e8_equal_session_r2'], 'max_batched_singleton_error': direct_error}
        arrays_all[arm] = arrays
        model.cpu(); del model
    post = authority()
    if pre != post:
        raise RuntimeError('fresh post-model/data/selection authority drift')
    for arm, arrays in arrays_all.items():
        path = OUT / (arm + '_selected_native_float64.npz')
        with tempfile.NamedTemporaryFile(dir=OUT, suffix='.npz', delete=False) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays); os.replace(temporary, path)
        results[arm]['native_archive_sha256'] = sha(path)
    result = {'status': 'COMPLETE_FIXED_SELECTED_EXT4_DEVELOPMENT_ONLY', 'pre': pre, 'post': post, 'arms': results,
        'elapsed_seconds': time.monotonic() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'parameter_updates': 0, 'new_calibration_fits': 0, 'epoch_or_model_selection': None, 'promotion': None}
    atomic_json(OUT / 'receipt.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                     'scores': {a: {k: r[k] for k in ('pooled_r2', 'equal_session_r2', 'pooled_delta_vs_spint')} for a, r in results.items()}}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    if args.prepare == (args.authorization is not None):
        parser.error('choose --prepare or --authorization PATH')
    prepare() if args.prepare else run(args.authorization)
