"""Reproduce every fixed canonical M3 neural support prediction from raw bins.

The producer's support P is a reference only. This replay independently loads
the committed NWB observations, consumes all gap bins from session start, and
uses the actual one-bin public API. An independently implemented OLS is
then refitted to replayed P to verify the resulting corrected exports as well.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.decoder_validation_v2.audit_h1_m3 import AUTH, fit, apply
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import _index_split, load_session_arrays
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.validate_m1_exports import _r2
from .public_benchmark import ahash, sha, save


def run(args):
    from tfpd_exploration.src.h1_optimized_v2.cache import CACHE, validate_authority
    from tfpd_exploration.src.h1_optimized_v6.model import H1RecencyPriorQuery
    from tfpd_exploration.src.h1_optimized_v4.model import H1SignedFull
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    from .query import StaticSignedQueryStream

    output = Path(args.output)
    if output.exists() or output.with_suffix('.npz').exists():
        raise FileExistsError(output)
    receipt_path, manifest_path = Path(args.receipt), Path(args.manifest)
    if sha(receipt_path) != args.expected_receipt_sha256:
        raise ValueError('explicit support receipt binding drift')
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if receipt['schema'] == 'v6_canonical_m3_mat7_readout_v1':
        if manifest['schema'] != 'h1_v6_independent_native_score_export_v1':
            raise ValueError('V6 manifest/readout mismatch')
        arm = 't'
    elif receipt['schema'] == 'v7_canonical_m3_mat7_readout_v1':
        contracts = {'t_v6': 'v6_recency_query_temporal_contract4',
                     'full_v4': 'v4_causal_full_window_no_query_temporal_contract'}
        arm = receipt.get('arm')
        if (manifest['schema'] != 'h1_v7_dropout30_independent_native_score_export_v1' or
                arm not in contracts or receipt.get('operator_contract') != contracts[arm]):
            raise ValueError('V7 manifest/readout/operator mismatch')
    else:
        raise ValueError('only strict V6/V7 support is supported')
    record = manifest['arms'][arm]['epoch12']
    for name, key in (('checkpoint', 'checkpoint_sha256'), ('plain_ema_model_state', 'plain_ema_sha256')):
        if sha(record[name]) != record[name + '_sha256'] or record[name + '_sha256'] != receipt[key]:
            raise ValueError('exact calibrated epoch12 state binding mismatch')
    if (receipt['family'], receipt['ridge'], receipt['scale_floor']) != ('MAT7', 0.0, 1e-6):
        raise ValueError('fixed OLS law drift')
    if sha(AUTH) != receipt['canonical_authority_sha256']:
        raise ValueError('canonical support authority drift')
    canonical = {r['session']: r for r in json.loads(AUTH.read_text())['sessions'] if r['scope'] == 'held-in-calib'}
    if len(canonical) != 13 or set(canonical) != set(receipt['sessions']):
        raise ValueError('support roster mismatch')
    cache = torch.load(CACHE, map_location='cpu', weights_only=False)
    validate_authority(cache, receipt['source_cache_authority'])
    device = torch.device(args.device)
    is_full = arm == 'full_v4'
    if is_full:
        from tfpd_exploration.src.h1_exact_cpu_v1.static_frontend_qonly import StaticCarrierQOnlyExactFullWindowStream
        engine_class = StaticCarrierQOnlyExactFullWindowStream
    else:
        engine_class = StaticSignedQueryStream
    model = (H1SignedFull() if is_full else H1RecencyPriorQuery()).eval()
    model.load_state_dict(torch.load(record['plain_ema_model_state'], map_location='cpu', weights_only=True), strict=True)
    if int(model.frontend_contract_version) != 4 or (not is_full and int(model.temporal.temporal_contract_version) != 4):
        raise ValueError('strict operator contracts changed')
    if is_full and hasattr(model.temporal, 'temporal_contract_version'):
        raise ValueError('FULL must not masquerade as query temporal contract')
    model.to(device)
    paths = _index_split('held-in-calib')
    sessions, maps, arrays = {}, {}, []
    with torch.no_grad():
        for session, official in sorted(canonical.items()):
            source = receipt['sessions'][session]
            support_path = Path(source['support_npz'])
            if sha(support_path) != source['support_npz_sha256']:
                raise ValueError('frozen support file changed')
            with np.load(support_path, allow_pickle=False) as z:
                ends, trial, reference, target = [z[k].copy() for k in
                    ('endpoint', 'trial_id', 'prediction_native_velocity', 'target_native_velocity')]
            raw_sha_before = sha(paths[session])
            if raw_sha_before != official['nwb_sha256'] or raw_sha_before != source['raw_nwb_sha256']:
                raise ValueError('actual source NWB binding mismatch')
            rec = load_session_arrays(paths[session], session, skip_first3=True)
            first = tuple(np.unique(rec.trial_num[np.asarray(rec.eval_mask, dtype=bool)])[:3])
            wanted = np.flatnonzero(np.asarray(rec.eval_mask, dtype=bool) & np.isin(rec.trial_num, first))
            if (first != tuple(official['calibration_trials']) or not np.array_equal(ends, wanted) or
                    not np.array_equal(trial, rec.trial_num[ends]) or not np.array_equal(target, rec.velocity[ends])):
                raise ValueError('raw canonical support endpoints/targets mismatch')
            raw_bank = cache['train'][session]['bank']
            authority = receipt['source_cache_authority']['arrays']['train'][session]
            if (ahash(raw_bank['E0'].numpy()) != authority['bank_e0_sha256'] or
                    ahash(raw_bank['T'].numpy()) != authority['bank_hc_sha256'] or
                    ahash(raw_bank['unit_mask'].numpy()) != manifest['per_session_bank_unit_mask_sha256']['train'][session]):
                raise ValueError('canonical neural-support bank binding mismatch')
            bank = H1Bank(*[raw_bank[k].to(device) for k in ('E0', 'T', 'unit_mask')])
            engine = engine_class(model, bank, task='h1', session_id=session, unit_ids=range(176))
            predicted = np.empty_like(reference)
            cursor = 0
            for step in range(int(ends[-1]) + 1):
                observation = np.asarray(rec.neural[step:step+1], dtype=np.float32)
                if cursor < len(ends) and step == ends[cursor]:
                    value = engine.predict(observation)
                    if value.shape != (1, 7) or value.dtype != np.float32 or not np.isfinite(value).all():
                        raise ValueError('invalid public neural support output')
                    predicted[cursor] = value[0]
                    cursor += 1
                else:
                    engine.observe(observation)
            if cursor != len(ends) or not np.all(np.abs(predicted-reference) <= 1e-5 + 1e-5*np.abs(reference)):
                raise AssertionError('public neural-support parity mismatch')
            delta = _r2(predicted, target) - _r2(reference, target)
            if abs(delta) > 1e-5:
                raise AssertionError('neural support R2 mismatch')
            if sha(paths[session]) != raw_sha_before:
                raise ValueError('source changed during actual streaming replay')
            maps[session] = fit(predicted, target)
            arrays.append({'session_id': np.full(len(ends), session), 'endpoint': ends,
                           'prediction_native_velocity': predicted, 'target_native_velocity': target})
            sessions[session] = {'support_bins': len(ends), 'observed_bins': int(ends[-1])+1,
                                 'raw_nwb_sha256': raw_sha_before, 'raw_source_posthash_equal': True,
                                 'max_native_error': float(np.max(np.abs(predicted-reference))),
                                 'r2_reference': _r2(reference, target), 'r2_replayed': _r2(predicted, target),
                                 'r2_delta': delta, 'support_reference_sha256': sha(support_path),
                                 'replayed_prediction_sha256': ahash(predicted)}
            print(json.dumps({'session': session, **sessions[session]}), flush=True)
    if sum(s['support_bins'] for s in sessions.values()) != 30879:
        raise AssertionError('support total mismatch')
    exports = {}
    for label, row in receipt['applied_exports'].items():
        if sha(row['source_npz']) != row['source_sha256'] or sha(row['npz']) != row['sha256']:
            raise ValueError('frozen native/corrected export binding changed')
        with np.load(row['source_npz'], allow_pickle=False) as source, np.load(row['npz'], allow_pickle=False) as original:
            native, target, ids = source['prediction_native_velocity'], source['target_native_velocity'], source['session_id']
            transformed = np.empty_like(native)
            for session, mapping in maps.items():
                mask = ids == session
                transformed[mask] = apply(mapping, native[mask])
            frozen = original['prediction_native_velocity']
            error = float(np.max(np.abs(transformed-frozen)))
            delta = _r2(transformed, target) - _r2(frozen, target)
            session_delta = {s: _r2(transformed[ids == s], target[ids == s]) - _r2(frozen[ids == s], target[ids == s]) for s in sessions}
            if (not np.all(np.abs(transformed-frozen) <= 1e-5+1e-5*np.abs(frozen)) or
                    max([abs(delta)] + list(map(abs, session_delta.values()))) > 1e-5):
                raise AssertionError('independent OLS from replayed P changes corrected exports')
            exports[label] = {'max_native_error': error, 'pooled_r2_delta': delta, 'session_r2_delta': session_delta}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), **{k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
    code_paths = {Path(__file__)}
    for value in (engine, model, model.frontend, model.temporal, engine.front, engine.memory):
        for cls in type(value).__mro__:
            if cls.__module__.startswith('tfpd_exploration.'):
                code_paths.add(Path(inspect.getfile(cls)))
    body = {'schema': 'h1_m3_actual_neural_support_public_replay_v1', 'status': 'PASS', 'arm': arm,
            'receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path), 'manifest_sha256': sha(manifest_path),
            'device': args.device, 'not_latency_benchmark': True, 'support_total': 30879,
            'no_gaps_omitted': True, 'trial_resets': False, 'sessions': sessions,
            'independent_ols_refit_from_replayed_P_export_composition': exports,
            'npz': str(output.with_suffix('.npz')), 'npz_sha256': sha(output.with_suffix('.npz')),
            'code_sha256': {str(path): sha(path) for path in sorted(code_paths)},
            'scope': 'raw held-in canonical support neural P reproduction plus independent OLS composition; no model/readout selection'}
    save(output, body)
    return body


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    result = run(arguments)
    print(json.dumps({'status': result['status'], 'support_total': result['support_total'], 'output': str(arguments.output)}), flush=True)
