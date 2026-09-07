"""Full chronological runtime proof against the once-frozen native ext4 replay."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from tfpd_exploration.src.m2_family_v1 import evaluate_frozen_ext4 as native
from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders
from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_same_query_comparator_v1 import core
from .m2_family_causal import M2FamilyCausalRuntime, M2RuntimeBank
from .complete_m2_family_source import sha, runtime_code, atomic_npz, close

GOLD = native.OUT
GOLD_SHA = 'e99df26b01990ad16c43d27d3e1e4ea3f3157920d864d17c3b8fe4c10917b238'
OUT = native.ROOT / 'tfpd_exploration/results/family_runtime_v1/m2_frozen24_ext4_complete_v1'

def authority():
    if sha(GOLD / 'receipt.json') != GOLD_SHA:
        raise RuntimeError('strict native ext4 oracle receipt drift')
    receipt = json.loads((GOLD / 'receipt.json').read_text())
    current = native.authority()
    if (receipt['status'] != 'COMPLETE_FIXED_SELECTED_EXT4_DEVELOPMENT_ONLY'
            or receipt['pre'] != current or receipt['post'] != current):
        raise RuntimeError('native ext4 authority drift')
    archives = {}
    for arm in ('FLAT', 'ROUTE'):
        p = GOLD / (arm + '_selected_native_float64.npz')
        if sha(p) != receipt['arms'][arm]['native_archive_sha256']:
            raise RuntimeError('strict native archive drift')
        archives[arm] = {'path': str(p), 'sha256': sha(p)}
    return receipt, {'native_authority': current, 'native_receipt_sha256': GOLD_SHA,
                     'archives': archives, 'runtime': runtime_code(), 'self_sha256': sha(Path(__file__))}

def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if os.environ.get('M2_EXT4_COMPLETE_GO') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1') or torch.cuda.is_available():
        raise RuntimeError('explicit GO and CPU-only proof required')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    receipt, pre = authority()
    OUT.mkdir(parents=True)
    native.atomic_json(OUT / 'authority_pre.json', pre)
    started, results, outputs = time.monotonic(), {}, {}
    for arm in ('FLAT', 'ROUTE'):
        record = pre['native_authority']['original_finalizer']['selected'][arm]
        with np.load(pre['archives'][arm]['path'], allow_pickle=False) as z:
            reference = {k: z[k] for k in z.files}
        if set(reference) != {'prediction', 'target', 'start', 'session'} or reference['prediction'].shape != (2069, 2):
            raise RuntimeError('complete native reference fields/count drift')
        pair = make_paired_decoders(42); model = pair[0 if arm == 'FLAT' else 1]; del pair
        model.load_state_dict(torch.load(record['path'], map_location='cpu', weights_only=True), strict=True)
        model.eval()
        offset, public_total, rows, parts = 0, 0, [], []
        for session in plan.EXT4_SESSIONS:
            source = data.load_session_bank('ext4', session, device='cpu')
            raw, starts, target = source.X_store, source.eligible_starts, source.target_store
            count = plan.EXT4_EXPECTED_WINDOWS[session]
            sl = slice(offset, offset + count)
            if (not np.array_equal(reference['target'][sl], target)
                    or not np.array_equal(reference['start'][sl], starts)
                    or not np.array_equal(reference['session'][sl], np.asarray([session] * count))):
                raise RuntimeError('exact session/start/target native archive identity drift')
            bank = M2RuntimeBank(source.E0, source.T, source.unit_mask)
            runtime = M2FamilyCausalRuntime(model, bank)
            endpoints = {int(start + 49): i for i, start in enumerate(starts)}
            observed, visited, maximum = [], [], 0.
            for tick in range(49, len(raw)):
                value = runtime.predict(np.array(raw[tick:tick + 1], dtype=np.float32, copy=True))
                if value.shape != (1, 2) or value.dtype != np.float32 or not value.flags.owndata or not np.isfinite(value).all():
                    raise RuntimeError('public prediction contract drift')
                if tick in endpoints:
                    index = endpoints[tick]
                    if not np.array_equal(runtime.raw[0].numpy(), raw[starts[index]:starts[index] + 50]):
                        raise RuntimeError('independent W50 endpoint raw history mismatch')
                    maximum = max(maximum, close(value, reference['prediction'][offset + index:offset + index + 1], 'native archive'))
                    observed.append(value[0].copy()); visited.append(index)
            if visited != list(range(count)):
                raise RuntimeError('complete ordered endpoint visitation required')
            prediction = np.stack(observed).astype(np.float64)
            rows.append({'session': session, 'public_calls': len(raw) - 49, 'scored_count': count,
                         'max_native_abs_error': maximum, 'r2': core.variance_weighted_r2(target, prediction),
                         'state_bytes': runtime.state_bytes()})
            parts.append(prediction); offset += count; public_total += len(raw) - 49
            print(json.dumps({'arm': arm, 'session': session, 'scored': offset, 'max_native_error': maximum}), flush=True)
        if offset != 2069 or public_total != 10839:
            raise RuntimeError('fixed2069/10839 cardinality drift')
        arrays = {**reference, 'prediction': np.concatenate(parts)}
        pooled = core.variance_weighted_r2(arrays['target'], arrays['prediction'])
        equal = float(np.mean([r['r2'] for r in rows]))
        if abs(pooled - receipt['arms'][arm]['pooled_r2']) > 1e-5 or abs(equal - receipt['arms'][arm]['equal_session_r2']) > 1e-5:
            raise RuntimeError('public/native complete R2 drift')
        outputs[arm] = arrays
        results[arm] = {'rows': rows, 'public_calls': public_total, 'scored_count': offset,
                        'pooled_r2': pooled, 'equal_session_r2': equal,
                        'max_native_abs_error': max(r['max_native_abs_error'] for r in rows)}
    receipt_post, post = authority()
    if pre != post or receipt_post != receipt:
        raise RuntimeError('fresh post-proof authority drift')
    for arm, arrays in outputs.items():
        path = OUT / (arm + '_public_native_float64.npz')
        atomic_npz(path, arrays); results[arm]['archive_sha256'] = sha(path)
    result = {'status': 'PASS_FULL_EXT4_IMPLEMENTATION_EQUIVALENCE_ONLY', 'pre': pre, 'post': post,
              'arms': results, 'elapsed_seconds': time.monotonic() - started,
              'new_selection_or_fitting': False, 'not_official_latency': True, 'threads': 2, 'batch': 1}
    native.atomic_json(OUT / 'receipt.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                     'arms': {a: {k: v for k, v in r.items() if k != 'rows'} for a, r in results.items()}}))

if __name__ == '__main__':
    run()
