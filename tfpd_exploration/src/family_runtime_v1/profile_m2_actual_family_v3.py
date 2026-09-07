"""Instrumented CPU T1 stage profile; source only, not a SPINT benchmark."""
from __future__ import annotations
import hashlib, json, os, tempfile, time
from pathlib import Path
import numpy as np
import torch
from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders
from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from .complete_m2_family_source import require_final
from .m2_family_causal import M2FamilyCausalRuntime, M2RuntimeBank

ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / 'tfpd_exploration/results/m2/family_v1/finalize_pair_v1'
OUT = ROOT / 'tfpd_exploration/results/m2/family_runtime_v1/m2_actual_family_profile_t1_v3.json'
EXPECTED_FINAL = '3be4a096f98bd9f77726094501271e2836e37c6f61c06ce136ce72505488cf47'
STAGES = ('audit', 'repair', 'temporal_total', 'temporal_first3', 'token_linear2', 'token_norm', 'slot_ffn', 'slot_proj')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def authority():
    if sha(FINAL / 'receipt.json') != EXPECTED_FINAL:
        raise RuntimeError('completed finalizer drift')
    bound = require_final()
    paths = [Path(__file__), FINAL / 'receipt.json']
    paths += [Path(__file__).with_name(name) for name in ('complete_m2_family_source.py', 'm2_family_causal.py',
        'm2_family_spatial.py', 'h1_causal.py', 'linear_conv.py', 'grouped_value.py')]
    receipt = json.loads((FINAL / 'receipt.json').read_text())
    for arm in ('FLAT', 'ROUTE'):
        keys = [k for k in receipt['exports'] if k.startswith(arm + '_selected_')]
        if len(keys) != 1:
            raise RuntimeError('unique selected export required')
        record = receipt['exports'][keys[0]]
        p = FINAL / (keys[0] + '_ema_state.pt')
        if record['export_path'] != str(p) or sha(p) != record['export_sha256']:
            raise RuntimeError('selected export drift')
        paths.append(p)
    for session in plan.HELDIN_SESSIONS:
        directory = data._session_dir('source_train', session)
        paths += [directory / name for name in ('X_store.npy', 'e0_u.pt', 'T.npy', 'mapping.json', 'provenance.json')]
    return {'family_authority': bound, 'files': {str(p): sha(p) for p in paths}}

def run():
    if os.environ.get('M2_PROFILE_V3_GO') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1') or torch.cuda.is_available():
        raise RuntimeError('explicit GO and CPU-only environment required')
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    pre = authority()
    receipt = json.loads((FINAL / 'receipt.json').read_text())
    banks = [data.load_session_bank('source_train', s, device='cpu') for s in plan.HELDIN_SESSIONS]
    bank = M2RuntimeBank(torch.stack([b.E0 for b in banks]), torch.stack([b.T for b in banks]), torch.stack([b.unit_mask for b in banks]))
    raws = [np.load(data._session_dir('source_train', s) / 'X_store.npy', mmap_mode='r') for s in plan.HELDIN_SESSIONS]
    if any(raw.dtype != np.float32 or raw.shape[1] != 96 or len(raw) < 241 or np.any(raw[:49]) for raw in raws):
        raise RuntimeError('source-only W50 stream drift')
    result = {'schema': 'm2_actual_family_profile_t1_v3', 'status': 'INSTRUMENTED_STAGE_PROFILE_NOT_SPINT_COMPARISON',
        'threads': 1, 'batch': 7, 'warmup': 64, 'timed': 128, 'source_sessions': list(plan.HELDIN_SESSIONS),
        'quality_evaluated': False, 'nwb_opened': False, 'minival_integrity_deserialization_only': True,
        'pre': pre, 'arms': {}, 'prior_v2_not_used': 'wrong global call count and invalid temporal accounting'}
    for arm in ('FLAT', 'ROUTE'):
        key = next(k for k in receipt['exports'] if k.startswith(arm + '_selected_'))
        pair = make_paired_decoders(42); model = pair[0 if arm == 'FLAT' else 1]; del pair
        model.load_state_dict(torch.load(FINAL / (key + '_ema_state.pt'), map_location='cpu', weights_only=True), strict=True)
        model.eval(); runtime = M2FamilyCausalRuntime(model, bank, batch_size=7)
        current, counts, samples, originals = {}, {}, [], []
        def instrument(obj, attribute, stage):
            original = getattr(obj, attribute); originals.append((obj, attribute, original))
            def wrapped(*args, **kwargs):
                began = time.perf_counter_ns(); value = original(*args, **kwargs)
                current[stage] = current.get(stage, 0) + time.perf_counter_ns() - began
                counts[stage] = counts.get(stage, 0) + 1
                return value
            setattr(obj, attribute, wrapped)
        instrument(runtime, '_audit', 'audit'); instrument(runtime.spatial, 'repair', 'repair'); instrument(runtime, '_last', 'temporal_total')
        for block in model.temporal.blocks[:3]:
            instrument(block, 'forward', 'temporal_first3')
        for obj, stage in ((model.frontend.token_mlp[2], 'token_linear2'), (model.frontend.token_norm, 'token_norm'),
                           (model.frontend.slot_ffn, 'slot_ffn'), (model.frontend.slot_proj, 'slot_proj')):
            instrument(obj, 'forward', stage)
        max_error, direct_calls = 0., 0
        try:
            for tick in range(192):
                observations = np.ascontiguousarray(np.stack([raw[49 + tick] for raw in raws]))
                current.clear(); counts.clear()
                began = time.perf_counter_ns(); public = runtime.predict(observations); elapsed = time.perf_counter_ns() - began
                if set(current) != set(STAGES) or any(counts[k] != (3 if k == 'temporal_first3' else 1) for k in STAGES):
                    raise RuntimeError('one-call stage cardinality drift')
                if tick >= 64:
                    samples.append({**current, 'predict': elapsed,
                        'temporal_final_remainder': current['temporal_total'] - current['temporal_first3'],
                        'repair_remainder': current['repair'] - sum(current[k] for k in ('token_linear2', 'token_norm', 'slot_ffn', 'slot_proj'))})
                independent = np.ascontiguousarray(np.stack([raw[tick:tick + 50] for raw in raws]))
                if not np.array_equal(runtime.raw.numpy(), independent):
                    raise RuntimeError('independent raw history mismatch')
                if tick in (0, 49, 50, 191):
                    with torch.no_grad():
                        expected = (model.forward_last(torch.from_numpy(independent), bank) / 5).numpy()
                    np.testing.assert_allclose(public, expected, atol=1e-5, rtol=1e-5)
                    max_error = max(max_error, float(np.abs(public - expected).max())); direct_calls += 1
        finally:
            for obj, attribute, original in reversed(originals):
                setattr(obj, attribute, original)
        if len(samples) != 128 or direct_calls != 4:
            raise RuntimeError('global timing/oracle count drift')
        result['arms'][arm] = {'selected_key': key, 'timed_calls': len(samples), 'native_subset_calls': direct_calls,
            'max_native_error': max_error, 'first3_timed_block_calls': 384,
            'mean_ms': {name: float(np.mean([s[name] for s in samples]) / 1e6) for name in samples[0]},
            'p95_ms': {name: float(np.percentile([s[name] for s in samples], 95) / 1e6) for name in samples[0]}}
    post = authority()
    if pre != post:
        raise RuntimeError('profile authority changed')
    result['post'] = post; OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=OUT.parent, mode='w', delete=False) as handle:
        temporary = Path(handle.name); json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, OUT)
    print(json.dumps({'status': result['status'], 'arms': result['arms']}))
    return result

if __name__ == '__main__':
    run()
