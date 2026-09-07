"""Fresh-process H1 V6 public-API benchmark with one fixed native M3 map.

Fixture materialization is a separate process, so full source-cache/NWB data
cannot inflate the measured process RSS. Timed calls include input conversion,
all guards, temporal state, native map application and owning NumPy output.
No fitting or model selection occurs here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ahash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def save(path, body):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + '\n')


def rss():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            k, v, _ = line.split(); values[k[:-1] + '_bytes'] = int(v) * 1024
    values['ru_maxrss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return values


def storage_bytes(values):
    seen = {}
    for x in values:
        storage = x.untyped_storage()
        seen[(str(x.device), storage.data_ptr())] = storage.nbytes()
    return sum(seen.values())


def prepare(args):
    from tfpd_exploration.src.h1_optimized_v2.cache import CACHE, ROOT, validate_authority
    destination = Path(args.fixture)
    if destination.exists() or destination.with_suffix('.json').exists():
        raise FileExistsError(destination)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 'h1_v6_independent_native_score_export_v1':
        raise ValueError('fixture requires the sealed V6 query manifest')
    record = manifest['arms']['t']['epoch12']
    receipt_path = Path(args.receipt)
    receipt = json.loads(receipt_path.read_text())
    if receipt['plain_ema_sha256'] != record['plain_ema_model_state_sha256']:
        raise ValueError('fixture must use exact calibrated epoch12 serialization')
    cache = torch.load(CACHE, map_location='cpu', weights_only=False)
    authority_path = ROOT / 'source_cache_authority.json'
    validate_authority(cache, json.loads(authority_path.read_text()))
    session = sorted(cache['train'])[0]
    row = cache['train'][session]
    if len(row['neural']) < args.fixture_bins:
        raise ValueError('fixture may not cycle or synthesize observations')
    arrays = {'observation': np.array(row['neural'][:args.fixture_bins], dtype=np.float32, order='C', copy=True)}
    arrays.update({name: row['bank'][name].numpy().copy() for name in ('E0', 'T', 'unit_mask')})
    if any(ahash(arrays[name]) != receipt['source_cache_authority']['arrays']['train'][session][key]
           for name, key in (('E0', 'bank_e0_sha256'), ('T', 'bank_hc_sha256'))):
        raise ValueError('fixture bank differs from canonical M3 support bank')
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    body = {'schema': 'h1_v6_public_runtime_fixture_v1', 'session': session, 'split': 'train',
            'bins': len(arrays['observation']), 'start_bin': 0, 'no_cycling': True, 'no_labels': True,
            'source_nwb_sha256': row['sha256'], 'cache_sha256': sha(CACHE),
            'authority_sha256': sha(authority_path), 'manifest': str(manifest_path),
            'manifest_sha256': sha(manifest_path), 'receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path),
            'checkpoint': record['checkpoint'], 'checkpoint_sha256': record['checkpoint_sha256'],
            'plain_ema_state': record['plain_ema_model_state'], 'plain_ema_state_sha256': record['plain_ema_model_state_sha256'],
            'fixture_sha256': sha(destination), 'arrays_sha256': {k: ahash(v) for k, v in arrays.items()}}
    save(destination.with_suffix('.json'), body)
    return body


def load(args):
    from tfpd_exploration.src.h1_optimized_v6.model import H1RecencyPriorQuery
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    from .readout import load_frozen_readout
    path = Path(args.fixture)
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['schema'] != 'h1_v6_public_runtime_fixture_v1' or sha(path) != meta['fixture_sha256']:
        raise ValueError('fixture binding drift')
    for field in ('receipt', 'checkpoint', 'plain_ema_state', 'manifest'):
        if sha(meta[field]) != meta[field + '_sha256']:
            raise ValueError(field + ' binding drift')
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    if any(ahash(v) != meta['arrays_sha256'][k] for k, v in arrays.items()):
        raise ValueError('fixture array drift')
    model = H1RecencyPriorQuery().eval()
    model.load_state_dict(torch.load(meta['plain_ema_state'], map_location='cpu', weights_only=True), strict=True)
    bank = H1Bank(*[torch.from_numpy(arrays[name]) for name in ('E0', 'T', 'unit_mask')])
    readout = load_frozen_readout(meta['receipt'], checkpoint_path=meta['checkpoint'], plain_ema_state_path=meta['plain_ema_state'])
    return meta, arrays['observation'], model, bank, readout


def make_engine(mode, model, bank, readout, session):
    from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.streaming import CurrentQueryStream
    from tfpd_exploration.src.h1_m3_runtime_v1.runtime import M3NativeReadoutStream
    from .query import StaticSignedQueryStream
    from .readout import CompactM3NativeReadoutStream
    query = CurrentQueryStream if mode == 'baseline' else StaticSignedQueryStream
    wrapper = M3NativeReadoutStream if mode == 'baseline' else CompactM3NativeReadoutStream
    return wrapper(query(model, bank, task='h1', session_id=session, unit_ids=range(176)), readout, session_id=session)


def check_value(value):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (1, 7)
            or not value.flags.c_contiguous or not value.flags.owndata or not np.isfinite(value).all()):
        raise AssertionError('public result must be finite owning C-float32 [1,7]')


def check(args):
    meta, obs, model, bank, readout = load(args)
    engines = [make_engine(mode, model, bank, readout, meta['session']) for mode in ('baseline', 'v3')]
    count = min(args.calls, len(obs))
    errors, oracle_errors = [], {}
    points = {0, 1, 3, 4, 5, 698, 699, 700, 703, count - 1}
    with torch.no_grad():
        for step in range(count):
            values = [engine.predict(obs[step:step+1]) for engine in engines]
            for value in values:
                check_value(value)
            if not np.all(np.abs(values[0]-values[1]) <= 1e-5 + 1e-5*np.abs(values[0])):
                raise AssertionError(f'public pair parity failed at {step}')
            errors.append(float(np.max(np.abs(values[0]-values[1]))))
            if step in points:
                raw = np.zeros((1, 700, 176), dtype=np.float32)
                history = obs[max(0, step-699):step+1]
                raw[0, -len(history):] = history
                reference = readout.apply(meta['session'], (model.forward_last(torch.from_numpy(raw), bank)/20).numpy())
                if any(not np.all(np.abs(v-reference) <= 1e-5 + 1e-5*np.abs(reference)) for v in values):
                    raise AssertionError(f'independent full-window oracle parity failed at {step}')
                oracle_errors[str(step)] = max(float(np.max(np.abs(v-reference))) for v in values)
    body = {'schema': 'h1_v6_m3_public_runtime_parity_v1', 'status': 'PASS', 'calls': count,
            'fixture': meta, 'max_native_pair_error': max(errors), 'independent_raw_full_oracle_errors': oracle_errors,
            'public_owning_float32_verified': True, 'not_latency_benchmark': True}
    save(args.output, body)
    return body


def bench(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    initial_rss = rss()
    begin = time.perf_counter_ns()
    meta, obs, model, bank, readout = load(args)
    load_ms = (time.perf_counter_ns()-begin)/1e6
    loaded_rss = rss()
    if 1 + args.warmup + args.calls > len(obs):
        raise ValueError('not enough real observations; cycling forbidden')
    begin = time.perf_counter_ns()
    engine = make_engine(args.mode, model, bank, readout, meta['session'])
    construct_ms = (time.perf_counter_ns()-begin)/1e6
    gc.collect()
    constructed_rss = rss()
    with torch.no_grad():
        begin = time.perf_counter_ns()
        first = engine.predict(obs[0:1])
        first_ms = (time.perf_counter_ns()-begin)/1e6
        check_value(first)
        for step in range(1, 1+args.warmup):
            check_value(engine.predict(obs[step:step+1]))
        timings = []
        for step in range(1+args.warmup, 1+args.warmup+args.calls):
            begin = time.perf_counter_ns()
            value = engine.predict(obs[step:step+1])
            timings.append((time.perf_counter_ns()-begin)/1e6)
            check_value(value)
        end_rss = rss()
        begin = time.perf_counter_ns()
        engine.reset(session_id=meta['session'], bank=bank, unit_ids=range(176))
        reset_ms = (time.perf_counter_ns()-begin)/1e6
    memory = {'engine_reported_state_bytes': engine.engine.state_bytes,
              'model_unique_storage_bytes': storage_bytes(list(model.parameters())+list(model.buffers())),
              'bank_unique_storage_bytes': storage_bytes([bank.E0, bank.T, bank.unit_mask]),
              'all_session_readout_arrays_bytes': sum(v.nbytes for m in readout.maps.values() for v in m.values()),
              'retained_duplicate_checkpoint_bytes': 0 if engine._expected_state is None else storage_bytes(engine._expected_state.values()),
              'fixture_observation_bytes': obs.nbytes,
              'counting_note': 'State, model, bank, maps, duplicate and fixture are separate; RSS includes imports, allocator and metadata.'}
    code_paths = {Path(__file__)}
    for value in (engine, engine.engine, model, model.frontend, model.temporal, engine.engine.memory, engine.engine.front):
        for cls in type(value).__mro__:
            if cls.__module__.startswith('tfpd_exploration.'):
                code_paths.add(Path(inspect.getfile(cls)))
    body = {'schema': 'h1_v6_m3_fresh_process_public_benchmark_v1', 'mode': args.mode, 'fixture': meta,
            'calls': args.calls, 'warmup': args.warmup, 'batch_size': 1, 'torch': torch.__version__,
            'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'affinity': sorted(os.sched_getaffinity(0)), 'artifact_binding_and_model_load_ms': load_ms,
            'construct_and_zero_reset_ms': construct_ms, 'first_public_return_ms': first_ms,
            'subsequent_session_reset_ms': reset_ms,
            'public_predict_ms': {key: float(value) for key, value in
                [('mean', np.mean(timings)), ('p50', np.percentile(timings, 50)), ('p95', np.percentile(timings, 95)),
                 ('p99', np.percentile(timings, 99)), ('max', max(timings))]},
            'memory': memory, 'rss': {'initial': initial_rss, 'loaded': loaded_rss,
                'constructed': constructed_rss, 'after_timing': end_rss},
            'code_sha256': {str(p): sha(p) for p in sorted(code_paths)}, 'raw_ms': timings,
            'scope': 'local CPU native public API; no Falcon wrapper/container/official score claim'}
    save(args.output, body)
    return body


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'check', 'bench'))
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--receipt', type=Path)
    p.add_argument('--fixture-bins', type=int, default=4096)
    p.add_argument('--output', type=Path)
    p.add_argument('--mode', choices=('baseline', 'v3'), default='v3')
    p.add_argument('--calls', type=int, default=704)
    p.add_argument('--warmup', type=int, default=128)
    args = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    body = {'prepare': prepare, 'check': check, 'bench': bench}[args.action](args)
    print(json.dumps({k: v for k, v in body.items() if k not in ('raw_ms', 'fixture', 'code_sha256')}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
