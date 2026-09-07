"""Hash-authorized, fixed two-epoch H1 history-treatment worker.

Each worker owns one architecture and two matched conditions. There is no
selection, automatic launch, automatic resume, or promotion path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
RESULTS = ROOT / 'results/family_runtime_v1'
ARMS, CELLS = ('flat', 'route'), ('CONTROL', 'PREFIX')
EPOCHS, UPDATES, MICRO, SEED, LIMIT, MEMORY = 2, 731, 8, 42, 14400, 22 << 30
IDS = ROOT / 'results/decoder_validation_v2/20260905_190000/h1/capacity_probe_208_source_v2/frozen_ids.json'
FIXED_FILES = {
    IDS: 'da4bf975a5c1023bbffe26da87d0d4977f437d7db1db012283511ef140d96211',
    RESULTS / 'h1_selected_source208_diagnostic_v1/receipt.json': 'f3ac9a999fef786ca435e0ff6eefcf679b82e3b3439a535cd66ae6e65c80e614',
    RESULTS / 'h1_selected_cold_segments_v1.json': '379094dab4623444d402a159b625382bc61c320bf399a29d3e07065cfcd7e460',
    RESULTS / 'h1_cold_resource_smoke_flat_v1/receipt.json': 'd3f87a19b9f24a2b6095897ed3fc08b84e692bade4b4e1c07b2a493e31e7b7da',
    RESULTS / 'h1_cold_resource_smoke_route_v1/receipt.json': '5cca9d3b0ba771491b76c6de3c40a778c3fa7ba56ac22a5b3199b4433b3758d2',
}
PROTOCOL = {
    'schema': 'h1_cold_history_phase_per_arm_v2', 'arms': list(ARMS), 'conditions': list(CELLS),
    'epochs': EPOCHS, 'updates_per_epoch': UPDATES, 'source_windows_per_epoch': 23212,
    'microbatch': MICRO, 'effective_batch': 32, 'keep_original_ragged_batches': True,
    'initial': 'own actual selected e12 plain EMA; byte-identical clones within each architecture',
    'optimizer': 'fresh AdamW trusted H1 groups wd=.01 constant lr=1e-5 clip1; no warmup',
    'ema': 'fresh trusted DecoderEMA .9995; first successful update copies RAW',
    'source': 'original ordered epochs1/2 batches and p.1 masks equal frozen formal identities',
    'treatment': 'CONTROL exact clone; PREFIX p.5 retains uniform1..699 right bins else700; zero missing left history only',
    'evaluation': 'e1 fixed source208 RAW+EMA; e2 strict disk checkpoint reload then ALL20325 RAW+EMA FP64 archives',
    'primary': 'fixed e2 EMA ALL20325 pooled R2 PREFIX minus matched CONTROL',
    'segments': 'fixed end<699 count8702 and end>=699 count11623; descriptive secondary',
    'no_selection_or_promotion': True, 'no_official_or_external': True,
    'disclosure': 'post-hoc source-minival-motivated diagnostic, not untouched confirmation',
    'hard_budget_seconds': LIMIT, 'peak_allocated_limit_bytes': MEMORY,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode='w', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(arrays, path):
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def state_digest(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        result.update(name.encode()); result.update(str(array.dtype).encode())
        result.update(np.asarray(array.shape, dtype=np.int64).tobytes()); result.update(array.tobytes())
    return result.hexdigest()


def collect_bindings(formal, output, *, arm, physical_gpu):
    """Read-only authority builder also used by the external authorizer."""
    if not formal.is_absolute() or not output.is_absolute():
        raise RuntimeError('authority paths must be absolute (CLI resolves relative paths)')
    if arm not in ARMS or physical_gpu not in (0, 1):
        raise RuntimeError('arm/GPU authority drift')
    paths = [Path(__file__), SRC/'h1_family_v1/cold_history.py',
             SRC/'family_runtime_v1/complete_h1_family_source.py',
             SRC/'family_runtime_v1/diagnose_h1_selected_source208.py',
             SRC/'h1_optimized_v2/capacity_probe.py',
             SRC/'h1_temporal_decoder_quick_product_v1/ema.py',
             formal/'exports'/f'{arm}_selected_plain_ema.pt', *FIXED_FILES]
    files = {str(path): sha(path) for path in paths}
    for path, expected in FIXED_FILES.items():
        if files[str(path)] != expected:
            raise RuntimeError('fixed prospective evidence SHA drift: '+str(path))
    # Own and auxiliary executable preimages are taken before imports below.
    from tfpd_exploration.src.family_runtime_v1.complete_h1_family_source import artifact_audit, code_source_audit, require_same_files
    artifact, source = artifact_audit(formal), code_source_audit(formal)
    if any(artifact['selected'][name]['epoch'] != 12 for name in ARMS):
        raise RuntimeError('requires actual frozen epoch12 selections')
    identities = {}
    for name in ARMS:
        worker = formal/'workers'/f'{name}_complete.json'
        files[str(worker)] = sha(worker)
        identities[name] = {str(e): read(worker)['identities'][str(e)] for e in (1, 2)}
    if identities['flat'] != identities['route']:
        raise RuntimeError('frozen original arms disagree on source identity')
    smokes = {}
    for name in ARMS:
        smoke = read(RESULTS/f'h1_cold_resource_smoke_{name}_v1/receipt.json')
        if (smoke.get('status') != 'PASS_32_UPDATE_RESOURCE_SMOKE'
                or smoke.get('pre') != smoke.get('post')
                or smoke['pre']['artifact'] != artifact or smoke['pre']['source'] != source
                or smoke['pre']['arm'] != name or smoke.get('updates_per_condition') != 32
                or smoke.get('microbatch') != MICRO or smoke.get('ema_updates') != {k: 32 for k in CELLS}
                or smoke['forecast_seconds']['total_conservative'] >= LIMIT):
            raise RuntimeError('completed paired resource smoke authority drift')
        require_same_files(smoke['pre']['files'])
        smokes[name] = {'identity': smoke['identity_sha256'], 'initial': smoke['initial_state'],
                        'forecast': smoke['forecast_seconds']}
    if smokes['flat']['identity'] != smokes['route']['identity']:
        raise RuntimeError('smoke source/dropout/prefix identities differ')
    return {'formal_artifact': artifact, 'formal_source': source, 'files': files,
            'formal_epoch_identities': identities['flat'], 'smokes': smokes,
            'protocol': PROTOCOL, 'protocol_sha256': digest(PROTOCOL),
            'formal': str(formal), 'output': str(output), 'arm': arm, 'physical_gpu': physical_gpu}


def preflight(formal, output, authorization, *, authorization_sha256, arm, physical_gpu, threads, allow_existing=False):
    if output.exists() and not allow_existing:
        raise FileExistsError(output)
    if (threads != 1 or os.environ.get('H1_COLD_PHASE_GO') != '1'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical_gpu)):
        raise RuntimeError('explicit GO/physical GPU/T1 gate required')
    if len(authorization_sha256) != 64 or sha(authorization) != authorization_sha256:
        raise RuntimeError('external authorization SHA drift')
    auth = read(authorization)
    bindings = collect_bindings(formal, output, arm=arm, physical_gpu=physical_gpu)
    if auth.get('schema') != 'h1_cold_phase_root_authorization_v2' or auth.get('bindings') != bindings:
        raise RuntimeError('external prospective authorization binding drift')
    return bindings


def checkpoint(torch, model, opt, ema, *, epoch, identities):
    if ema.n_updates != epoch*UPDATES:
        raise RuntimeError('checkpoint EMA/update count drift')
    return {'schema': 'h1_cold_phase_epoch_v2', 'raw': model.state_dict(),
            'optimizer': opt.state_dict(), 'ema': ema.checkpoint_state(),
            'raw_digest': state_digest(model.state_dict()), 'ema_digest': state_digest(ema.shadow),
            'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
            'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(),
            'epoch': epoch, 'global_step': epoch*UPDATES, 'identities': identities,
            'protocol_sha256': digest(PROTOCOL)}


def restore(torch, payload, model, opt, ema, epoch, identities):
    if (payload.get('schema') != 'h1_cold_phase_epoch_v2' or payload.get('epoch') != epoch
            or payload.get('global_step') != epoch*UPDATES or payload.get('identities') != identities
            or payload.get('protocol_sha256') != digest(PROTOCOL)
            or payload['ema']['n_updates'] != epoch*UPDATES or payload['ema']['decay'] != .9995
            or state_digest(payload['raw']) != payload['raw_digest']
            or state_digest(payload['ema']['shadow']) != payload['ema_digest']):
        raise RuntimeError('strict checkpoint epoch/identity/state drift')
    model.load_state_dict(payload['raw'], strict=True); opt.load_state_dict(payload['optimizer'])
    ema.load_checkpoint_state(payload['ema'])
    # Trusted EMA load preserves its input device; align CPU-loaded shadows to
    # the active model before any subsequent GPU update.
    device = next(model.parameters()).device
    ema.shadow = {name: tensor.to(device) for name, tensor in ema.shadow.items()}
    if state_digest(model.state_dict()) != payload['raw_digest'] or state_digest(ema.shadow) != payload['ema_digest']:
        raise RuntimeError('strict restored state differs from disk')
    torch.set_rng_state(payload['rng_cpu']); torch.cuda.set_rng_state_all(payload['rng_cuda'])
    np.random.set_state(payload['rng_numpy']); random.setstate(payload['rng_python'])


def checked_score(model, ema, callback):
    before = state_digest(model.state_dict())
    result = callback(model) if ema is None else ema.score_with_ema(model, callback)
    if state_digest(model.state_dict()) != before:
        raise RuntimeError('scoring failed to preserve RAW state')
    return result


def train_update(torch, cells, opts, emas, x, target, bank, keep, *, epoch, batch_id):
    from .cold_history import apply_cold_history
    if (not 1 <= len(x) <= 32 or target.shape != (len(x), 7) or target.dtype != torch.float32
            or keep.shape != (len(x), 176) or keep.dtype != torch.bool
            or not bool(torch.isfinite(target).all())):
        raise RuntimeError('phase effective batch contract')
    full, _ = apply_cold_history(x, seed=SEED, epoch=epoch, batch_id=batch_id, probability=0.)
    prefix, lengths = apply_cold_history(x, seed=SEED, epoch=epoch, batch_id=batch_id, probability=.5)
    losses = {}
    for name, value in (('CONTROL', full), ('PREFIX', prefix)):
        model, opt = cells[name], opts[name]; model.train(); opt.zero_grad(set_to_none=True)
        total = 0.
        for index in range(0, len(x), MICRO):
            n = len(x[index:index+MICRO])
            loss = torch.nn.functional.mse_loss(model.forward_last(value[index:index+MICRO], bank, dropout_keep=keep[index:index+MICRO]), target[index:index+MICRO])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('nonfinite phase loss')
            weight = n/len(x); (loss*weight).backward(); total += float(loss.detach())*weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); emas[name].update_after_step(model); losses[name] = total
    return {'loss': losses, 'cold_rows': int((lengths < 700).sum()), 'rows': len(x),
            '_prefix_lengths': lengths.detach().cpu().numpy()}


def validate_source_batch(row, starts):
    neural, velocity = row['neural'], row['velocity']
    starts = np.asarray(starts)
    if (starts.dtype != np.int64 or starts.ndim != 1 or not 1 <= len(starts) <= 32
            or np.any(starts < 0) or len(np.unique(starts)) != len(starts)
            or neural.dtype != np.float32 or neural.ndim != 2 or neural.shape[1] != 176
            or velocity.dtype != np.float32 or velocity.shape != (len(neural), 7)
            or not np.isfinite(neural).all() or not np.isfinite(velocity).all()
            or np.any(starts+700 > len(neural))
            or not np.all(np.isin(starts, row['query_starts']))):
        raise RuntimeError('source starts/arrays/finite membership drift')
    return np.stack([neural[s:s+700] for s in starts]), np.stack([velocity[s+699] for s in starts])*20


def run(formal, output, authorization, *, authorization_sha256, arm, physical_gpu, threads=1):
    bindings = preflight(formal, output, authorization, authorization_sha256=authorization_sha256,
                         arm=arm, physical_gpu=physical_gpu, threads=threads)
    import copy
    import torch
    if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError('visible cuda:0 unavailable')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED); torch.cuda.reset_peak_memory_stats()
    from tfpd_exploration.src.h1_optimized_v2.cache import CACHE, ROOT as CACHE_ROOT, validate_authority
    from tfpd_exploration.src.h1_optimized_v4.paired_train import batches
    from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
    from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
    from tfpd_exploration.src.h1_family_v1.familyformal_split_train import dropout_keep, sampler_identity_digest, atomic_torch_save, score_complete_cached_ema
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    from tfpd_exploration.src.family_runtime_v1.diagnose_h1_selected_source208 import _require_ids, validate_manifest_against_cache, evaluate_source208
    from tfpd_exploration.src.family_runtime_v1.complete_h1_family_source import check_metadata, validate_archive, metric
    dev = torch.device('cuda:0')
    cache = torch.load(CACHE, map_location='cpu', weights_only=False)
    authority = read(CACHE_ROOT/'source_cache_authority.json'); validate_authority(cache, authority)
    fixed = _require_ids(); validate_manifest_against_cache(cache, authority, fixed)
    state = torch.load(formal/'exports'/f'{arm}_selected_plain_ema.pt', map_location='cpu', weights_only=True)
    if not state or any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for t in state.values()):
        raise RuntimeError('initial export must be finite FP32')
    frozen = state_digest(state)
    pair = make_v2_unscaled_dot_localbalanced_pair(seed=42); base = pair[ARMS.index(arm)]
    base.load_state_dict(state, strict=True)
    cells = {name: copy.deepcopy(base).to(dev) for name in CELLS}; del pair, base, state
    if any(state_digest(model.state_dict()) != frozen for model in cells.values()):
        raise RuntimeError('initial clone/frozen drift')
    if any(value != frozen for value in bindings['smokes'][arm]['initial'].values()):
        raise RuntimeError('initial state differs from actual resource smoke')
    opts = {n: torch.optim.AdamW(groups(m), lr=1e-5, weight_decay=.01) for n, m in cells.items()}
    emas = {n: DecoderEMA(m, decay=.9995) for n, m in cells.items()}
    output.mkdir(parents=True, exist_ok=False)
    atomic({'bindings': bindings, 'authorization_sha256': authorization_sha256, 'initial_state_digest': frozen}, output/'input_authority.json')
    started = time.monotonic(); rows, checkpoints, evaluations, epoch_identities = [], [], {}, {}

    def budget():
        if time.monotonic()-started > LIMIT:
            raise TimeoutError('4h phase wall')
        if torch.cuda.max_memory_allocated() > MEMORY:
            raise MemoryError('22GiB phase threshold')

    def bank_factory(row, device):
        return H1Bank(*[row['bank'][k].to(device) for k in ('E0', 'T', 'unit_mask')])

    class Raw:
        def score_with_ema(self, model, callback):
            return callback(model)

    for epoch in (1, 2):
        ordered = batches(cache, epoch)
        sampler = sampler_identity_digest(ordered)
        expected = bindings['formal_epoch_identities'][str(epoch)]
        if sampler != expected['sampler_sha256'] or sum(len(starts) for _, starts in ordered) != 23212:
            raise RuntimeError('original exact sampler identity/cardinality drift')
        keep_digest, prefix_digest = hashlib.sha256(), hashlib.sha256()
        for batch_id, (session, starts) in enumerate(ordered):
            budget(); row = cache['train'][session]
            x, target = validate_source_batch(row, starts); bank = bank_factory(row, dev)
            keep = dropout_keep(n=len(x), epoch=epoch, batch_index=batch_id, bank_mask=row['bank']['unit_mask'], device=dev)
            item = train_update(torch, cells, opts, emas, torch.as_tensor(x, device=dev), torch.as_tensor(target, device=dev), bank, keep, epoch=epoch, batch_id=batch_id)
            lengths = item.pop('_prefix_lengths')
            for hasher in (keep_digest, prefix_digest):
                hasher.update(session.encode()); hasher.update(np.asarray(starts, dtype=np.int64).tobytes())
            keep_digest.update(keep.cpu().numpy().tobytes()); prefix_digest.update(lengths.tobytes())
            rows.append({'epoch': epoch, 'batch': batch_id, **item}); budget()
            if (batch_id+1) % 32 == 0 or batch_id+1 == UPDATES:
                atomic({'status': 'TRAINING', 'arm': arm, 'epoch': epoch, 'batch': batch_id+1,
                        'global_updates_per_condition': (epoch-1)*UPDATES+batch_id+1,
                        'elapsed_seconds': time.monotonic()-started, 'last': item}, output/'live.json')
        if keep_digest.hexdigest() != expected['keep_sha256']:
            raise RuntimeError('original exact dropout identity drift')
        identity = {'sampler_sha256': sampler, 'keep_sha256': keep_digest.hexdigest(), 'prefix_sha256': prefix_digest.hexdigest()}
        epoch_identities[str(epoch)] = identity
        payload = {n: checkpoint(torch, cells[n], opts[n], emas[n], epoch=epoch, identities=identity) for n in CELLS}
        path = output/f'epoch{epoch:02d}_checkpoint.pt'
        if path.exists():
            raise FileExistsError(path)
        atomic_torch_save(payload, path)
        recorded = sha(path)
        checkpoints.append({'epoch': epoch, 'path': str(path), 'sha256': recorded, 'identities': identity})
        del payload
        # The disk file is authoritative, not a state_dict alias of live tensors.
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        if sha(path) != recorded or set(loaded) != set(CELLS):
            raise RuntimeError('saved checkpoint SHA/cell topology drift')
        for name in CELLS:
            restore(torch, loaded[name], cells[name], opts[name], emas[name], epoch, identity)
        del loaded
        mode_key = 'epoch1_source208' if epoch == 1 else 'epoch2_complete'
        evaluations[mode_key] = {}
        for name in CELLS:
            evaluations[mode_key][name] = {}
            for label in ('RAW', 'EMA'):
                budget()
                atomic({'status': 'SCORING', 'arm': arm, 'epoch': epoch, 'condition': name,
                        'mode': label, 'elapsed_seconds': time.monotonic()-started}, output/'live.json')
                if epoch == 1:
                    callback = lambda candidate: evaluate_source208(candidate, cache, fixed, dev, bank_factory)
                    scores, arrays = checked_score(cells[name], None if label == 'RAW' else emas[name], callback)
                else:
                    active = Raw() if label == 'RAW' else emas[name]
                    callback = lambda candidate: score_complete_cached_ema(model=candidate, ema=active, cache=cache, device=dev)
                    scores = checked_score(cells[name], None, callback)
                    arrays = {k: scores.pop('_'+k) for k in ('prediction', 'target', 'session_id', 'end')}
                    check_metadata(arrays, cache)
                archive = output/f'epoch{epoch:02d}_{name.lower()}_{label.lower()}.npz'
                atomic_npz(arrays, archive)
                if epoch == 2:
                    checked = validate_archive(archive)
                    recomputed = metric(checked)
                    if abs(recomputed['r2_concat_float64']-scores['r2_concat_float64']) > 1e-12:
                        raise RuntimeError('disk archive recomputation drift')
                evaluations[mode_key][name][label] = {**scores, 'archive': {'path': str(archive), 'sha256': sha(archive)}}
                budget()
        atomic({'epochs': epoch_identities, 'checkpoints': checkpoints, 'evaluations': evaluations}, output/'progress.json')
    post = preflight(formal, output, authorization, authorization_sha256=authorization_sha256,
                     arm=arm, physical_gpu=physical_gpu, threads=threads, allow_existing=True)
    if post != bindings:
        raise RuntimeError('fresh post phase authority drift')
    for entry in checkpoints:
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('immutable epoch checkpoint changed')
    for modes in evaluations.values():
        for cell in modes.values():
            for value in cell.values():
                if sha(value['archive']['path']) != value['archive']['sha256']:
                    raise RuntimeError('immutable prediction archive changed')
    result = {'schema': 'h1_cold_history_phase_worker_v2', 'status': 'COMPLETE_NO_SELECTION_OR_PROMOTION',
              'pre': bindings, 'post': post, 'authorization_sha256': authorization_sha256,
              'initial_state_digest': frozen, 'checkpoints': checkpoints, 'identities': epoch_identities,
              'evaluations': evaluations, 'updates': rows, 'ema_updates': {n: e.n_updates for n, e in emas.items()},
              'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'elapsed_seconds': time.monotonic()-started,
              'no_selection_or_promotion': True, 'not_official_or_external': True}
    atomic(result, output/'receipt.json')
    atomic({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}, output/'live.json')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    for key in ('formal', 'output', 'authorization'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--authorization-sha256', required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--physical-gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--threads', type=int, choices=(1,), required=True)
    args = parser.parse_args(argv)
    return run(args.formal.resolve(), args.output.resolve(), args.authorization.resolve(), authorization_sha256=args.authorization_sha256,
               arm=args.arm, physical_gpu=args.physical_gpu, threads=args.threads)


if __name__ == '__main__':
    main()
