#!/usr/bin/env python3
"""Read-only CPU audit of production M2 MOVE-T4 M33 support stability.

The current joint-RIFT M2 encoder is bound to canonical p0 FiLM plus the
selected EMPTY head.  This audit measures its production MOVE-T4 input only:
no decoder forward, loss, target query, optimizer, GPU, or hidden-test access.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
for p in (WS, WS / 'tfpd_exploration', WS / 'streaming_calibration_exp', ROOT / 'src', WS / 'btransform_unified_v1' / 'src'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from tfpd_exploration.src.m2_dual_track_v1 import champion, data, plan
from tfpd_exploration.src.m2_hold_film_probe_v1.encoder import HoldContrastFiLMEarlyPoolEncoder

CACHE = WS / 'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache'
FILM = WS / 'tfpd_exploration/results/m2_hold_film_probe_v1/film_states.pt'
HEAD = WS / 'tfpd_exploration/results/m2_movement_t4_empty_epoch_pick_v1/selected_head.pt'
DEFAULT_OUT = ROOT / 'results/diagnostics_v1/m2_carrier_support_stability_v1'
BUDGETS = (8, 16, 25, 33); SEEDS = tuple(range(101, 111))

def fsha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def asha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def normalizer() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    body = json.loads((CACHE / 'move_t4_normalizer.json').read_text())
    return np.asarray(body['mean'], np.float32), np.asarray(body['std'], np.float32), body

def selected_carrier(bundle: dict[str, np.ndarray], ids: np.ndarray, *, mean: np.ndarray, std: np.ndarray, session: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact production T4 formula, but selects rows before `t4_from_trial_sums`."""
    champion.ensure_streaming_on_path()
    from src.data.falcon_t4_features import t4_from_trial_sums
    neural = np.asarray(bundle['calib_neural'], np.float32)
    changes = np.asarray(bundle['calib_trial_change'], bool)
    angles = np.asarray(bundle['angles'], np.float32)
    starts = np.flatnonzero(changes); ends = np.r_[starts[1:], len(neural)]
    if len(starts) < 33 or angles.shape[0] < 33: raise RuntimeError(f'{session}: lacks raw M33 trials')
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids) or int(ids.min()) < 0 or int(ids.max()) >= 33: raise ValueError('invalid support subset')
    sums, lengths = [], []
    for i in ids:
        begin, end = int(starts[i]), int(ends[i])
        if end - begin < plan.MOVE_T4_STOP_BIN: raise RuntimeError(f'{session} trial {i}: short raw trial')
        sums.append(neural[begin + plan.MOVE_T4_START_BIN: begin + plan.MOVE_T4_STOP_BIN].sum(axis=0, dtype=np.float64))
        lengths.append(plan.MOVE_T4_STOP_BIN - plan.MOVE_T4_START_BIN)
    raw = t4_from_trial_sums(np.asarray(sums, np.float32), np.asarray(lengths, np.int64), angles[ids], source=f'{session}:selected-MOVE-T4')
    carrier = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
    theta = angles[ids].astype(np.float64, copy=False)
    theta = theta[np.isfinite(theta)]
    design = np.stack([np.ones(len(theta)), np.cos(theta), np.sin(theta)], axis=1)
    return carrier, raw, design

def build_production_encoder() -> HoldContrastFiLMEarlyPoolEncoder:
    """The same `canonical_p0 + selected EMPTY head` construction as joint M2."""
    encoder = HoldContrastFiLMEarlyPoolEncoder(100, 50, 64, side_dim=8, film_rank=8, num_post_layers=3, film_input='t4_plus_contrast')
    meta = champion.overlay_canonical_p0_and_empty_head(encoder)
    if meta['head_state_sha256'] != plan.SELECTED_HEAD_STATE_SHA256: raise RuntimeError('selected EMPTY head drift')
    encoder.eval()
    return encoder

def calibration_cost(activity: np.ndarray, bundle: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray, session: str, carrier: np.ndarray) -> dict[str, Any]:
    """Three independent CPU cold constructions; no decoder is instantiated."""
    trials = torch.from_numpy(np.ascontiguousarray(activity, dtype=np.float32))
    side = champion.empty_contrast_side(carrier)
    runs = []
    for repeat in range(3):
        t0 = time.perf_counter(); enc = build_production_encoder(); loaded = time.perf_counter()
        e0, u = champion.native_e0_and_u(enc, trials, side); done = time.perf_counter()
        runs.append({'repeat': repeat + 1, 'cold_load_seconds': loaded - t0, 'identity_construct_seconds': done - loaded, 'e0_sha256': asha(e0.numpy()), 'u_sha256': asha(u.numpy())})
        del enc, e0, u
    if len({r['e0_sha256'] for r in runs}) != 1 or len({r['u_sha256'] for r in runs}) != 1: raise RuntimeError('cold construction nondeterminism')
    carrier_runs = []
    ids = np.arange(33, dtype=np.int64)
    for repeat in range(3):
        t0 = time.perf_counter(); rebuilt, _raw, _design = selected_carrier(bundle, ids, mean=mean, std=std, session=session); elapsed = time.perf_counter() - t0
        if not np.array_equal(rebuilt, carrier): raise RuntimeError('M33 carrier changed during cost measurement')
        carrier_runs.append({'repeat': repeat + 1, 'm33_carrier_rebuild_seconds': elapsed, 'carrier_sha256': asha(rebuilt)})
    return {'production_materializer_scope':'native_e0_and_u produces both E0 and per-trial pre_pool u; this is not a final trained joint-runtime timing', 'encoder_materializer_runs': runs, 'carrier_rebuild_runs': carrier_runs, 'mean_cold_load_seconds': float(np.mean([r['cold_load_seconds'] for r in runs])), 'mean_native_e0_and_u_seconds': float(np.mean([r['identity_construct_seconds'] for r in runs])), 'mean_m33_carrier_rebuild_seconds': float(np.mean([r['m33_carrier_rebuild_seconds'] for r in carrier_runs])), 'mean_inmemory_total_calibration_wall_seconds': float(np.mean([r['identity_construct_seconds'] for r in runs]) + np.mean([r['m33_carrier_rebuild_seconds'] for r in carrier_runs]))}

def source_bundles() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    log = data.FileAccessLog(role='source'); dm = data.construct_source_datamodule(access=log); ds = dm.train_dataset
    bundles = {s: data._calib_bundle(ds, s) for s in plan.HELDIN_SESSIONS}
    return bundles, {'opened_nwbs': log.as_receipt(), 'surface': 'source_train held-in-calib support only'}
def ext4_bundles() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    log = data.FileAccessLog(role='ext4'); ds = data._build_ext4_dataset(log)
    bundles = {s: data._calib_bundle(ds, s) for s in plan.EXT4_SESSIONS}
    return bundles, {'opened_nwbs': log.as_receipt(), 'surface': 'ext4 held-out-calib support only'}

def run_session(surface: str, session: str, bundle: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray, *, costs: bool, dest: Path) -> dict[str, Any]:
    full_ids = np.arange(33, dtype=np.int64)
    started = time.perf_counter(); full, full_raw, full_design = selected_carrier(bundle, full_ids, mean=mean, std=std, session=session); full_seconds = time.perf_counter() - started
    cached_path = CACHE / surface / session / 'T.npy'; cached = np.load(cached_path)
    byte_equal = bool(np.array_equal(full, cached))
    if not byte_equal: raise RuntimeError(f'{surface}/{session}: direct M33 carrier differs from frozen cache')
    full_norm = float(np.linalg.norm(full)); rows = []; vectors: dict[str, np.ndarray] = {'full_m33': full, 'raw_full_m33': full_raw}
    for budget in BUDGETS:
        for seed in SEEDS:
            ids = full_ids if budget == 33 else np.sort(np.random.default_rng(seed).choice(33, size=budget, replace=False))
            selected_angles = np.asarray(bundle['angles'], np.float64)[ids]
            angles = selected_angles[np.isfinite(selected_angles)]
            design = np.stack([np.ones(len(angles)), np.cos(angles), np.sin(angles)], axis=1) if len(angles) else np.empty((0,3), np.float64)
            rank = int(np.linalg.matrix_rank(design)) if len(angles) else 0; condition = float(np.linalg.cond(design)) if rank == 3 else None
            base_row = {'budget_trials': budget, 'seed': seed, 'trial_ids': [int(x) for x in ids], 'finite_direction_trials': int(len(angles)), 'nonfinite_direction_trials': int(len(selected_angles)-len(angles)), 'direction_design_rank': rank, 'direction_design_condition': condition, 'direction_coverage_count': int(len(np.unique(angles)))}
            t0 = time.perf_counter()
            try:
                got, raw, _ = selected_carrier(bundle, ids, mean=mean, std=std, session=session)
            except ValueError as exc:
                # The production function refuses fewer than three distinct direction means.
                # Preserve this coverage failure as an outcome; do not invent a carrier.
                rows.append({**base_row, 'valid_production_carrier': False, 'carrier_seconds': time.perf_counter() - t0, 'failure': str(exc)})
                continue
            elapsed = time.perf_counter() - t0; norm = float(np.linalg.norm(got))
            key = f'carrier_m{budget:02d}_s{seed}'; raw_key = f'raw_m{budget:02d}_s{seed}'; vectors[key] = got; vectors[raw_key] = raw
            rows.append({**base_row, 'valid_production_carrier': True, 'carrier_seconds': elapsed, 'carrier_sha256': asha(got), 'raw_carrier_sha256': asha(raw), 'vector_key': key, 'raw_vector_key': raw_key, 'relative_frobenius_vs_m33': float(np.linalg.norm(got-full)/full_norm), 'cosine_vs_m33': float(got.ravel() @ full.ravel()/(norm*full_norm)), 'carrier_norm_frobenius': norm, 'per_unit_l2_mean': float(np.linalg.norm(got,axis=1).mean())})
    vector_dir = dest / 'vectors'; vector_dir.mkdir(parents=True, exist_ok=True); vector_path = vector_dir / f'{surface}__{session}.npz'; np.savez_compressed(vector_path, **vectors)
    result = {'surface': surface, 'session': session, 'vectors_npz': {'path': str(vector_path), 'sha256': fsha(vector_path), 'keys': sorted(vectors)}, 'm33_direct_cached_byte_equal': byte_equal, 'm33_cached_sha256': asha(cached), 'm33_direct_sha256': asha(full), 'm33_raw_sha256': asha(full_raw), 'm33_design_rank': int(np.linalg.matrix_rank(full_design)), 'm33_design_condition': float(np.linalg.cond(full_design)), 'm33_carrier_seconds': full_seconds, 'cache_activity_sha256': fsha(CACHE / surface / session / 'calib_activity.npy'), 'runs': rows}
    if costs: result['calibration_cost'] = calibration_cost(np.load(CACHE / surface / session / 'calib_activity.npy'), bundle, mean, std, session, full)
    return result

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--surface', choices=('source_train','ext4','both'), default='both'); ap.add_argument('--session', action='append'); ap.add_argument('--cost-session', default=None); ap.add_argument('--dest', type=Path, default=DEFAULT_OUT); ap.add_argument('--cpu-threads', type=int, default=2); args = ap.parse_args()
    if args.cpu_threads != 2: ap.error('contract requires --cpu-threads 2')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['OMP_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'; torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if args.dest.exists() and any(args.dest.iterdir()): ap.error(f'destination must be new and empty: {args.dest}')
    mean, std, norm = normalizer(); needed = ('source_train','ext4') if args.surface == 'both' else (args.surface,)
    loaded: dict[str, tuple[dict[str, dict[str,np.ndarray]],dict[str,Any]]] = {}
    for surface in needed: loaded[surface] = source_bundles() if surface == 'source_train' else ext4_bundles()
    allowed = set().union(*(set(x[0]) for x in loaded.values())); requested = set(args.session or allowed)
    if not requested <= allowed: ap.error(f'unknown/out-of-surface sessions: {sorted(requested-allowed)}')
    report: dict[str,Any] = {'schema':'m2_carrier_support_stability_v2','status':'COMPLETED','utc':datetime.now(timezone.utc).isoformat(),'device':'cpu','cpu_threads':2,'affinity':sorted(os.sched_getaffinity(0)),'no_decoder_or_target_bp':True,'production_binding':{'joint_train':str(ROOT/'scripts/rift_v1/m2_joint_train.py'),'joint_model':str(ROOT/'src/btransform_unified_v2/joint_m2_model.py'),'identity_provider':'HoldContrastFiLMEarlyPoolEncoder(100,50,64,side8,rank8,num_post3,t4_plus_contrast) + champion.overlay_canonical_p0_and_empty_head','film_states':{'path':str(FILM),'sha256':fsha(FILM)},'selected_head':{'path':str(HEAD),'sha256':fsha(HEAD)},'champion_wrapper':str(Path(champion.__file__).resolve()),'champion_wrapper_sha256':fsha(Path(champion.__file__).resolve()),'move_t4_function':str(WS/'streaming_calibration_exp/src/data/falcon_t4_features.py'),'move_t4_function_sha256':fsha(WS/'streaming_calibration_exp/src/data/falcon_t4_features.py'),'data_module_source':str(Path(data.__file__).resolve()),'data_module_source_sha256':fsha(Path(data.__file__).resolve()),'encoder_source':str(WS/'tfpd_exploration/src/m2_hold_film_probe_v1/encoder.py'),'encoder_source_sha256':fsha(WS/'tfpd_exploration/src/m2_hold_film_probe_v1/encoder.py'),'joint_m2_source_sha256':fsha(ROOT/'src/btransform_unified_v2/joint_m2_model.py'),'script_sha256':fsha(Path(__file__).resolve()),'normalizer_sha256':fsha(CACHE/'move_t4_normalizer.json'),'normalizer':norm},'protocol':{'M33':33,'budgets':list(BUDGETS),'seeds':list(SEEDS),'sampling':'without replacement; M33 uses canonical [0..32] on every seed','carrier':'direct `t4_from_trial_sums` then fixed source-seven M33 normalizer','metrics':['relative Frobenius versus M33','flattened cosine versus M33','Frobenius norm','per-unit L2 mean','direction design rank/condition/coverage'],'cost':'three independent CPU cold encoder loads and M33 identity constructions, separately timed'},'access':{},'sessions':{}}
    for surface,(bundles,access) in loaded.items():
        report['access'][surface]=access
        opened = sorted(set(access['opened_nwbs']['opened']))
        report['access'][surface]['opened_nwb_sha256'] = {path: fsha(Path(path)) for path in opened}
        for session,bundle in bundles.items():
            if session not in requested: continue
            report['sessions'][f'{surface}/{session}']=run_session(surface,session,bundle,mean,std,costs=(args.cost_session is None or args.cost_session==session),dest=args.dest)
            report['sessions'][f'{surface}/{session}']['raw_bundle_sha256'] = {name: asha(np.asarray(bundle[name])) for name in ('calib_neural','calib_trial_change','angles')}
    args.dest.mkdir(parents=True,exist_ok=True); out=args.dest/'report.json'; out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(out)
if __name__ == '__main__': main()
