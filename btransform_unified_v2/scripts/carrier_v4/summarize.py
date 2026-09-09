#!/usr/bin/env python3
"""Manifest-only carrier-v4 result summarizer; it never discovers runs or reads NPZs.

Each cell declares id, task, information_arm, fusion, proj_dim, path, format,
carrier_family, and comparison_family. References are explicit:
information_reference, fusion_reference, or external_baseline_reference.
Formats: m1_v4_score, m1_baseline_replay, h1_ho_selection.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
from typing import Any, Mapping
import numpy as np

MARGIN = .01
RNG = 42
BOOT = 20_000
M1_SESSIONS = {'20121004', '20121017', '20121024'}
H1_SESSIONS = {'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'}
ARM_ALIASES = {'FULL': 'full', 'full': 'full', 'D_JOINT': 'full',
               'B_ACTIVITY_ONLY': 'activity_only', 'ACTIVITY_ONLY': 'activity_only',
               'activity_only': 'activity_only', 'CARRIER_ONLY': 'carrier_only',
               'carrier_only': 'carrier_only', 'NONE': 'none', 'none': 'none'}

def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)

def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    need(isinstance(value, dict), f'object JSON required: {path}')
    return value

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()

def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)

def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)

def canonical_arm(value: Any) -> str:
    need(isinstance(value, str) and value in ARM_ALIASES, f'unknown information arm: {value!r}')
    return ARM_ALIASES[value]

def canonical_projection(fusion: Any, value: Any) -> int | None:
    """Normalize the CLI's concat placeholder while rejecting semantic drift."""
    need(isinstance(fusion, str), 'fusion must be a string')
    if fusion == 'concat':
        need(value in (None, 16), 'concat proj_dim must be null or placeholder 16')
        return None
    if fusion == 'proj_add':
        need(value in (16, 32), 'proj_add proj_dim must be 16 or 32')
        return value
    raise RuntimeError(f'unknown fusion: {fusion!r}')

def expected_epochs(task: str) -> int:
    return 24 if task == 'm1' else 32

def assert_unique_epochs(curve: list[dict[str, Any]], task: str) -> None:
    epochs = [row['epoch'] for row in curve]
    need(len(epochs) == len(set(epochs)), f'{task.upper()} duplicate epochs')

def m1_score(receipt: Mapping[str, Any], fmt: str) -> tuple[list[dict[str, Any]], bool, int]:
    if fmt == 'm1_v4_score':
        need(receipt.get('schema') == 'm1_carrier_v4_ho_calib_epoch_scan_v1', 'M1 v4 schema')
        rows = receipt.get('ema_by_epoch', {})
        complete = receipt.get('status') == 'COMPLETED'
        def row(epoch: str, value: Mapping[str, Any]) -> dict[str, Any]:
            per = value['per_session']
            return {'epoch': int(epoch), 'mean': float(value['equal_session_mean_channel_variance_weighted_r2']),
                    'per_session': {session: float(item['channel_variance_weighted_r2']) for session, item in per.items()}}
    else:
        schema = receipt.get('schema')
        need(schema in ('carrier_v4_m1_activity_only_baseline_replay_v1', 'm1_muscle_r100_baseline_replay_v1'), 'M1 replay schema')
        rows = receipt.get('completed', {})
        complete = receipt.get('status') == 'COMPLETED'
        metric = 'channel_variance_weighted_r2' if schema == 'carrier_v4_m1_activity_only_baseline_replay_v1' else 'sklearn_channel_centered_variance_weighted_r2'
        mean_metric = 'equal_session_mean_channel_variance_weighted_r2' if schema == 'carrier_v4_m1_activity_only_baseline_replay_v1' else 'equal_session_mean_sklearn_channel_centered'
        def row(epoch: str, value: Mapping[str, Any]) -> dict[str, Any]:
            metrics = value['metrics']; per = metrics['per_session']
            return {'epoch': int(epoch), 'mean': float(metrics[mean_metric]),
                    'per_session': {session: float(item[metric]) for session, item in per.items()}}
    need(isinstance(rows, Mapping), 'M1 epoch rows')
    curve = [row(epoch, value) for epoch, value in rows.items()]
    assert_unique_epochs(curve, 'm1')
    need(all(set(item['per_session']) == M1_SESSIONS and np.isfinite(item['mean']) and
             all(np.isfinite(value) for value in item['per_session'].values()) and
             abs(item['mean'] - np.mean(list(item['per_session'].values()))) <= 1e-12 for item in curve),
         'M1 per-group metric/mean drift')
    need(all(1 <= item['epoch'] <= 24 for item in curve), 'M1 epoch range')
    return curve, complete, 24

def h1_score(selection: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool, int]:
    # A non-final selection receipt is reportable only as a pending partial cell.
    if selection.get('status') != 'HO_M3_DEVELOPMENT_SELECTION':
        return [], False, 32
    need(isinstance(selection.get('curve'), list), 'H1 curve missing')
    curve: list[dict[str, Any]] = []
    for value in selection['curve']:
        per = value.get('per_session_r2', {})
        need(set(per) == H1_SESSIONS, 'H1 requires seven S6..S12 groups')
        values = {session: float(score) for session, score in per.items()}
        mean = float(value['val_ho_m3_grouped/r2_mean'])
        need(np.isfinite(mean) and all(np.isfinite(score) for score in values.values()) and
             abs(mean - np.mean(list(values.values()))) <= 1e-12, 'H1 nonfinite or mean/group drift')
        curve.append({'epoch': int(value['epoch']), 'mean': mean, 'per_session': values})
    assert_unique_epochs(curve, 'h1')
    need(all(1 <= item['epoch'] <= 32 for item in curve), 'H1 epoch range')
    return curve, True, 32

def pending_cell(spec: Mapping[str, Any], path: Path, reason: str) -> dict[str, Any]:
    task = spec['task']
    return {**dict(spec), 'information_arm': canonical_arm(spec['information_arm']), 'path': str(path),
            'proj_dim': canonical_projection(spec['fusion'], spec['proj_dim']),
            'recorded_projection_optional': spec['proj_dim'],
            'path_sha256': None, 'curve': [], 'complete': False, 'expected_epochs': expected_epochs(task),
            'pending_reason': reason}

def validate_m1_identity(receipt: Mapping[str, Any], spec: Mapping[str, Any], allow_partial: bool) -> str | None:
    """Return a pending reason only when an otherwise valid replay lacks provenance."""
    arm = canonical_arm(spec['information_arm'])
    if spec['format'] == 'm1_v4_score':
        need(canonical_arm(receipt.get('information_arm')) == arm, 'M1 receipt information arm drift')
        need(receipt.get('fusion') == spec['fusion'] and canonical_projection(receipt.get('fusion'), receipt.get('proj_dim')) == spec['proj_dim'], 'M1 receipt fusion/proj_dim drift')
        need(receipt.get('seed') == 42, 'M1 receipt seed must be 42')
        spec['receipt_recorded_projection_optional'] = receipt.get('proj_dim')
        return None
    schema = receipt.get('schema')
    expected_arm = 'B_ACTIVITY_ONLY' if schema == 'carrier_v4_m1_activity_only_baseline_replay_v1' else 'D_JOINT'
    need(receipt.get('arm') == expected_arm and canonical_arm(receipt.get('arm')) == arm, 'M1 replay arm drift')
    # Legacy B/D replay receipts identify the formal run. Its architecture is fixed concat/no projection.
    need(spec['fusion'] == 'concat' and spec['proj_dim'] is None, 'legacy M1 replay requires concat with proj_dim null')
    run = receipt.get('baseline_run')
    if not isinstance(run, str) or not Path(run).resolve().joinpath('run_meta.json').is_file():
        if allow_partial:
            return 'baseline run metadata unavailable'
        raise RuntimeError('baseline run metadata unavailable')
    meta = read(Path(run).resolve() / 'run_meta.json')
    need(canonical_arm(meta.get('arm')) == arm and meta.get('seed') == 42, 'M1 replay run metadata arm/seed drift')
    need(meta.get('identity_interface') == 'live_b3s_concat' and canonical_projection('concat', meta.get('proj_dim')) == spec['proj_dim'], 'M1 replay run metadata fusion/proj_dim drift')
    spec['receipt_recorded_projection_optional'] = meta.get('proj_dim')
    return None

def validate_h1_identity(meta: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    schema = meta.get('schema')
    common = meta.get('status') == 'FORMAL' and meta.get('epochs') == 32 and meta.get('seed') == 42 and meta.get('official_test_used') is False
    if schema == 'rift_h1_signed_state_r300_v1':
        need(common, 'old H1 signed run provenance mismatch')
        need(canonical_arm(spec['information_arm']) == 'full' and spec['fusion'] == 'proj_add' and spec['proj_dim'] == 16,
             'old H1 signed state is locked to full/proj_add/P16')
        spec['run_meta_recorded_projection_optional'] = meta.get('proj_dim')
        return
    if schema == 'rift_h1_carrier_v4_r300_v1':
        need(common, 'new H1 v4 run provenance mismatch')
        need(canonical_arm(meta.get('information_arm')) == canonical_arm(spec['information_arm']), 'new H1 v4 information arm drift')
        need(meta.get('fusion') == spec['fusion'] and canonical_projection(meta.get('fusion'), meta.get('proj_dim')) == spec['proj_dim'], 'new H1 v4 fusion/proj_dim drift')
        spec['run_meta_recorded_projection_optional'] = meta.get('proj_dim')
        return
    raise RuntimeError('unknown H1 run_meta schema')

def parse_cell(spec: Mapping[str, Any], allow_partial: bool) -> dict[str, Any]:
    required = ('id', 'task', 'information_arm', 'fusion', 'proj_dim', 'path', 'format', 'carrier_family', 'comparison_family')
    need(all(key in spec for key in required), f'cell missing {required}')
    task = spec['task']; need(task in ('m1', 'h1'), 'task')
    need(isinstance(spec['id'], str) and isinstance(spec['fusion'], str), 'cell id/fusion')
    need(isinstance(spec['carrier_family'], str) and isinstance(spec['comparison_family'], str), 'carrier/comparison family')
    canonical_arm(spec['information_arm'])
    canonical_projection(spec['fusion'], spec['proj_dim'])
    path = Path(spec['path']).resolve()
    if not path.is_file():
        if allow_partial:
            return pending_cell(spec, path, 'result receipt missing')
        raise RuntimeError(f'result receipt missing: {path}')
    normalized = {**dict(spec), 'information_arm': canonical_arm(spec['information_arm']),
                  'proj_dim': canonical_projection(spec['fusion'], spec['proj_dim']),
                  'recorded_projection_optional': spec['proj_dim']}
    receipt = read(path)
    if normalized['format'] in ('m1_v4_score', 'm1_baseline_replay'):
        need(task == 'm1', 'M1 format/task')
        pending = validate_m1_identity(receipt, normalized, allow_partial)
        if pending is not None:
            return pending_cell(spec, path, pending)
        curve, complete, total = m1_score(receipt, normalized['format'])
    elif normalized['format'] == 'h1_ho_selection':
        need(task == 'h1', 'H1 format/task')
        meta_path = path.parent / 'run_meta.json'
        if not meta_path.is_file():
            if allow_partial:
                return pending_cell(spec, path, 'H1 sibling run_meta.json missing')
            raise RuntimeError('H1 selection requires sibling run_meta.json')
        validate_h1_identity(read(meta_path), normalized)
        curve, complete, total = h1_score(receipt)
    else:
        raise RuntimeError('unknown format')
    complete = complete and {item['epoch'] for item in curve} == set(range(1, total + 1))
    result = {**normalized, 'path': str(path),
              'path_sha256': sha(path), 'curve': sorted(curve, key=lambda item: item['epoch']),
              'complete': complete, 'expected_epochs': total}
    if not complete:
        result['pending_reason'] = 'receipt not completed or required epochs are absent'
    return result

def bootstrap(delta: np.ndarray, task: str) -> dict[str, Any]:
    group_count = len(delta)
    need(group_count == (3 if task == 'm1' else 7), 'unexpected paired group count')
    rng = np.random.default_rng(RNG)
    means = (np.array([np.mean(delta[list(indices)]) for indices in np.ndindex(*(group_count,) * group_count)], float)
             if task == 'm1' else np.mean(delta[rng.integers(0, group_count, size=(BOOT, group_count))], axis=1))
    return {'n_groups': group_count, 'method': 'exact_enumeration_3^3' if task == 'm1' else 'paired_session_bootstrap',
            'rng_seed': RNG, 'resamples': len(means), 'mean_delta': float(delta.mean()),
            'one_sided_lower_95': float(np.quantile(means, .05)),
            'two_sided_95': [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            'public_single_seed_descriptive_evidence': bool(np.quantile(means, .05) >= -MARGIN)}

def best_epoch(cell: Mapping[str, Any]) -> int:
    return min(cell['curve'], key=lambda item: (-item['mean'], item['epoch']))['epoch']

def comparison_record(kind: str, candidate: Mapping[str, Any], reference: Mapping[str, Any], endpoint_label: str,
                      candidate_epoch: int, reference_epoch: int) -> dict[str, Any]:
    candidate_rows = {item['epoch']: item for item in candidate['curve']}
    reference_rows = {item['epoch']: item for item in reference['curve']}
    need(candidate_epoch in candidate_rows and reference_epoch in reference_rows, 'comparison endpoint epoch absent')
    left, right = candidate_rows[candidate_epoch], reference_rows[reference_epoch]
    need(set(left['per_session']) == set(right['per_session']), 'paired session roster drift')
    delta = np.array([left['per_session'][session] - right['per_session'][session] for session in sorted(left['per_session'])])
    return {'kind': kind, 'cell': candidate['id'], 'reference': reference['id'], 'endpoint_label': endpoint_label,
            'candidate_epoch': candidate_epoch, 'reference_epoch': reference_epoch,
            'bootstrap': bootstrap(delta, candidate['task'])}

def pending_comparison(kind: str, candidate: Mapping[str, Any], reference_id: Any, reason: str) -> dict[str, Any]:
    return {'kind': kind, 'cell': candidate['id'], 'reference': reference_id, 'status': 'pending', 'pending_reason': reason}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve(); need(not output.exists(), '--output-dir must be fresh')
    manifest = read(args.manifest.resolve())
    cells = [parse_cell(spec, args.allow_partial) for spec in manifest.get('cells', [])]
    need(cells, 'manifest cells required')
    need(len({cell['id'] for cell in cells}) == len(cells), 'duplicate cell id')
    if not args.allow_partial:
        need(all(cell['complete'] for cell in cells), 'incomplete cell; use --allow-partial for an explicitly in-progress report')
    by_id = {cell['id']: cell for cell in cells}; comparisons: list[dict[str, Any]] = []
    for candidate in cells:
        references = []
        if candidate['information_arm'] != 'full' and candidate.get('information_reference') is not None:
            references.append(('information_reduced_minus_FULL', candidate['information_reference'], False, 'information'))
        if candidate['information_arm'] == 'full' and candidate.get('fusion_reference') is not None:
            references.append(('fusion_FULL_variant_minus_referencefusion', candidate['fusion_reference'], False, 'fusion'))
        if candidate['information_arm'] == 'full' and candidate.get('external_baseline_reference') is not None:
            references.append(('external_FULL_variant_minus_external_baseline', candidate['external_baseline_reference'], True, 'external'))
        for kind, reference_id, external, relation in references:
            reference = by_id.get(reference_id)
            if reference is None:
                if args.allow_partial:
                    comparisons.append(pending_comparison(kind, candidate, reference_id, 'reference cell absent from manifest'))
                    continue
                raise RuntimeError(f'{kind} reference must name a manifest cell')
            if not candidate['complete'] or not reference['complete']:
                if args.allow_partial:
                    comparisons.append(pending_comparison(kind, candidate, reference_id, 'candidate or reference is incomplete'))
                    continue
                raise RuntimeError('reference task/completion drift')
            need(reference['task'] == candidate['task'], 'reference task drift')
            if relation == 'information':
                need(reference['information_arm'] == 'full' and reference['fusion'] == candidate['fusion'] and reference['proj_dim'] == candidate['proj_dim'], 'information_reference must be same fusion/proj FULL')
            elif relation == 'fusion':
                need(reference['information_arm'] == 'full' and (reference['fusion'], reference['proj_dim']) != (candidate['fusion'], candidate['proj_dim']), 'fusion_reference must be distinct FULL fusion/proj')
            else:
                need(reference['information_arm'] == 'full' and (reference['fusion'], reference['proj_dim']) == (candidate['fusion'], candidate['proj_dim']), 'external baseline must be FULL and share fusion/proj')
            if not external:
                need(reference['carrier_family'] == candidate['carrier_family'] and reference['comparison_family'] == candidate['comparison_family'], 'within-family comparison drift')
            else:
                need(reference['carrier_family'] != candidate['carrier_family'] and reference['comparison_family'] == candidate['comparison_family'], 'external family comparison drift')
            fixed = 3 if candidate['task'] == 'm1' else 16
            comparisons.append(comparison_record(kind, candidate, reference, f'fixed_e{fixed}', fixed, fixed))
            comparisons.append(comparison_record(kind, candidate, reference, 'selected_each', best_epoch(candidate), best_epoch(reference)))
    for cell in cells:
        fixed = 3 if cell['task'] == 'm1' else 16
        cell['fixed_epoch'] = fixed
        cell['fixed_epoch_row'] = next((row for row in cell['curve'] if row['epoch'] == fixed), None)
        cell['selected_earliest_mean_max'] = None if not cell['complete'] else min(cell['curve'], key=lambda item: (-item['mean'], item['epoch']))
    output.mkdir(parents=True)
    summary = {'schema': 'carrier_v4_manifest_summary_v1', 'manifest': str(args.manifest.resolve()),
               'manifest_sha256': sha(args.manifest.resolve()), 'noninferiority_margin': MARGIN,
               'caveat': 'Public calibration, single-seed descriptive comparisons only; these results do not prove a null effect or hidden/generalization performance.',
               'cells': cells, 'comparisons': comparisons}
    atomic_json(output / 'summary.json', summary)
    with (output / 'curves.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['cell_id', 'task', 'epoch', 'mean', 'session', 'session_r2'])
        for cell in cells:
            for row in cell['curve']:
                for session, score in sorted(row['per_session'].items()):
                    writer.writerow([cell['id'], cell['task'], row['epoch'], row['mean'], session, score])
    lines = ['# Carrier v4 结果汇总', '', f'预声明非劣界值：`{MARGIN:.2f}`。仅描述公开校准、单 seed 结果；不证明零效应，也不外推到 hidden 或泛化表现。', '', '## 单元']
    for cell in cells:
        if cell['fixed_epoch_row'] is None:
            lines.append(f"- `{cell['id']}`：进行中；{cell.get('pending_reason', '尚无固定 endpoint')}。")
        else:
            lines.append(f"- `{cell['id']}`：{'完整' if cell['complete'] else '进行中'}；固定 epoch {cell['fixed_epoch']}={cell['fixed_epoch_row']['mean']:.6f}")
    lines += ['', '## 配对差值']
    for item in comparisons:
        if item.get('status') == 'pending':
            lines.append(f"- `{item['kind']}` {item['cell']} − {item['reference']}：进行中；{item['pending_reason']}。")
        else:
            evidence = item['bootstrap']
            lines.append(f"- `{item['kind']}` {item['cell']} − {item['reference']}（{item['endpoint_label']}；candidate epoch {item['candidate_epoch']}，reference epoch {item['reference_epoch']}）：均值 {evidence['mean_delta']:.6f}；单侧95%下界 {evidence['one_sided_lower_95']:.6f}；双侧95%区间 {evidence['two_sided_95']}；公开单-seed 描述证据={evidence['public_single_seed_descriptive_evidence']}。")
    atomic_text(output / 'results.md', '\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()
