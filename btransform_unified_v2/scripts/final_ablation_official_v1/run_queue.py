#!/usr/bin/env python3
"""ROOT's fresh-only serial GPU1 queue for the five fixed information controls.

Prepare seals local inputs; run waits for the existing M1 queue, then trains and
scores. This controller performs no Docker, registry, or EvalAI action.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
OUT = ROOT / 'results/final_ablation_official_v1'
QUEUE = OUT / 'root_formal_queue.json'
FREEZE = OUT / 'root_formal_inputs_v1.json'
BARRIER = ROOT / 'results/m1_muscle_r100_multiseed_v1/root_formal_queue.json'
PY = '/home/xinyuan/miniconda3/envs/spint/bin/python'
GPU = 'GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86'
SMOKES = {
    'm2_activity_only': 'm2_activity_only_smoke_cpu_v2',
    'm2_none': 'm2_none_smoke_cpu_v1',
    'h1_activity_only': 'h1_activity_only_smoke_cpu_v1',
    'h1_none': 'h1_none_smoke_cpu_v1',
    'm1_none': 'm1_none_smoke_cpu_v1',
}
BANKS = {
    'm2_activity_only': OUT / 'm2_activity_only_banks_v2',
    'm2_none': OUT / 'm2_none_banks_v1',
    'h1_activity_only': OUT / 'h1_activity_only_banks_v1',
    'h1_none': OUT / 'h1_none_banks_v1',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def job_list():
    jobs = []
    for name in SMOKES:
        run = OUT / ('formal_' + name + '_s42_v1')
        arm = 'ACTIVITY_ONLY' if name.endswith('activity_only') else 'NONE'
        if name.startswith('m2'):
            script = ROOT / 'scripts/m2_rift_r50_ablation_v1'
            argv = [PY, '-u', str(script / 'm2_ablation.py'), '--arm', arm,
                    '--bank-cache-root', str(BANKS[name]), '--dest', str(run),
                    '--device', 'cuda:0', '--cpu-threads', '4']
            score_dest = OUT / ('selection_' + name + '_ext6_v1')
            score = [PY, '-u', str(script / 'score_ext6.py'), '--arm', arm,
                     '--bank-cache-root', str(BANKS[name]), '--run-dir', str(run),
                     '--dest', str(score_dest), '--device', 'cuda:0', '--cpu-threads', '4']
            stages = [('train', argv, run), ('score', score, score_dest)]
        elif name.startswith('h1'):
            argv = [PY, '-u', str(ROOT / 'scripts/h1_signed_state_r300_ablation_v1/train.py'),
                    '--arm', arm, '--banks', str(BANKS[name]), '--dest', str(run),
                    '--device', 'cuda:0', '--stage', 'train']
            stages = [('train_and_score', argv, run)]
        else:
            argv = [PY, '-u', str(ROOT / 'scripts/m1_muscle_r100_none_ablation_v1/train.py'),
                    '--dest', str(run), '--baseline-run',
                    str(ROOT / 'results/m1_muscle_r100_v1/formal_s42_gpu1'),
                    '--device', 'cuda:0', '--cpu-threads', '4']
            stages = [(stage, argv + ['--stage', stage], run) for stage in ('train', 'score')]
        for stage, argv, dest in stages:
            jobs.append({'name': name, 'arm': arm, 'stage': stage, 'argv': argv,
                         'run_dir': str(run), 'dest': str(dest), 'status': 'PENDING',
                         'log': str(OUT / f'formal_{name}_{stage}_attempt01.log')})
    return jobs


def prepare():
    require(not QUEUE.exists() and not FREEZE.exists(), 'fresh queue/freeze files required')
    audit_path = OUT / 'root_five_arm_smoke_baseline_audit_v1.json'
    audit = read(audit_path)
    require(audit['status'] == 'PASSED_FIVE_CPU_SMOKES', 'five actual smoke audits required')
    h1_path = OUT / 'root_h1_runtime_bank_binding_audit_v1.json'
    h1 = read(h1_path)
    require(h1['status'] == 'PASSED_ALL27_RUNTIME_BANK_BINDINGS', 'H1 runtime bank audit required')
    files = {Path(__file__).resolve(), audit_path, h1_path}
    expected = dict(h1['source_file_sha256'])
    for name, smoke in SMOKES.items():
        meta_path = OUT / smoke / 'run_meta.json'
        require(sha(meta_path) == audit['rows'][name]['run_meta_sha256'], f'{name}: audit/meta drift')
        meta = read(meta_path)
        files.add(meta_path)
        files.add(Path(audit['rows'][name]['baseline_run']) / 'run_meta.json')
        for key in ('source_hashes', 'source_manifest_sha256', 'ablation_implementation_sha256'):
            for path, digest in meta.get(key, {}).items():
                if str(path).startswith('/'):
                    require(path not in expected or expected[path] == digest, f'conflicting binding {path}')
                    expected[path] = digest
    for folder in BANKS.values():
        files.update(p for p in folder.rglob('*') if p.is_file())
    # Freeze shared decoder/training dependencies, while excluding unrelated work.
    code_dirs = [WS / 'btransform_unified_v1/src', ROOT / 'src/btransform_unified_v2',
                 ROOT / 'scripts/rift_v1', ROOT / 'scripts/carrier_profile_v2',
                 ROOT / 'scripts/carrier_v4', ROOT / 'scripts/m1_muscle_r100_none_ablation_v1',
                 ROOT / 'scripts/m2_rift_r50_ablation_v1', ROOT / 'scripts/h1_signed_state_r300_ablation_v1']
    for folder in code_dirs:
        files.update(folder.rglob('*.py'))
    for name in ('h1_c2_cal1_b2_l200_p16.py', 'h1_profiles.py'):
        files.update((WS / 'btransform_unified_v1/scripts').rglob(name))
    # Only the same visible calibration records used by these fixed recipes.
    for dataset, subject in (('000941', 'MonkeyL'), ('000954', 'HumanPitt')):
        for split in ('held-in-calib', 'held-out-calib'):
            folder = WS / f'SPINT-main/data/{dataset}/sub-{subject}-{split}'
            paths = list(folder.glob('*.nwb'))
            require(paths, f'missing public calibration directory {folder}')
            files.update(paths)
    files.update(Path(p) for p in expected)
    pins = {}
    for path in sorted(files):
        require(path.is_file(), f'missing frozen input {path}')
        digest = sha(path)
        if str(path) in expected:
            require(digest == expected[str(path)], f'smoke/source binding drift {path}')
        pins[str(path)] = digest
    jobs = job_list()
    require(all(not Path(j['dest']).exists() for j in jobs), 'formal destinations must be fresh')
    require(all(not Path(j['log']).exists() for j in jobs), 'stage logs must be fresh')
    write(FREEZE, {'schema': 'root_final_ablation_input_freeze_v1', 'utc': now(),
                   'file_sha256': pins, 'file_count': len(pins), 'official_test_used': False})
    write(QUEUE, {'schema': 'root_final_ablation_formal_queue_v1', 'status': 'PREPARED',
                  'created_utc': now(), 'physical_gpu': 1, 'gpu_uuid': GPU,
                  'freeze_path': str(FREEZE), 'freeze_sha256': sha(FREEZE),
                  'barrier_queue': str(BARRIER), 'jobs': jobs,
                  'official_submission_execution': 'NOT_PART_OF_THIS_CONTROLLER',
                  'external_actions': False})
    print(json.dumps({'status': 'PREPARED', 'jobs': len(jobs), 'input_files': len(pins)}), flush=True)


def validate_inputs(queue):
    require(sha(FREEZE) == queue['freeze_sha256'], 'input freeze receipt changed')
    for path, digest in read(FREEZE)['file_sha256'].items():
        require(sha(path) == digest, f'frozen input changed: {path}')


def gpu_processes():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    require(f'1, {GPU}' in gpu.splitlines(), 'physical GPU1 UUID changed')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    return [line for line in apps.splitlines() if line.startswith(GPU + ',')]


def completion(job):
    run = Path(job['run_dir'])
    receipt = read(run / 'train_receipt.json')
    require(receipt.get('status') == 'COMPLETED', 'train completion receipt absent')
    if job['name'].startswith('m2'):
        require(receipt.get('epochs') == 24 and receipt.get('global_step') == 75960, 'M2 full train incomplete')
        if job['stage'] == 'score':
            score = read(Path(job['dest']) / 'score_receipt.json')
            require(score.get('status') == 'COMPLETED' and set(score['ema_by_epoch']) == {str(x) for x in range(1, 25)}, 'M2 all24 selection incomplete')
    elif job['name'].startswith('h1'):
        require(receipt.get('epochs') == 32 and receipt.get('updates') == 23392, 'H1 full train incomplete')
        require({r['epoch'] for r in read(run / 'ho_m3_selection.json')['curve']} == set(range(1, 33)), 'H1 all32 selection incomplete')
    else:
        require(receipt.get('epochs') == 24 and receipt.get('steps') == 159960, 'M1 full train incomplete')
        if job['stage'] == 'score':
            score = read(run / 'score_receipt.json')
            require(score.get('status') == 'COMPLETED' and set(score['ema_by_epoch']) == {str(x) for x in range(1, 25)}, 'M1 all24 selection incomplete')


def run():
    with (OUT / 'root_formal_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q = read(QUEUE)
        require(q['status'] == 'PREPARED', 'queue is fresh-only; inspect prior execution before any recovery')
        q.update(status='WAITING_FOR_M1_QUEUE', pid=os.getpid(), started_utc=now())
        write(QUEUE, q)
        try:
            expected = [(n, s) for n in ('original_s43', 'mean_rate4_s42', 'original_s44') for s in ('train', 'score')]
            while True:
                require(not (OUT / 'STOP_BEFORE_NEXT_STAGE').exists(), 'requested stop before next stage')
                b = read(BARRIER)
                require([(j['name'], j['stage']) for j in b['jobs']] == expected, 'M1 prerequisite stage inventory drift')
                require(b['status'] not in ('FAILED', 'STOPPED', 'CANCELLED'), 'M1 prerequisite queue failed/stopped')
                if b['status'] == 'COMPLETED' and all(j['status'] == 'COMPLETED' and j.get('returncode') == 0 for j in b['jobs']):
                    break
                q['heartbeat_utc'] = now()
                write(QUEUE, q)
                time.sleep(30)
            q['prerequisite_completed_sha256'] = sha(BARRIER)
            env = os.environ.copy()
            env.update(PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
            for job in q['jobs']:
                require(not (OUT / 'STOP_BEFORE_NEXT_STAGE').exists(), 'requested stop before next stage')
                q.update(status='VERIFYING_FROZEN_INPUTS', heartbeat_utc=now())
                write(QUEUE, q)
                validate_inputs(q)
                while gpu_processes():
                    require(not (OUT / 'STOP_BEFORE_NEXT_STAGE').exists(), 'requested stop while GPU occupied')
                    q.update(status='WAITING_FOR_GPU1', heartbeat_utc=now())
                    write(QUEUE, q)
                    time.sleep(30)
                if job['stage'] != 'score' or job['name'].startswith('m2'):
                    require(not Path(job['dest']).exists(), 'stage destination no longer fresh')
                with Path(job['log']).open('x') as log:
                    child = subprocess.Popen(job['argv'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    job.update(status='RUNNING', child_pid=child.pid, started_utc=now())
                    q.update(status='RUNNING', heartbeat_utc=now())
                    write(QUEUE, q)
                    print(json.dumps({'event': 'STAGE_STARTED', 'name': job['name'], 'stage': job['stage'], 'pid': child.pid}), flush=True)
                    while child.poll() is None:
                        q['heartbeat_utc'] = now()
                        write(QUEUE, q)
                        time.sleep(15)
                job.update(returncode=child.returncode, finished_utc=now())
                require(child.returncode == 0, f"stage failed: {job['name']} {job['stage']}; see {job['log']}")
                completion(job)
                job['status'] = 'COMPLETED'
                write(QUEUE, q)
                print(json.dumps({'event': 'STAGE_COMPLETED', 'name': job['name'], 'stage': job['stage']}), flush=True)
            q.update(status='COMPLETED', finished_utc=now())
            write(QUEUE, q)
        except BaseException as error:
            q.update(status='FAILED', failure=str(error), failed_utc=now())
            for job in q['jobs']:
                if job['status'] == 'RUNNING':
                    job['status'] = 'FAILED'
            write(QUEUE, q)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run'), required=True)
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else run()
