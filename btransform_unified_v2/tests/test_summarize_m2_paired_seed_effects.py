import importlib.util
import hashlib
import json
from pathlib import Path

P = Path(__file__).parents[1] / "scripts/diagnostics_v1/summarize_m2_paired_seed_effects.py"
spec = importlib.util.spec_from_file_location("paired", P); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def _sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()

def _write(root: Path, arm: str, seed: int):
    run=root/f"results/rift_v1/m2_r50_joint_{arm.lower()}_s{seed}_formal_v1"
    run.mkdir(parents=True, exist_ok=True)
    source=root/'fixture_source.py';source.parent.mkdir(parents=True,exist_ok=True);source.write_text('fixture')
    cache=root.parent/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/ext4'
    hashes={}
    for s in mod.SESSIONS:
        f=cache/s/'calib_activity.npy';f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(s.encode());hashes[s]={'calib_activity.npy':_sha(f)}
    meta={"schema":"m2_rift_joint_train_v2","status":"FORMAL","cell":"M2-RIFT-R50-D4-JOINT-FILM-M33-V1","arm":mod.ARMS[arm],"seed":seed,"sampler_seed":42,"epochs":24,"source_hashes":{str(source):_sha(source)},"cache_hashes":{"ext4":hashes}}
    train={"schema":"m2_rift_joint_train_receipt_v2","status":"COMPLETED","cell":meta['cell'],"arm":mod.ARMS[arm],"seed":seed,"sampler_seed":42,"epochs":24,"global_step":75960,"source_hashes":meta['source_hashes'],"cache_hashes":meta['cache_hashes']}
    rows={}
    for e in range(1,25):
        cp=run/f'epoch_{e:03d}.pt';cp.write_bytes(str(e).encode())
        rows[str(e)]={"checkpoint_sha256":_sha(cp),"partial":False,"n_windows":2069,"equal_session_mean":float(e),"per_session":{s:{"r2":float(e),"window_count":mod.WINDOWS[s]} for s in mod.SESSIONS}}
    score={"schema":"m2_rift_joint_ext4_epoch_scan_v1","status":"COMPLETED","cell":meta['cell'],"arm":mod.ARMS[arm],"seed":seed,"sampler_seed":42,"official_test_used":False,"source_hashes":meta['source_hashes'],"cache_hashes":meta['cache_hashes'],"ema_by_epoch":rows,"selection":{"epoch":24,"rule":"earliest best equal_session_mean","equal_session_mean":24.0}}
    for name,obj in (("run_meta.json",meta),("train_receipt.json",train),("score_receipt.json",score)):
        (run/name).write_text(json.dumps(obj))

def test_missing_pair_stays_pending(tmp_path):
    root=tmp_path/'btransform_unified_v2';_write(root,"B",42)
    result=mod.summarize(root)
    assert result["status"]=="PENDING"
    assert "three_seed_summary" not in result

def test_three_complete_seed_pairs_are_seed_level_only(tmp_path):
    root=tmp_path/'btransform_unified_v2'
    for seed in mod.SEEDS:
        for arm in mod.ARMS: _write(root,arm,seed)
    result=mod.summarize(root)
    assert result["status"]=="COMPLETED"
    assert result["three_seed_summary"]["independent_epoch_pick_delta_D_minus_B"]["n_independent_seed_pairs"]==3
    assert result["no_significance_test"] is True

def test_tampered_mean_or_checkpoint_hash_keeps_summary_pending(tmp_path):
    root=tmp_path/'btransform_unified_v2'
    for seed in mod.SEEDS:
        for arm in mod.ARMS: _write(root,arm,seed)
    score=root/'results/rift_v1/m2_r50_joint_d_s44_formal_v1/score_receipt.json';body=json.loads(score.read_text());body['ema_by_epoch']['19']['equal_session_mean']=999.;score.write_text(json.dumps(body))
    assert mod.summarize(root)['status']=='PENDING'

def test_nan_selected_mean_keeps_summary_pending(tmp_path):
    root=tmp_path/'btransform_unified_v2'
    for seed in mod.SEEDS:
        for arm in mod.ARMS: _write(root,arm,seed)
    score=root/'results/rift_v1/m2_r50_joint_b_s42_formal_v1/score_receipt.json';body=json.loads(score.read_text());body['selection']['equal_session_mean']=float('nan');score.write_text(json.dumps(body))
    assert mod.summarize(root)['status']=='PENDING'
    _write(root,'D',44); score=root/'results/rift_v1/m2_r50_joint_d_s44_formal_v1/score_receipt.json';body=json.loads(score.read_text());body['ema_by_epoch']['19']['checkpoint_sha256']='wrong';score.write_text(json.dumps(body))
    assert mod.summarize(root)['status']=='PENDING'
