from __future__ import annotations
import base64, hashlib, importlib.util, json, sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import t4_m30_experiment_a_r5_authorization as auth
spec=importlib.util.spec_from_file_location('agg5',ROOT/'scripts/aggregate_t4_m30_experiment_a_r5.py');agg=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(agg)
S=agg.SESSIONS
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(tmp, mutate=None):
 tmp.mkdir(exist_ok=True); key=Ed25519PrivateKey.generate();pub=tmp/'pub';pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo));src=tmp/'x';src.write_text('x')
 pre=tmp/'pre';authj=tmp/'auth';sig=tmp/'sig';claim=tmp/'claim';result=tmp/'results';ref=tmp/'refs';status=tmp/'status';result.mkdir();ref.mkdir();status.mkdir();trainpath=tmp/'sua_exploration/scripts/train_variant_dandi688.py';trainpath.parent.mkdir(parents=True);trainpath.write_text('train');train=h(trainpath)
 refs={'t4':{},'ts4':{}}
 for arm in agg.ARMS+agg.REFS:
  for seed in agg.SEEDS:
   run=tmp/f'run_{arm}_{seed}';run.mkdir();meta=run/'run_metadata.json';normal='a7ee643d066bd3db9ce6e2b178527bb6655d90c485fb5a0151e5babb8fe255ad' if arm=='ph4' else '293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0';contract={'raw_refit_from_label_permuted_directions_before_normalization':True,'aligned_intercept_b_copied_from_ordinary_t4':True,'label_permutation_seed':seed} if arm=='ls4' else {};m={'status':'completed','variant':'B3S','seed':seed,'task':'CO','split_counts':[27,6,6],'max_units_exclusive':100,'signal_view':'sua','teacher_checkpoint':'/home/xinyuan/Work_host/SPINT/sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt','train_val_manifest':'/home/xinyuan/Work_host/SPINT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json','side_features':{'group':arm,'side_dim':4,'pool_size':30,'feature_version':1,'normalization_sha256':normal,'descriptor_contract':contract},'held_out_test_evaluated':False,'teacher_sha256':'9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d','train_val_manifest_sha256':'4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9','training':{'calibration_n_trials':30,'max_epochs':12,'no_early_stopping':True,'checkpoint_every_epoch':True,'loss_mode':'task_only','learning_rate':1e-4,'batch_size':32,'freeze_decoder':False,'identity_mode':'calibrated','deterministic':True},'t4_logit_residual':{'enabled':False}};meta.write_text(json.dumps(m));per={str(e):{'per_session_r2':{s:.1 for s in S}} for e in agg.EPOCHS};d={'variant':'B3S','seed':seed,'signal_view':'sua','epoch_list':list(agg.EPOCHS),'protocol':{'calibration_n':30,'pool_size':30,'total_epochs':12,'burn_in_epochs':4},'no_test_files_evaluated':True,'uses_backward_gradients':False,'run_metadata_path':str(meta),'run_metadata_sha256':h(meta),'per_epoch':per};path=(result if arm in agg.ARMS else ref)/f'{arm}_s{seed}.json';path.write_text(json.dumps(d));
   if arm in agg.REFS:refs[arm][str(seed)]=h(path)
   else:
    cost=run/'post_run_cost_receipt.json';cost.write_text(json.dumps({'run_metadata_sha256':h(meta),'train_variant_source_sha256':train,'accelerator':'gpu','fit_wall_clock_seconds':1,'cuda_peak_memory_allocated_bytes':1,'cuda_peak_memory_reserved_bytes':1}));status.joinpath(f'{arm}_s{seed}.json').write_text(json.dumps({'arm':arm,'seed':seed,'status':'completed','exit_code':0,'result_sha256':h(path),'metadata_sha256':h(meta),'cost_sha256':h(cost)}))
 pre.write_text(json.dumps({'source_sha256':{'x':h(src),'sua_exploration/scripts/train_variant_dandi688.py':train},'qualified_reference_artifacts':refs}));now=datetime.now(timezone.utc);a={'authorization':{'gpu_launch_authorized':True,'authorization_id':'id','single_use_nonce':'n','issued_at':(now-timedelta(seconds=1)).isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat(),'prelaunch_receipt_sha256':h(pre),'prelaunch_source_sha256':json.loads(pre.read_text())['source_sha256'],'public_key_path':str(pub),'public_key_sha256':h(pub)}};authj.write_text(json.dumps(a));sig.write_bytes(base64.b64encode(key.sign(authj.read_bytes())));claim.write_text(json.dumps({'authorization_id':'id','single_use_nonce':'n','authorization_sha256':h(authj)}))
 auth.base.ROOT=tmp;auth.base.PUBLIC_KEY=pub;auth.base.RECEIPT=pre;auth.base.AUTH=authj;auth.base.SIG=sig;auth.base.CLAIM=claim;agg.RECEIPT=pre;agg.require_claim=auth.require_claim
 if mutate:mutate(result,ref,status,tmp,authj,sig,claim)
 return result,ref,status,tmp/'out.json',key,authj,sig,claim,tmp
def run(x):return agg.main(['--result-dir',str(x[0]),'--reference-dir',str(x[1]),'--status-dir',str(x[2]),'--out',str(x[3])])
def test_full_24_artifact_on_disk_success(tmp_path):
 x=fixture(tmp_path);run(x);d=json.loads(x[3].read_text());assert len(d['artifact_evidence'])==24 and len(d['per_cell_operational_costs'])==18
 assert set(d['primary_oriented_contrasts'])=={'t4_minus_z4','t4_minus_ls4','t4_minus_mb4','t4_minus_ts4'}
 assert set(d['component_sufficiency'])=={'ph4','ac4','mb4','b4'} and isinstance(d['conditional_row_shuffle_trigger_arms'],list)
 for v in d['primary_oriented_contrasts'].values():assert 'paired_vs_unpaired'in v and 'per_seed_dispersion'in v and 'unpaired_seed_mean_se'in v['paired_vs_unpaired']
 for v in d['component_sufficiency'].values():assert {'component_minus_t4_lower_paired_2se','component_minus_t4_lower_hierarchical_95','component_vs_z4_state'}<=set(v)
 for v in d['contrasts'].values():assert 'hierarchical_bootstrap'in v and 'exact_paired_wilcoxon_sessions'in v and 'decision'in v and 'paired_vs_unpaired'in v and 'per_seed_dispersion'in v
def resign(x, mutate):
 a=json.loads(x[5].read_text());mutate(a['authorization']);x[5].write_text(json.dumps(a));x[6].write_bytes(base64.b64encode(x[4].sign(x[5].read_bytes())))
def metadata_mutation(x,key,value):
 arm='ls4' if key=='ls_seed' else 'z4';meta=x[8]/f'run_{arm}_42/run_metadata.json';m=json.loads(meta.read_text());
 if key=='pool':m['side_features']['pool_size']=value
 elif key=='normalizer':m['side_features']['normalization_sha256']=value
 elif key=='version':m['side_features']['feature_version']=value
 elif key=='ls_seed':m['side_features']['descriptor_contract']['label_permutation_seed']=value
 elif key=='residual':m['t4_logit_residual']['enabled']=value
 elif key in {'learning_rate','batch_size','freeze_decoder','identity_mode','deterministic'}:m['training'][key]=value
 else:m[key]=value
 meta.write_text(json.dumps(m));art=x[0]/f'{arm}_s42.json';d=json.loads(art.read_text());d['run_metadata_sha256']=h(meta);art.write_text(json.dumps(d));st=x[2]/f'{arm}_s42.json';q=json.loads(st.read_text());q['result_sha256']=h(art);q['metadata_sha256']=h(meta);q['cost_sha256']=h(meta.parent/'post_run_cost_receipt.json');st.write_text(json.dumps(q))
def test_disk_fail_closed_full_matrix(tmp_path):
 now=datetime.now(timezone.utc)
 cases=[
  ('bad detached signature',lambda x:x[6].write_bytes(b'bad'),'bad detached signature'),('nonce replay',lambda x:x[7].unlink(),'scheduler nonce claim missing'),('source drift',lambda x:(x[8]/'x').write_text('drift'),'source/dependency drift'),
  ('future auth',lambda x:resign(x,lambda a:a.update(issued_at=(now+timedelta(hours=1)).isoformat())),'invalid authorization window'),('expired auth',lambda x:resign(x,lambda a:a.update(expires_at=(now-timedelta(hours=1)).isoformat())),'invalid authorization window'),('long auth',lambda x:resign(x,lambda a:a.update(expires_at=(now+timedelta(hours=31)).isoformat())),'invalid authorization window'),
  ('teacher drift',lambda x:metadata_mutation(x,'teacher_sha256','bad'),'metadata drift'),('manifest drift',lambda x:metadata_mutation(x,'train_val_manifest_sha256','bad'),'metadata drift'),('variant drift',lambda x:metadata_mutation(x,'variant','B0'),'metadata drift'),('seed drift',lambda x:metadata_mutation(x,'seed',99),'metadata drift'),('pool drift',lambda x:metadata_mutation(x,'pool',99),'metadata drift'),('residual drift',lambda x:metadata_mutation(x,'residual',True),'training/residual drift'),
  ('teacher path drift',lambda x:metadata_mutation(x,'teacher_checkpoint','bad'),'metadata path drift'),('manifest path drift',lambda x:metadata_mutation(x,'train_val_manifest','bad'),'metadata path drift'),('task drift',lambda x:metadata_mutation(x,'task','bad'),'metadata drift'),('splits drift',lambda x:metadata_mutation(x,'split_counts',[1,2,3]),'metadata drift'),('max units drift',lambda x:metadata_mutation(x,'max_units_exclusive',99),'metadata drift'),('signal drift',lambda x:metadata_mutation(x,'signal_view','bad'),'metadata drift'),('lr drift',lambda x:metadata_mutation(x,'learning_rate',.2),'training/residual drift'),('batch drift',lambda x:metadata_mutation(x,'batch_size',99),'training/residual drift'),('freeze drift',lambda x:metadata_mutation(x,'freeze_decoder',True),'training/residual drift'),('identity drift',lambda x:metadata_mutation(x,'identity_mode','bad'),'training/residual drift'),('determinism drift',lambda x:metadata_mutation(x,'deterministic',False),'training/residual drift'),
  ('normalizer drift',lambda x:metadata_mutation(x,'normalizer','bad'),'normalizer drift'),('feature version drift',lambda x:metadata_mutation(x,'version',99),'metadata drift'),('LS4 permutation drift',lambda x:metadata_mutation(x,'ls_seed',99),'LS4 permutation contract'),
  ('missing status',lambda x:(x[2]/'z4_s42.json').unlink(),'missing status'),('failed status',lambda x:(x[2]/'z4_s42.json').write_text(json.dumps({'arm':'z4','seed':42,'status':'failed','exit_code':1})),'bad status'),('wrong status cell',lambda x:(x[2]/'z4_s42.json').write_text(json.dumps({'arm':'ac4','seed':42,'status':'completed','exit_code':0})),'bad status'),
  ('missing cost',lambda x:(x[8]/'run_z4_42/post_run_cost_receipt.json').unlink(),'missing cost'),('NaN score',lambda x: _nan(x),'NaN result'),('output collision',lambda x:x[3].write_text('x'),'aggregate output collision')]
 for i,(name,mut,expected) in enumerate(cases):
  x=fixture(tmp_path/f'c{i}');mut(x)
  try:run(x);assert False,name
  except Exception as e:assert expected in str(e),f'{name}: {e}'
  assert not x[3].exists() or name=='output collision'
def _nan(x):
 p=x[0]/'z4_s42.json';d=json.loads(p.read_text());d['per_epoch']['5']['per_session_r2'][S[0]]=float('nan');p.write_text(json.dumps(d));q=json.loads((x[2]/'z4_s42.json').read_text());q['result_sha256']=h(p);(x[2]/'z4_s42.json').write_text(json.dumps(q))
