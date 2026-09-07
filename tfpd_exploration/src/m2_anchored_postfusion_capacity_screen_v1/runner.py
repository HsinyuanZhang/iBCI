"""Coordinated one-loader GPU0 APFC source-only screen.

The frozen B3S branches are evaluated once per causal pool state.  Three tiny
gates then share one concatenated frozen-decoder forward and one backward pass.
No target records are constructed in this module.
"""
from __future__ import annotations
import copy, math, time
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
import torch
from . import plan
from .gates import CapacityGate, disagreement_statistic

class RunnerError(RuntimeError): pass
def _need(ok,msg):
    if not ok: raise RunnerError(msg)

def validate_gpu0_only() -> dict[str,object]:
    _need(torch.cuda.is_available(),'APFC GPU0 unavailable'); torch.cuda.init()
    _need(torch.cuda.current_device()==0 and torch.cuda.device_count()==1,'APFC logical GPU topology drift')
    raw=str(torch.cuda.get_device_properties(0).uuid); uuid=raw if raw.startswith('GPU-') else 'GPU-'+raw
    _need(uuid==plan.GPU_UUID,'APFC GPU0 UUID drift')
    return {'cuda_visible_devices':'0','logical_device':0,'physical_device':0,'uuid_raw':raw,'uuid':uuid,
            'cuda_initialized':True,'device_name':str(torch.cuda.get_device_name(0))}

@dataclass
class _Branch:
    native: torch.Tensor; post: torch.Tensor; statistic: torch.Tensor

class APFCCoordinatedRunner:
    def __init__(self,*,repo_root:Any,device:str='cuda:0',launch_attestation:Mapping[str,object]) -> None:
        self.repo_root=repo_root; self.device=torch.device(device); self.launch=dict(launch_attestation)
        self.prepared=None; self.materials={}; self.gates={}; self.optimizers={}; self.cache={}; self.dc_fit=(0.,1.)
        self.step=0
    def prepare(self)->dict[str,object]:
        from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import training, source_replay
        self.prepared=training.prepare_selected_t4_apfg_after_attempt(repo_root=self.repo_root,device='cuda:0',launch_attestation=self.launch)
        # APFG's installed scalar is only the immutable branch provider here.
        self.prepared.adapter.set_alpha_training(False)
        for p in self.prepared.module.parameters(): p.requires_grad_(False); p.grad=None
        ds=self.prepared.datamodule.train_dataset; raw=self.prepared.datamodule.train_calib_heldin_sessions
        names=tuple(sorted(ds.calib_trialized_neural_features)); _need(len(names)==plan.SOURCE_SESSIONS,'APFC source roster drift')
        self.materials={n:source_replay.materialize_pooled_g00m_source_session(dataset=ds,raw_sessions=raw,session=n) for n in names}
        fit=names[:plan.FIT_SESSIONS]
        # Fit DC2 feature normalization solely from frozen causal source states.
        stats=[]
        for epoch in range(1,plan.EPOCHS+1):
            ordinal=0
            for n in fit:
                for batch in source_replay.canonical_m30_source_batches(self.materials[n]):
                    state=source_replay.pool_for_controller_coordinate(material=self.materials[n],coordinate=batch[0],epoch_one_indexed=epoch,canonical_batch_ordinal=ordinal)
                    stats.append(float(self._branch(self.materials[n],batch[0],state).statistic.detach().cpu())); ordinal+=1
        mean=float(np.mean(stats)); std=float(np.std(stats)); _need(math.isfinite(mean) and std>0 and math.isfinite(std),'APFC DC2 source standardizer invalid')
        self.dc_fit=(mean,std)
        self.gates={arm:CapacityGate(arm,source_fit_mean=mean,source_fit_std=std).to(self.device) for arm in plan.ARMS}
        self.optimizers={arm:torch.optim.Adam([self.gates[arm].params],lr=plan.ADAM_LR,betas=plan.ADAM_BETAS,eps=plan.ADAM_EPS,weight_decay=0.,amsgrad=False) for arm in plan.ARMS}
        return {'pit_materializations':1,'source_sessions':list(names),'activity_authority':plan.ACTIVITY_AUTHORITY,
                'selected_checkpoint_sha256':plan.SELECTED_CHECKPOINT_SHA256,'selected_student_state_sha256':plan.SELECTED_STUDENT_STATE_SHA256,
                'dc2_source_fit_mean':mean,'dc2_source_fit_std':std,'arms':{a:self.gates[a].evidence() for a in plan.ARMS}}
    def _branch(self,material,coord,state)->_Branch:
        key=state.identity_digest()
        if key in self.cache:return self.cache[key]
        native_encoder=self.prepared.adapter.native
        activity=np.ascontiguousarray(material.activities[np.asarray(state.member_trial_ids,dtype=np.int64)],dtype=np.float32)
        cal=torch.from_numpy(activity).unsqueeze(0).to(self.device)
        side=torch.from_numpy(material.normalized_side).unsqueeze(0).to(self.device)
        with torch.no_grad():
            native=native_encoder.forward_batch(cal,side_features=side).detach()
            per=native_encoder.pre_pool(cal.permute(0,1,3,2)); side_many=side.unsqueeze(1).expand(-1,per.shape[1],-1,-1)
            values=native_encoder.post_pool(torch.cat((per,side_many),dim=-1))[0].detach()
            post=values[0]
            for i in range(1,values.shape[0]): post=post+values[i]
            post=(post/int(values.shape[0])).unsqueeze(0)
            stat=disagreement_statistic(values).detach()
        _need(native.shape==post.shape and native.shape[-1]==plan.WINDOW_BINS,'APFC branch shape drift')
        b=_Branch(native=native,post=post,statistic=stat); self.cache[key]=b; return b
    def _batch_tensors(self,material,batch):
        anchor=batch[0]; starts=np.asarray([x.window_start for x in batch],dtype=np.int64); ds=self.prepared.datamodule.train_dataset
        neural_all=np.asarray(ds.neural_data[anchor.session],dtype=np.float32); target_all=np.asarray(ds.covariate_data[anchor.session],dtype=np.float32)
        ix=starts[:,None]+np.arange(plan.WINDOW_BINS)[None,:]
        neural=torch.from_numpy(np.ascontiguousarray(neural_all[ix],dtype=np.float32)).to(self.device)
        target=torch.from_numpy(np.ascontiguousarray(target_all[starts+49],dtype=np.float32)).to(self.device)
        return neural,target
    def _step(self,material,batch,epoch,ordinal)->dict[str,float]:
        from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import source_replay
        state=source_replay.pool_for_controller_coordinate(material=material,coordinate=batch[0],epoch_one_indexed=epoch,canonical_batch_ordinal=ordinal)
        branch=self._branch(material,batch[0],state); neural,target=self._batch_tensors(material,batch); b=neural.shape[0]
        identities=[]
        for arm in plan.ARMS:
            z=(branch.statistic-self.dc_fit[0])/self.dc_fit[1] if arm=='A-DC2' else None
            ident=self.gates[arm].gate(z=z,native=branch.native,post=branch.post,training=True).expand(b,-1,-1)
            identities.append(ident)
            self.optimizers[arm].zero_grad(set_to_none=True)
        all_neural=torch.cat([neural]*3,dim=0); all_identity=torch.cat(identities,dim=0)
        pred=self.prepared.module.student.decode_with_identity(all_neural,all_identity)/5.0
        _need(torch.isfinite(pred).all(),'APFC decoder nonfinite')
        losses={}; total=0.
        for i,arm in enumerate(plan.ARMS):
            item=pred[i*b:(i+1)*b,-1,:]; loss=torch.mean((item-target)**2); losses[arm]=loss; total=total+loss
        total.backward()
        for arm in plan.ARMS:
            p=self.gates[arm].params; _need(p.grad is not None and torch.isfinite(p.grad).all(),'APFC gate gradient drift')
            self.optimizers[arm].step(); _need(torch.isfinite(p).all(),'APFC gate parameter nonfinite')
        self.step+=1
        return {arm:float(losses[arm].detach().cpu()) for arm in plan.ARMS}
    def _rollout(self,material,gate,law)->float:
        from tfpd_exploration.src.m2_anchored_postfusion_gate_v1.source_replay import OrderedRawActivityPool, _r2
        pool=OrderedRawActivityPool(support_trial_ids=material.support_indices,support_activities=[material.activities[i] for i in material.support_indices],capacity=30 if law=='FIXED30' else None)
        ds=self.prepared.datamodule.train_dataset; neural=np.asarray(ds.neural_data[material.session],dtype=np.float32); beh=np.asarray(ds.covariate_data[material.session],dtype=np.float32)
        pred=[]; target=[]
        with torch.inference_mode():
            for row in material.query_rows:
                starts=np.asarray(row['metric_starts'],dtype=np.int64)
                if starts.size:
                    # Validation replay materializes its current causal stack
                    # directly; its cardinality is unrestricted for UNCAPPED.
                    activity=np.ascontiguousarray(pool.stack(),dtype=np.float32); cal=torch.from_numpy(activity).unsqueeze(0).to(self.device); side=torch.from_numpy(material.normalized_side).unsqueeze(0).to(self.device)
                    native=self.prepared.adapter.native.forward_batch(cal,side_features=side); per=self.prepared.adapter.native.pre_pool(cal.permute(0,1,3,2)); vals=self.prepared.adapter.native.post_pool(torch.cat((per,side.unsqueeze(1).expand(-1,per.shape[1],-1,-1)),dim=-1))[0]
                    post=vals[0]
                    for i in range(1,vals.shape[0]):post=post+vals[i]
                    post=(post/int(vals.shape[0])).unsqueeze(0); z=(disagreement_statistic(vals)-gate.source_fit_mean)/gate.source_fit_std if gate.arm=='A-DC2' else None
                    ident=gate.gate(z=z,native=native,post=post,training=False)
                    ix=starts[:,None]+np.arange(50)[None,:]; windows=np.ascontiguousarray(neural[ix],dtype=np.float32); pieces=[]
                    for off in range(0,len(windows),1024): pieces.append((self.prepared.module.student.decode_with_identity(torch.from_numpy(windows[off:off+1024]).to(self.device),ident)[:,-1,:]/5.0).cpu().numpy())
                    pred.append(np.concatenate(pieces)); target.append(beh[starts+49])
                pool.commit_completed(trial_id=int(row['position']),activity=np.asarray(row['activity'],dtype=np.float32))
        return float(_r2(np.concatenate(target),np.concatenate(pred)))
    def run(self)->dict[str,object]:
        _need(self.prepared is not None,'APFC not prepared'); names=tuple(sorted(self.materials)); fit=names[:5]; val=names[5:]
        batches=[(n,b) for n in fit for b in __import__('tfpd_exploration.src.m2_anchored_postfusion_gate_v1.source_replay',fromlist=['canonical_m30_source_batches']).canonical_m30_source_batches(self.materials[n])]
        history={a:[] for a in plan.ARMS}; started=time.monotonic()
        for epoch in range(1,13):
            losses={a:[] for a in plan.ARMS}
            for ordinal,(n,b) in enumerate(batches):
                result=self._step(self.materials[n],b,epoch,ordinal)
                for a in plan.ARMS:losses[a].append(result[a])
            for arm in plan.ARMS:
                metrics={s:self._rollout(self.materials[s],self.gates[arm],'UNCAPPED') for s in val}
                history[arm].append({'epoch':epoch,'per_session_r2':metrics,'mean_r2':float(np.mean(list(metrics.values()))),'mean_loss':float(np.mean(losses[arm])),'params':[float(x) for x in self.gates[arm].params.detach().cpu()]})
        selected={a:max(history[a],key=lambda x:(x['mean_r2'],-x['epoch'])) for a in plan.ARMS}
        s=selected['A-S1']; report={'history':history,'selected':selected,'runtime_seconds':time.monotonic()-started,'optimizer_steps':self.step}
        for arm in ('A-TB4','A-DC2'):
            c=selected[arm]; delta=c['mean_r2']-s['mean_r2']; ps=[c['per_session_r2'][k]-s['per_session_r2'][k] for k in sorted(val)]
            report.setdefault('gates',{})[arm]={'mean_delta_vs_s1':delta,'positive_sessions':sum(x>0 for x in ps),'worst_delta':min(ps),'delta_vs_zero':None,'passed':delta>=.003 and sum(x>0 for x in ps)==2 and min(ps)>=-.002}
        report['scalar_control']={'selected_alpha':s['params'][0],'negative':s['params'][0]<0,'validation_gain_vs_v1_target':None}
        return report

__all__=('APFCCoordinatedRunner','RunnerError','validate_gpu0_only')
