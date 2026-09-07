"""NLE5 v2: r1's fixed RFF16 model over the audited active tag ontology only."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
from sua_exploration.mc_maze import h1_event_carrier_nle5 as r1
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as hse5

SCHEMA="h1_event_carrier_nle5_source_screen_v2"; PROTOCOL="h1_event_carrier_nonlinear_label_embedding_20260812_v2"
ACTIVE_TAGS=("Reach","Orient","Shape","Grasp","Carry","Orient2")
INACTIVE_TAGS=("Release","SnapTo")
RANK,CARRIER_DIM,SUPPORT_BUDGETS=r1.RANK,r1.CARRIER_DIM,r1.SUPPORT_BUDGETS
RFF_WIDTH,RFF_SEED,SOURCE_RIDGE_LAMBDA,TARGET_RIDGE_LAMBDA=r1.RFF_WIDTH,r1.RFF_SEED,r1.SOURCE_RIDGE_LAMBDA,r1.TARGET_RIDGE_LAMBDA
MATERIAL_MEAN,MATERIAL_MEDIAN,MINIMUM_POSITIVE=r1.MATERIAL_MEAN,r1.MATERIAL_MEDIAN,r1.MINIMUM_POSITIVE

def _need(x:bool,m:str)->None:v1._need(x,m)
def source_names_for_outer(d:str)->tuple[str,...]:return r1.source_names_for_outer(d)
def _raw(events:Sequence[v1.MovementEvent],tags:Sequence[str]|None=None)->np.ndarray:
    chosen=tuple(e.tag for e in events) if tags is None else tuple(tags);_need(len(chosen)==len(events) and set(chosen).issubset(ACTIVE_TAGS),"NLE5v2 inactive/invalid tag entered scoring")
    return np.c_[np.stack([e.displacement for e in events]),np.asarray([e.start_time for e in events])[:,None],np.log(np.asarray([e.duration_seconds for e in events]))[:,None],np.asarray([[float(t==a) for a in ACTIVE_TAGS] for t in chosen])]
def _canon(a:np.ndarray)->np.ndarray:
    a=a.copy()
    for j in range(a.shape[1]):
        if a[np.argmax(abs(a[:,j])),j]<0:a[:,j]*=-1
    return a
@dataclass(frozen=True)
class NonlinearMap:
    outer_date:str;source_sessions:tuple[str,...];raw_mean:np.ndarray;raw_scale:np.ndarray;rff_weight:np.ndarray;rff_phase:np.ndarray;projection:np.ndarray;latent_scale:np.ndarray;source_event_count:int;map_sha256:str
    def transform(self,e:Sequence[v1.MovementEvent],tags:Sequence[str]|None=None)->np.ndarray:
        x=(_raw(e,tags)-self.raw_mean)/self.raw_scale;r=np.sqrt(2/RFF_WIDTH)*np.cos(x@self.rff_weight+self.rff_phase);z=r@self.projection/self.latent_scale;_need(z.shape==(len(e),4) and np.isfinite(z).all(),"NLE5v2 latent drift");return z
    def manifest(self)->dict[str,Any]:return {"outer_date":self.outer_date,"source_sessions":list(self.source_sessions),"source_event_count":self.source_event_count,"active_tags":list(ACTIVE_TAGS),"inactive_tags_excluded":list(INACTIVE_TAGS),"raw_width":15,"rff_width":16,"rff_seed":RFF_SEED,"output_rank":4,"array_sha256":{k:v1.array_sha256(getattr(self,k)) for k in ("raw_mean","raw_scale","rff_weight","rff_phase","projection","latent_scale")},"map_sha256":self.map_sha256}
def fit_nonlinear_map(s:Mapping[str,v1.EventSession],*,outer_date:str)->NonlinearMap:
    names=source_names_for_outer(outer_date);_need(all(v1.session_date(n)!=outer_date for n in names),"NLE5v2 outer leak");events=tuple(e for n in names for e in s[n].events);_need(all(e.tag in ACTIVE_TAGS for e in events),"NLE5v2 inactive tag in source")
    raw=_raw(events);mean=raw.mean(0);scale=np.maximum(raw.std(0),1e-8);_need(np.all(scale[-6:]>1e-8),"NLE5v2 active tag absent in outer source")
    rng=np.random.default_rng(RFF_SEED);w=rng.normal(size=(15,16));b=rng.uniform(0,2*np.pi,16);r=np.sqrt(2/16)*np.cos(((raw-mean)/scale)@w+b);y=np.stack([e.log_rates for e in events]);ys=np.maximum(y.std(0),1e-8);coef=np.linalg.solve(r.T@r+len(r)*SOURCE_RIDGE_LAMBDA*np.eye(16),r.T@((y-y.mean(0))/ys));left,*_=np.linalg.svd(coef,full_matrices=False);p=_canon(left[:,:4]);ls=np.maximum((r@p).std(0),1e-8)
    body={"protocol":PROTOCOL,"outer_date":outer_date,"source_sessions":list(names),"mean":v1.array_sha256(mean),"scale":v1.array_sha256(scale),"w":v1.array_sha256(w),"b":v1.array_sha256(b),"p":v1.array_sha256(p),"ls":v1.array_sha256(ls)}
    return NonlinearMap(outer_date,names,mean,scale,w,b,p,ls,len(events),v1.canonical_sha256(body))
def _arrays(s:v1.EventSession,m:NonlinearMap,b:int):
    a,l=s.events_before(b),s.events_after(b);_need(a and l and all(e.tag in ACTIVE_TAGS for e in a+l),"NLE5v2 inactive tag in support/later");return a,l,m.transform(a),np.stack([e.log_rates for e in a]),m.transform(l),np.stack([e.log_rates for e in l])
def evaluate_session(s:v1.EventSession,m:NonlinearMap,*,budget:int)->dict[str,Any]:
    _need(s.date==m.outer_date,"NLE5v2 map mismatch");a,l,z,y,zl,yl=_arrays(s,m,budget);correct,fit=r1.fit_carrier(z,y);lo,lm=r1._fixed_permutation(len(z),f"{PROTOCOL}:endpoint-label:{s.session_name}:M{budget}");label,lf=r1.fit_carrier(z[lo],y);to,tm=r1._fixed_permutation(len(z),f"{PROTOCOL}:tag:{s.session_name}:M{budget}");tag,tf=r1.fit_carrier(m.transform(a,[e.tag for e in np.asarray(a,object)[to]]),y);ro,rm=r1._fixed_permutation(176,f"{PROTOCOL}:row:{s.session_name}:M{budget}");row,rf=r1.fit_carrier(z,y);row=row[ro];inter=np.c_[np.zeros((176,4)),y.mean(0)];q=r1._rotation();rot,_=r1.fit_carrier(z@q,y);err=float(np.max(abs(r1.predict(correct,zl)-r1.predict(rot,zl@q))));_need(err<=1e-11,"NLE5v2 rotation")
    scores={"correct":v1.r2_by_channel(yl,r1.predict(correct,zl)),"label":v1.r2_by_channel(yl,r1.predict(label,zl)),"tag":v1.r2_by_channel(yl,r1.predict(tag,zl)),"row":v1.r2_by_channel(yl,r1.predict(row,zl)),"intercept":v1.r2_by_channel(yl,r1.predict(inter,zl))};good=np.logical_and.reduce([np.isfinite(x) for x in scores.values()]);med=lambda x:float(np.median(x[good]))
    return {"session":s.session_name,"date":s.date,"budget":budget,"support_events":len(a),"later_events":len(l),"carrier_dim":5,"defined_channels":int(good.sum()),"target_design_rank":fit["design_rank"],"median_r2_correct":med(scores["correct"]),"median_r2_label_shuffle":med(scores["label"]),"median_r2_tag_shuffle":med(scores["tag"]),"median_r2_row_shuffle":med(scores["row"]),"median_r2_intercept":med(scores["intercept"]),"median_delta_label_shuffle":med(scores["correct"]-scores["label"]),"median_delta_tag_shuffle":med(scores["correct"]-scores["tag"]),"median_delta_row_shuffle":med(scores["correct"]-scores["row"]),"median_delta_intercept":med(scores["correct"]-scores["intercept"]),"rotation_invariance_max_abs_error":err,"correct_fit":fit,"label_fit":lf,"tag_fit":tf,"row_fit":rf,"label_shuffle":lm,"tag_shuffle":tm,"row_shuffle":rm,"outer_future_used_only_for_scoring":True}
def aggregate(rows:Mapping[str,Mapping[str,Any]],base:Mapping[str,Mapping[str,Any]])->dict[str,Any]:
    x={n:{**r,"delta_vs_hse5":r["median_r2_correct"]-base[n]["median_r2_correct"]} for n,r in rows.items()};sm=lambda k:v1.paired_summary([(n,r[k]) for n,r in x.items()]);return {"correct_r2":sm("median_r2_correct"),"correct_minus_hse5":sm("delta_vs_hse5"),"correct_minus_label_shuffle":sm("median_delta_label_shuffle"),"correct_minus_tag_shuffle":sm("median_delta_tag_shuffle"),"correct_minus_row_shuffle":sm("median_delta_row_shuffle"),"correct_minus_intercept":sm("median_delta_intercept")}
def _pos(x:Mapping[str,Any])->bool:return x["defined_sessions"]==13 and x["mean"]>0 and x["median"]>0 and x["positive"]>=10 and x["leave_largest_absolute_out_mean"]>0
def gate(a:Mapping[str,Any])->dict[str,Any]:
    material=_pos(a["correct_minus_hse5"]) and a["correct_minus_hse5"]["mean"]>=.02 and a["correct_minus_hse5"]["median"]>=.01;c={k:_pos(a[k]) for k in ("correct_minus_label_shuffle","correct_minus_tag_shuffle","correct_minus_row_shuffle","correct_minus_intercept")};return {"passed":bool(material and all(c.values())),"material_gain_vs_hse5":bool(material),**c,"thresholds":{"mean_delta_vs_hse5":.02,"median_delta_vs_hse5":.01,"minimum_positive_sessions":10,"leave_largest_absolute_out_mean_positive":True}}
def run_screen(s:Mapping[str,v1.EventSession])->dict[str,Any]:
    _need(tuple(s)==v1.H1_HELDIN_SESSIONS,"NLE5v2 session order");maps={d:fit_nonlinear_map(s,outer_date=d) for d in v1.H1_DATES};bases={d:hse5.fit_source_all_event_basis(s,outer_date=d) for d in v1.H1_DATES};budgets={}
    for b in SUPPORT_BUDGETS:
        rows={n:evaluate_session(s[n],maps[v1.session_date(n)],budget=b) for n in v1.H1_HELDIN_SESSIONS};base={n:r1.hse5_baseline_session(s[n],bases[v1.session_date(n)],budget=b) for n in v1.H1_HELDIN_SESSIONS};a=aggregate(rows,base);budgets[f"M{b}"]={"budget_trials":b,"nonlinear_map_by_outer_date":{d:maps[d].manifest() for d in v1.H1_DATES},"hse5_basis_by_outer_date":{d:bases[d].manifest() for d in v1.H1_DATES},"hse5_baseline_sessions":base,"nle5_sessions":rows,"aggregate":a,"gate":gate(a)}
    passed=all(budgets[f"M{b}"]["gate"]["passed"] for b in SUPPORT_BUDGETS);return {"schema":SCHEMA,"protocol":PROTOCOL,"candidate_matrix_predeclared_before_data_run":True,"frozen_constants":{"carrier_dim":5,"embedding_dim":4,"active_tags":list(ACTIVE_TAGS),"raw_width":15,"rff_width":16,"rff_seed":RFF_SEED,"source_ridge_lambda":1.,"target_ridge_lambda":3.,"support_budgets":[3,4],"single_model_no_outer_expanded_grid":True},"budgets":budgets,"status":"PASS_CPU_NLE5_V2_MATERIAL" if passed else "STOP_CPU_NLE5_V2_NOT_MATERIAL","gpu_authorized_by_this_screen":False,"scope":{"public_held_in_calibration_nwbs_opened":13,"minival_nwbs_opened":0,"held_out_nwbs_opened":0,"formal_test_labels_opened":0,"dense_velocity_opened":False,"within_event_position_trajectory_opened":False,"target_session_optimizer_steps":0,"target_session_backward_steps":0,"decoder_constructed":False,"trainer_constructed":False,"cuda_used":False}}
