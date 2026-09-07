"""Experiment-B v2 analytic estimator selection (CPU/source-only, append-only).

v2 supersedes v1's prelaunch only.  Its essential fairness contract is one row per observed
canonical direction for *every* candidate: ordinary T4, EB rate likelihood, 2H OLS, and
Poisson IRLS.  Directional counts/exposure are first aggregated, then Poisson's likelihood is
equal-direction normalised so trial replication cannot create a hidden arm advantage.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np

from mc_maze.t4_cross_budget_audit import REPEATABILITY_SEEDS, disjoint_trial_count_partitions
from mc_maze.t4_cross_budget_protocol import direction_design_metadata, fit_t4_prefix
from mc_maze.unit_side_features import CANONICAL_DIRECTIONS_RAD

N_SOURCE = 27; SUPPORT = 30; SCORE_START = 30; SCORE_STOP = 50
SEEDS = tuple(REPEATABILITY_SEEDS); EPS_NORM = 1e-8; RATE_FLOOR = 1e-6
EB_GRID = (0.25, 1.0, 4.0); IRLS_MAX_ITER = 32; IRLS_TOL = 1e-7; ETA_CLIP = (-12.0, 12.0)
CANDIDATES = ("eb_ridge", "second_harmonic", "poisson_irls")


@dataclass(frozen=True)
class Session:
    name: str; counts: np.ndarray; durations: np.ndarray; directions: np.ndarray; trial_ordinals: np.ndarray
    source_sha256: str = ""
    def __post_init__(self):
        if self.counts.ndim != 2 or self.counts.shape[1] != SCORE_STOP or not np.issubdtype(self.counts.dtype, np.integer): raise ValueError("counts must be integer [units,50]")
        if self.durations.shape != (SCORE_STOP,) or np.any(self.durations <= 0) or not np.isfinite(self.durations).all(): raise ValueError("durations must be finite [50]")
        if self.directions.shape != (SCORE_STOP,) or np.any((self.directions < -1) | (self.directions > 7)): raise ValueError("directions must be -1..7 [50]")
        if self.trial_ordinals.shape != (SCORE_STOP,) or np.any(np.diff(self.trial_ordinals) <= 0): raise ValueError("real chronological ordinals [50] required")


@dataclass(frozen=True)
class Fit:
    t4: np.ndarray; rank_deficient: int; nonconverged: int; invalid: int; metadata: Mapping[str, object]


def _aggregate(session: Session, counts: np.ndarray | None = None):
    """Equal-direction calibration rows: rates are arithmetic trial means, no replication weight."""
    c = np.asarray(session.counts if counts is None else counts, dtype=np.float64)[:, :SUPPORT]
    d, q = session.durations[:SUPPORT], session.directions[:SUPPORT]
    present = np.unique(q[q >= 0]); theta = np.asarray([CANONICAL_DIRECTIONS_RAD[int(x)] for x in present])
    rates = []; total_counts = []; exposures = []; means = []
    for direction in present:
        which = q == direction; # One equal-weight directional rate row for OLS/EB.
        trial_rates = c[:, which] / d[which][None, :]
        rates.append(trial_rates.mean(axis=1)); total_counts.append(c[:, which].sum(axis=1)); exposures.append(float(d[which].sum())); means.append(float(d[which].mean()))
    x1 = np.column_stack([np.ones(len(theta)), np.cos(theta), np.sin(theta)])
    x2 = np.column_stack([x1, np.cos(2 * theta), np.sin(2 * theta)])
    return x1, x2, np.asarray(rates).T, np.asarray(total_counts).T, np.asarray(exposures), np.asarray(means), present


def _t4(bac: np.ndarray):
    return np.column_stack([bac[:, 1], bac[:, 2], np.hypot(bac[:, 1], bac[:, 2]), bac[:, 0]])


def ordinary(session: Session, counts: np.ndarray | None = None) -> Fit:
    c = np.asarray(session.counts if counts is None else counts, dtype=np.float64)
    base = fit_t4_prefix(c[:, :SUPPORT] / session.durations[None, :SUPPORT], session.durations[:SUPPORT], session.directions[:SUPPORT], budget=SUPPORT)
    invalid = int((~np.isfinite(base.t4).all(axis=1)).sum())
    return Fit(np.asarray(base.t4, float), int(not base.fit_defined) * c.shape[0], 0, invalid, {"fit": "ordinary_equal_per_direction_mean"})


def eb_prior(training: Sequence[Session]) -> dict:
    rows = []
    for s in training:
        f = ordinary(s); ok = np.isfinite(f.t4).all(1)
        rows.append(np.column_stack([f.t4[ok, 3], f.t4[ok, 0], f.t4[ok, 1]]))
    z = np.concatenate(rows) if rows else np.empty((0, 3))
    if len(z) < 2: raise ValueError("no valid outer-training units for EB prior")
    cov = np.cov(z, rowvar=False, ddof=1); reg = max(float(np.trace(cov) / 3), 1e-6) * 1e-5
    # Fixed rotationally symmetric mean: baseline may be nonzero; directional population mean is zero.
    mean = np.asarray([float(z[:, 0].mean()), 0.0, 0.0])
    return {"mean_bac": mean, "covariance_bac": cov + np.eye(3) * reg, "regularization": reg,
            "source_names": [s.name for s in training], "source_hashes": [s.source_sha256 for s in training]}


def fit_eb(session: Session, prior: Mapping[str, object], lam: float, counts: np.ndarray | None = None) -> Fit:
    if lam not in EB_GRID: raise ValueError("unregistered EB lambda")
    x, _, rates, _, _, _, _ = _aggregate(session, counts); n = session.counts.shape[0]
    if np.linalg.matrix_rank(x) != 3: return Fit(np.full((n,4),np.nan),n,0,n,{"reason":"rank"})
    prec0 = np.linalg.pinv(np.asarray(prior["covariance_bac"], float)) / lam; out = np.full((n,3),np.nan)
    for u in range(n):
        beta = np.linalg.lstsq(x, rates[u], rcond=None)[0]; r = rates[u] - x @ beta; var = max(float(r @ r / max(1,len(r)-3)),1e-6); p = x.T@x/var + prec0
        try: out[u] = np.linalg.solve(p, x.T@rates[u]/var + prec0@np.asarray(prior["mean_bac"]))
        except np.linalg.LinAlgError: pass
    t = _t4(out); return Fit(t,0,0,int((~np.isfinite(t).all(1)).sum()), {"candidate":"eb_ridge","lambda":lam,"prior":dict(prior),"likelihood":"equal_unique_direction_rate_rows"})


def fit_2h(session: Session, counts: np.ndarray | None = None) -> Fit:
    _, x, rates, _, _, _, _ = _aggregate(session, counts); n=session.counts.shape[0]; rank=np.linalg.matrix_rank(x)
    if rank != 5: return Fit(np.full((n,4),np.nan),n,0,n,{"reason":"rank"})
    beta=np.linalg.lstsq(x,rates.T,rcond=None)[0].T; t=_t4(beta[:,:3])
    return Fit(t,0,0,int((~np.isfinite(t).all(1)).sum()), {"candidate":"second_harmonic","fit":"equal_unique_direction_rate_rows","export_and_score":"first_harmonic_only"})


def _project_poisson(beta):
    theta=np.asarray(CANONICAL_DIRECTIONS_RAD); x=np.column_stack([np.ones(8),np.cos(theta),np.sin(theta)])
    rate=np.exp(np.clip(x@beta,*ETA_CLIP)); return np.linalg.lstsq(x,rate,rcond=None)[0]


def fit_poisson(session: Session, counts: np.ndarray | None = None) -> Fit:
    x, _, _, summed, exposure, mean_exposure, _ = _aggregate(session, counts); n=session.counts.shape[0]; rank=np.linalg.matrix_rank(x)
    if rank != 3: return Fit(np.full((n,4),np.nan),n,0,n,{"reason":"rank"})
    # Directional aggregate is retained for receipt.  Equal-direction normalisation gives every
    # unique direction one row and one common exposure, preventing replication/imbalance weights.
    reference_exposure=max(float(np.median(mean_exposure)), RATE_FLOOR); y=summed/exposure[None,:]*reference_exposure
    out=np.full((n,3),np.nan); records=[]
    for u in range(n):
        beta=np.linalg.lstsq(x,np.log((y[u]+.5)/reference_exposure),rcond=None)[0]; converged=False; reason="max_iterations"
        for iteration in range(1,IRLS_MAX_ITER+1):
            mu=np.exp(np.clip(x@beta+math.log(reference_exposure),*ETA_CLIP)); h=x.T@(mu[:,None]*x)+np.eye(3)*1e-8; g=x.T@(y[u]-mu)
            try: step=np.linalg.solve(h,g)
            except np.linalg.LinAlgError: reason="singular_hessian"; break
            beta=beta+step
            if float(np.max(np.abs(step))) <= IRLS_TOL: converged=True; reason="converged"; break
        records.append({"unit":u,"iterations":iteration,"converged":converged,"failure_reason":None if converged else reason})
        if converged: out[u]=_project_poisson(beta)
    t=_t4(out); non=sum(not r["converged"] for r in records)
    return Fit(t,0,non,int((~np.isfinite(t).all(1)).sum()), {"candidate":"poisson_irls","aggregate":"counts_and_exposure_by_direction_then_equal_direction_normalized","reference_exposure":reference_exposure,"max_iterations":IRLS_MAX_ITER,"tolerance":IRLS_TOL,"eta_clip":list(ETA_CLIP),"unit_records":records})


def _rates(t4, dirs):
    ok=dirs>=0; out=np.full((len(t4),len(dirs)),np.nan); th=np.asarray([CANONICAL_DIRECTIONS_RAD[int(a)] for a in dirs[ok]])
    out[:,ok]=t4[:,3,None]+t4[:,0,None]*np.cos(th)+t4[:,1,None]*np.sin(th); return np.maximum(out,RATE_FLOOR)


def deviance(session: Session, fit: Fit):
    dirs=session.directions[SCORE_START:SCORE_STOP]; valid=dirs>=0; units=np.isfinite(fit.t4).all(1)
    if not valid.any() or not units.any(): return None
    y=session.counts[units,SCORE_START:SCORE_STOP][:,valid]; mu=_rates(fit.t4[units],dirs[valid])*session.durations[SCORE_START:SCORE_STOP][valid]
    pos=y>0; term=mu-y; term[pos]+=y[pos]*np.log(y[pos]/mu[pos]); return float(2*term.sum())


def reliability(session: Session, fitfn):
    records=[]
    for seed in SEEDS:
        parts=disjoint_trial_count_partitions(session.counts[:,:SUPPORT],seed=seed); full=[]
        for part in parts:
            c=np.zeros_like(session.counts); c[:,:SUPPORT]=part*2; full.append(c)
        o1,o2=ordinary(session,full[0]),ordinary(session,full[1]); c1,c2=fitfn(full[0]),fitfn(full[1]); vec=[x.t4[:,:2] for x in (o1,o2,c1,c2)]
        common=np.logical_and.reduce([np.isfinite(x).all(1)&(np.linalg.norm(x,axis=1)>EPS_NORM) for x in vec])
        if not common.any() or c1.nonconverged or c2.nonconverged or c1.invalid or c2.invalid: records.append({"seed":seed,"defined":False,"reason":"no_common_valid_or_candidate_health_failure"}); continue
        def med(a,b): return float(np.median((a[common]*b[common]).sum(1)/(np.linalg.norm(a[common],axis=1)*np.linalg.norm(b[common],axis=1))))
        records.append({"seed":seed,"defined":True,"common_valid":int(common.sum()),"ordinary":med(o1.t4[:,:2],o2.t4[:,:2]),"candidate":med(c1.t4[:,:2],c2.t4[:,:2])})
    if len([r for r in records if r["defined"]]) != len(SEEDS): return {"defined":False,"records":records,"delta":None}
    return {"defined":True,"records":records,"delta":float(np.median([r["candidate"]-r["ordinary"] for r in records]))}


def evaluate(session: Session, train: Sequence[Session], candidate: str, config: Mapping[str, object]):
    timings={}; start=time.perf_counter(); o=ordinary(session); timings["ordinary_s"]=time.perf_counter()-start
    prior=eb_prior(train) if candidate=="eb_ridge" else None
    fn=(lambda c:fit_eb(session,prior,float(config["lambda"]),c)) if candidate=="eb_ridge" else ((lambda c:fit_2h(session,c)) if candidate=="second_harmonic" else (lambda c:fit_poisson(session,c)))
    start=time.perf_counter(); f=fn(None); timings["candidate_s"]=time.perf_counter()-start
    start=time.perf_counter(); rel=reliability(session,fn); timings["reliability_s"]=time.perf_counter()-start
    od,cd=deviance(session,o),deviance(session,f); ratio=None if od is None or cd is None or od<=0 else cd/od
    pr=f.metadata.get("unit_records",[]); its=[r["iterations"] for r in pr]
    return {"session":session.name,"outer_train_names":[x.name for x in train],"outer_train_hashes":[x.source_sha256 for x in train],"config":dict(config),"ratio":ratio,"reliability":rel,"rank_increase":f.rank_deficient>o.rank_deficient,"nonconvergence_increase":f.nonconverged>o.nonconverged,"invalid_increase":f.invalid>o.invalid,"poisson_iterations":None if not pr else {"mean":float(np.mean(its)),"max":max(its),"q50":float(np.quantile(its,.5)),"q90":float(np.quantile(its,.9)),"failure_reasons":[r["failure_reason"] for r in pr if r["failure_reason"]]},"cost":timings,"candidate_metadata":f.metadata}


def gate(rows):
    if len(rows)!=N_SOURCE: return {"pass":False,"reason":"not_27_folds"}
    ratios=[r["ratio"] for r in rows]; rel=[r["reliability"]["delta"] for r in rows]
    if any(x is None or not np.isfinite(x) for x in ratios) or any(x is None or not np.isfinite(x) or not r["reliability"]["defined"] for x,r in zip(rel,rows)): return {"pass":False,"reason":"undefined_ratio_or_reliability"}
    joint=sum(x<=1 and y>=0 for x,y in zip(ratios,rel)); bad=any(r["rank_increase"] or r["nonconvergence_increase"] or r["invalid_increase"] for r in rows)
    ok=np.mean(ratios)<=.98 and np.mean(rel)>=.02 and joint>=20 and not bad
    return {"pass":bool(ok),"mean_ratio":float(np.mean(ratios)),"mean_reliability_delta":float(np.mean(rel)),"joint":joint,"reason":"pass" if ok else "threshold_or_health_failure"}


def choose_inner(train: Sequence[Session], candidate: str):
    """Return chosen config plus all 26-fold objectives; held session never enters its prior."""
    configs=([{"candidate":candidate,"lambda":x} for x in EB_GRID] if candidate=="eb_ridge" else [{"candidate":candidate}])
    table=[]
    for cfg in configs:
        values=[]; wall_start=time.perf_counter()
        for i,held in enumerate(train):
            row=evaluate(held,list(train[:i])+list(train[i+1:]),candidate,cfg)
            values.append(row["ratio"])
        finite=[x for x in values if x is not None and np.isfinite(x)]
        table.append({"config":cfg,"per_inner_fold_ratio":values,"defined_count":len(finite),"equal_session_mean_ratio":float(np.mean(finite)) if len(finite)==len(train) else None,"wall_clock_s":time.perf_counter()-wall_start})
    # Undefined has no selection privilege; then lower lambda makes the tie break deterministic.
    chosen=min(table,key=lambda r:(math.inf if r["equal_session_mean_ratio"] is None else r["equal_session_mean_ratio"],float(r["config"].get("lambda",0))))
    return {"chosen_config":dict(chosen["config"]),"all_candidate_objectives":table,"tie_break":"lower_equal_session_prospective_deviance_then_lower_lambda"}


def run_candidate(sessions: Sequence[Session], candidate: str):
    if len(sessions)!=N_SOURCE or len({x.name for x in sessions})!=N_SOURCE: raise ValueError("requires exactly 27 unique source sessions")
    rows=[]; failures=0
    for i,held in enumerate(sessions):
        train=list(sessions[:i])+list(sessions[i+1:]); inner_start=time.perf_counter(); selection=choose_inner(train,candidate)
        row=evaluate(held,train,candidate,selection["chosen_config"]); row["inner_selection"]=selection; row["cost"]["inner_selection_s"]=time.perf_counter()-inner_start; rows.append(row)
        joint=row["ratio"] is not None and row["ratio"]<=1 and row["reliability"]["defined"] and row["reliability"]["delta"]>=0
        failures+=int(not joint)
        if failures>7: return {"candidate":candidate,"rows":rows,"gate":{"pass":False,"reason":"fail_fast_joint_20_of_27_impossible","joint_failures":failures,"remaining_not_run":N_SOURCE-len(rows)}}
    return {"candidate":candidate,"rows":rows,"gate":gate(rows)}
