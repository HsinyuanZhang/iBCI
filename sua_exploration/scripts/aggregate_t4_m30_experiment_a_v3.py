#!/usr/bin/env python3
"""Write-once, fail-closed Experiment-A M30 aggregate (never launches training)."""
from __future__ import annotations

import argparse, hashlib, itertools, json, math
from pathlib import Path
import numpy as np

ARMS = ("z4", "ph4", "ac4", "mb4", "b4", "ls4")
REFERENCES = ("t4", "ts4")
SEEDS = (42, 43, 44)
EPOCHS = tuple(range(5, 13))
SESSIONS = ("sub-C_ses-CO-20151103","sub-C_ses-CO-20151104","sub-C_ses-CO-20151106","sub-C_ses-CO-20151109","sub-C_ses-CO-20151110","sub-C_ses-CO-20151112")
REFERENCE_SHA256 = {
    "t4": {42: "b8f659a46ad55eea766cbad1be70e1cc99df4c3c4c6c38863f2a5a5ee3104148", 43: "e18a52a750b44f426ba1e39f3a8f791806f53228e5a8a69b73a491178b8a09b4", 44: "704f5a40bb07e53dc3267d4ed70e434255c84cb9e9b60dcfcb8dd7dcecac3e64"},
    "ts4": {42: "0178e384eb8976b931b0fcc47ce53354e84500a65672ee51fd4d9c19ab41c841", 43: "ca146f795aa342560a1220c7a6e3218f4370f10493bbbb76f7e808a1591936ba", 44: "e697e870d40efe5357b9c3ea410eb286fdd2d537e0134007b2b33f19d6641f7d"},
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(value: bool, message: str) -> None:
    if not value: raise ValueError(message)

def _wilcoxon_exact(values: np.ndarray) -> dict:
    """Two-sided exact signed-rank p-value by enumerating the 2^n sign flips."""
    x = np.asarray(values, dtype=float)
    nz = x[x != 0]
    if not len(nz): return {"n_nonzero": 0, "statistic": 0.0, "p_two_sided": 1.0, "method": "exact_enumeration_all_zero"}
    # Average ranks for tied absolute differences.
    order = np.argsort(np.abs(nz)); ranks = np.empty(len(nz), float); i = 0
    while i < len(nz):
        j = i + 1
        while j < len(nz) and np.isclose(abs(nz[order[j]]), abs(nz[order[i]])): j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0; i = j
    observed = float(ranks[nz > 0].sum()); total = float(ranks.sum()); stat = min(observed, total-observed)
    scores = np.asarray([sum(ranks[k] for k, sign in enumerate(signs) if sign) for signs in itertools.product((False, True), repeat=len(ranks))])
    p = float(np.mean(np.minimum(scores, total-scores) <= stat + 1e-12))
    return {"n_nonzero": int(len(nz)), "statistic": stat, "p_two_sided": p, "method": "exact_sign_flip_enumeration"}

def _hierarchical_ci(cells: np.ndarray, rng: np.random.Generator, draws: int = 50000) -> dict:
    """Resample sessions outer, then seed values within the sampled session."""
    n_seed, n_session = cells.shape
    boot = np.empty(draws)
    for i in range(draws):
        session_idx = rng.integers(n_session, size=n_session)
        boot[i] = np.mean([cells[rng.integers(n_seed, size=n_seed), session].mean() for session in session_idx])
    return {"draws": draws, "resampling": "sessions outer, then seeds within each sampled session (preserves session pairing)", "lower_95": float(np.quantile(boot,.025)), "upper_95": float(np.quantile(boot,.975))}

def _load(path: Path, arm: str, seed: int, *, reference: bool, expected_train_source: str | None = None, status_dir: Path | None = None) -> tuple[np.ndarray, list[str], dict]:
    require(path.is_file(), f"missing artifact: {path}")
    if reference: require(sha(path) == REFERENCE_SHA256[arm][seed], f"unqualified reference SHA: {path}")
    d = json.loads(path.read_text())
    protocol, epochs = d.get("protocol", {}), d.get("epoch_list")
    require((d.get("seed"),d.get("variant"),d.get("signal_view"),epochs) == (seed,"B3S","sua",list(EPOCHS)), f"artifact identity/epochs: {path}")
    require((protocol.get("calibration_n"),protocol.get("pool_size"),protocol.get("total_epochs"),protocol.get("burn_in_epochs")) == (30,30,12,4), f"M30 protocol: {path}")
    require(d.get("no_test_files_evaluated") is True and d.get("uses_backward_gradients") is False, f"test/gradient contract: {path}")
    meta_path = Path(d["run_metadata_path"])
    require(meta_path.is_file() and sha(meta_path) == d.get("run_metadata_sha256"), f"metadata hash: {path}")
    meta = json.loads(meta_path.read_text()); side = meta.get("side_features", {})
    require(meta.get("status") == "completed" and meta.get("held_out_test_evaluated") is False, f"completed validation-only metadata: {path}")
    require(side.get("group") == arm and side.get("side_dim") == 4, f"descriptor identity: {path}")
    operational_cost = None
    if not reference:
        require(status_dir is not None, 'r4 requires fixed cell status directory')
        status_path=status_dir/f'{arm}_s{seed}.json'; require(status_path.is_file(), f'missing status: {status_path}')
        status=json.loads(status_path.read_text()); require((status.get('arm'),status.get('seed'),status.get('status'),status.get('exit_code'))==(arm,seed,'completed',0), f'bad/wrong cell status: {status_path}')
        receipt = meta_path.parent / "post_run_cost_receipt.json"
        require(receipt.is_file(), f"missing post-run cost receipt: {receipt}")
        cost = json.loads(receipt.read_text())
        require(cost.get("status") == "completed" and cost.get("run_metadata_sha256") == sha(meta_path), f"cost receipt binding: {receipt}")
        require(cost.get("accelerator") == "gpu", f"GPU operational receipt required: {receipt}")
        require(cost.get("train_variant_source_sha256") == expected_train_source, f"authorized train source receipt: {receipt}")
        for key in ("fit_wall_clock_seconds","cuda_peak_memory_allocated_bytes","cuda_peak_memory_reserved_bytes"):
            require(key in cost and isinstance(cost[key], (int,float)) and cost[key] > 0, f"cost field {key}: {receipt}")
        operational_cost = {key: cost[key] for key in ("fit_wall_clock_seconds", "cuda_peak_memory_allocated_bytes", "cuda_peak_memory_reserved_bytes", "accelerator", "train_variant_source_sha256")}
    per_epoch = d["per_epoch"]; names = sorted(per_epoch[str(EPOCHS[0])]["per_session_r2"])
    require(names == list(SESSIONS), f"exact frozen validation sessions: {path}")
    values = np.asarray([[per_epoch[str(e)]["per_session_r2"][n] for n in names] for e in EPOCHS], dtype=float)
    require(np.isfinite(values).all(), f"non-finite score: {path}")
    return values, names, {"artifact_sha256":sha(path),"run_metadata_sha256":sha(meta_path),"operational_cost":operational_cost}

def summarize(delta: np.ndarray, rng: np.random.Generator) -> dict:
    # 8 epoch means yield the protocol 3x6 paired observations.
    require(delta.shape == (3,8,6), f"expected seed×epoch×session (3,8,6), got {delta.shape}")
    cells = delta.mean(axis=1); seed = cells.mean(axis=1); session = cells.mean(axis=0)
    mean = float(cells.mean()); se = float(seed.std(ddof=1)/math.sqrt(3)); unpaired = float("nan")
    return {"all_18_pair_deltas": cells.tolist(), "mean_delta": mean, "seed_mean_deltas": seed.tolist(), "session_mean_deltas": session.tolist(), "seed_mean_se_paired":se, "mde_2se":2*se, "paired_two_se_lower":mean-2*se, "positive_seed_means":int((seed>0).sum()), "positive_session_means":int((session>0).sum()), "hierarchical_bootstrap":_hierarchical_ci(cells,rng), "exact_paired_wilcoxon_sessions":_wilcoxon_exact(session), "per_epoch_dispersion":{"mean":delta.mean(axis=(0,2)).tolist(),"sd":delta.std(axis=(0,2),ddof=1).tolist()}, "paired_vs_unpaired":{"note":"computed below per contrast from arm seed scores"}}

def decide(stats: dict) -> str:
    effective = stats["mean_delta"] >= .03 and stats["positive_session_means"] >= 5 and stats["positive_seed_means"] == 3 and stats["paired_two_se_lower"] > 0 and stats["hierarchical_bootstrap"]["lower_95"] > 0
    if effective: return "effective"
    if stats["mean_delta"] + 2*stats["seed_mean_se_paired"] < .03: return "ineffective_for_practical_0.03"
    return "indeterminate"

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--result-dir",type=Path,required=True); p.add_argument("--reference-dir",type=Path,required=True); p.add_argument("--status-dir",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--prelaunch",type=Path,required=True); p.add_argument("--prelaunch-sha256",required=True); p.add_argument("--authorization",type=Path,required=True); p.add_argument("--authorization-sha256",required=True); p.add_argument("--bootstrap-seed",type=int,default=20260802); a=p.parse_args()
    require(not a.out.exists(), f"write-once output exists: {a.out}")
    require(sha(a.prelaunch)==a.prelaunch_sha256 and sha(a.authorization)==a.authorization_sha256, 'authorization/prelaunch hash drift')
    pre=json.loads(a.prelaunch.read_text()); auth=json.loads(a.authorization.read_text()).get('authorization',{})
    require(auth.get('gpu_launch_authorized') is True and pre.get('status') == 'PASS', 'GPU authorization or prelaunch status')
    require(auth.get('prelaunch_receipt_sha256')==a.prelaunch_sha256 and auth.get('prelaunch_source_sha256')==pre.get('source_sha256'), 'authorization v3 binding')
    source_root=Path(__file__).resolve().parents[2]
    for rel, expected in pre.get('source_sha256',{}).items(): require((source_root/rel).is_file() and sha(source_root/rel)==expected, f'v3 source drift: {rel}')
    data, names, evidence = {}, None, {}
    for arm in (*ARMS,*REFERENCES):
        result_root = a.reference_dir if arm in REFERENCES else a.result_dir; ref = arm in REFERENCES; rows=[]
        for seed in SEEDS:
            x, n, e = _load(result_root/f"{arm}_s{seed}.json", arm, seed, reference=ref, expected_train_source=pre['source_sha256']['sua_exploration/scripts/train_variant_dandi688.py'], status_dir=None if ref else a.status_dir); rows.append(x); names=n if names is None else names; require(n==names, "session ordering drift"); evidence[f"{arm}_s{seed}"]=e
        data[arm]=np.stack(rows) # seed, epoch, session
    rng=np.random.default_rng(a.bootstrap_seed); contrasts={}
    for arm in ARMS:
        for ref in ("z4","t4","ts4"):
            if arm==ref: continue
            st=summarize(data[arm]-data[ref],rng)
            arm_seed=data[arm].mean(axis=(1,2)); ref_seed=data[ref].mean(axis=(1,2)); corr=float(np.corrcoef(arm_seed,ref_seed)[0,1]); unpaired=float(math.sqrt(arm_seed.var(ddof=1)/3+ref_seed.var(ddof=1)/3)); corr_out=corr if math.isfinite(corr) else None; st["paired_vs_unpaired"]={"unpaired_seed_mean_se":unpaired,"seed_correlation":corr_out,"seed_correlation_undefined_reason":None if corr_out is not None else "zero variance across three seed means","paired_se_is_smaller":st["seed_mean_se_paired"] < unpaired}; st["per_seed_dispersion"]={"arm_seed_scores":arm_seed.tolist(),"reference_seed_scores":ref_seed.tolist(),"arm_sd":float(arm_seed.std(ddof=1)),"reference_sd":float(ref_seed.std(ddof=1))}
            st["decision"]=decide(st); contrasts[f"{arm}_minus_{ref}"]=st
    primary={}
    for left, right in (("t4","z4"),("t4","ls4"),("t4","mb4"),("t4","ts4")):
        st=summarize(data[left]-data[right],rng); left_seed=data[left].mean(axis=(1,2)); right_seed=data[right].mean(axis=(1,2)); corr=float(np.corrcoef(left_seed,right_seed)[0,1]); corr_out=corr if math.isfinite(corr) else None; unpaired=float(math.sqrt(left_seed.var(ddof=1)/3+right_seed.var(ddof=1)/3)); st['paired_vs_unpaired']={'unpaired_seed_mean_se':unpaired,'seed_correlation':corr_out,'seed_correlation_undefined_reason':None if corr_out is not None else 'zero variance across three seed means','paired_se_is_smaller':st['seed_mean_se_paired'] < unpaired}; st['per_seed_dispersion']={'arm_seed_scores':left_seed.tolist(),'reference_seed_scores':right_seed.tolist(),'arm_sd':float(left_seed.std(ddof=1)),'reference_sd':float(right_seed.std(ddof=1))}; st["decision"]=decide(st); primary[f"{left}_minus_{right}"]=st
    suff={}
    for arm in ("ph4","ac4","mb4","b4"):
        comp=contrasts[f"{arm}_minus_t4"]; z=contrasts[f"{arm}_minus_z4"]
        suff[arm]={"noninferiority_margin":-0.03,"component_minus_t4_lower_paired_2se":comp["paired_two_se_lower"],"component_minus_t4_lower_hierarchical_95":comp["hierarchical_bootstrap"]["lower_95"],"component_vs_z4_state":z["decision"],"sufficient":comp["paired_two_se_lower"]>=-.03 and comp["hierarchical_bootstrap"]["lower_95"]>=-.03 and z["decision"]=="effective"}
    absolute={arm:{"mean_r2":float(values.mean()),"seed_mean_r2":values.mean(axis=(1,2)).tolist(),"session_mean_r2":values.mean(axis=(0,1)).tolist()} for arm,values in data.items()}
    trigger=[arm for arm in ("ph4","ac4","mb4","b4") if suff[arm]["sufficient"]]
    out={"schema_version":3,"status":"completed","formal_test_used":False,"conditional_row_shuffle_launched":False,"arms":list(ARMS),"qualified_reused_references":list(REFERENCES),"seeds":list(SEEDS),"epochs":list(EPOCHS),"sessions":names,"artifact_evidence":evidence,"per_cell_operational_costs":{k:v['operational_cost'] for k,v in evidence.items() if v['operational_cost'] is not None},"absolute_r2":absolute,"contrasts":contrasts,"primary_oriented_contrasts":primary,"component_sufficiency":suff,"conditional_row_shuffle_trigger_arms":trigger,"analysis_notes":{"reference_policy":"T4/TS4 only reused after exact artifact-SHA and metadata-contract verification","row_shuffle":"not authorized or launched; it remains conditional on sufficient PH4, AC4, MB4, or B4 evidence; trigger arms are enumerated above"}}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
if __name__ == "__main__": main()
