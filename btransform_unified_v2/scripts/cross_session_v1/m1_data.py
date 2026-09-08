"""Strict source/target readers for the public M1 cross-session protocol."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import hashlib, json, os
import numpy as np

FOLDS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
SUPPORT, TRAIN, VAL, QUERY = (0, 10), (10, 310), (310, None), (10, 210)
DATA = Path(__file__).resolve().parents[3] / "SPINT-main/data/000941/sub-MonkeyL-held-in-calib"
def source_for(target: str) -> tuple[str, ...]:
    if target not in FOLDS: raise ValueError(target)
    return tuple(s for s in FOLDS if s != target)
def nwb_path(session: str) -> Path: return DATA / f"sub-MonkeyL-held-in-calib_{session}_behavior+ecephys.nwb"
def contract(target: str) -> dict[str, object]:
    return {"target":target,"sources":list(source_for(target)),"target_support_trials":list(SUPPORT),"target_query_trials":list(QUERY),"source_support_trials":list(SUPPORT),"source_train_trials":list(TRAIN),"source_val_trials":[310,"end"],"target_query_labels_for_selection":False,"target_optimizer_steps":0,"padding":"each range is sliced at its trial boundary then independently left-padded/reset"}
def assert_disjoint(row: Mapping[str, object]) -> None:
    if tuple(row["target_support_trials"]) != SUPPORT or tuple(row["target_query_trials"]) != QUERY or tuple(row["source_support_trials"]) != SUPPORT or tuple(row["source_train_trials"]) != TRAIN or row["source_val_trials"] != [310,"end"]: raise RuntimeError("M1 trial contract drift")

def _records(sessions: tuple[str, ...]) -> dict[str, Any]:
    from falcon_challenge.config import FalconConfig, FalconTask
    from src.data.falcon_datamodule import FalconDataModule
    dm=FalconDataModule(task="m1",data_dir=str(DATA.parent),heldin_session_names=[],batch_size=32,window_size=100,calibration_n_trials=10,random_calibration=False,smooth_calibration=False,max_trial_length=1024,standardize_covariates=False,use_intertrials=True,use_calib_intertrials=False,trial_feature_type="raw",interpolate_trials=True,interpolate_trials_kind="cubic",pad_value=-1.,num_workers=0,pin_memory=False,include_heldout_in_fit=False,include_heldout_in_test=False)
    task=FalconConfig(task=FalconTask.m1).task
    return {s:dm.prepare_session_data(nwb_path(s),task,standardize_covariates=False,covariates_mean=None,covariates_std=None,use_intertrials=True) for s in sessions}

def _slice_trials(record: Mapping[str, Any], start: int, stop: int | None) -> dict[str, Any]:
    """Copy an interval and force a new raw-stream boundary at its first trial."""
    changes=np.flatnonzero(np.asarray(record["trial_change"],dtype=bool))
    if start < 0 or start >= len(changes) or (stop is not None and (stop <= start or stop > len(changes))): raise ValueError(f"invalid trial slice [{start},{stop})")
    left=int(changes[start]); right=int(changes[stop]) if stop is not None and stop < len(changes) else len(record["neural"]); out={}
    for k,v in record.items(): out[k]=np.ascontiguousarray(v[left:right].copy()) if isinstance(v,np.ndarray) and v.ndim and v.shape[0] == len(record["neural"]) else v
    out["trial_change"]=np.asarray(out["trial_change"],dtype=bool); out["trial_change"][0]=True
    return out

def _dataset(records: Mapping[str, Any], calib: Mapping[str, Any]):
    from src.data.falcon_datamodule import FalconDataset
    return FalconDataset(sessions_dict=records,calib_sessions_dict=calib,window_size=100,split="train",calibration_n_trials=10,random_calibration=False,smooth_calibration=False,max_trial_length=1024,use_calib_intertrials=False,trial_feature_type="raw",remove_still_times=False,remove_calib_still_times=False,use_calib_active_segments=False,calib_n_active_segments=1,interpolate_trials=True,interpolate_trials_kind="cubic",pad_value=-1.)
def _bank(session: str, carrier: np.ndarray):
    from btransform_unified_v1.bank import TaskBank, array_sha256
    z=np.zeros((64,100),np.float32)
    return TaskBank(session,z,carrier,np.ones(64,bool),np.zeros((0,100,64),np.float32),np.zeros((0,16),np.float32),np.zeros(0,np.int64),{"shape":[64,100],"trial_count":10,"estimator":"source_only_rsyn3","array_sha256":array_sha256(z),"budget":10})
def _assert_reset(ds: Any, expected: tuple[int,int|None]) -> None:
    for s, starts in ds.trial_start_indices.items():
        if not len(starts) or int(starts[0]) != ds.window_size-1: raise RuntimeError(f"{s}: non-reset trial range {expected}")

def _rectify(emg: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    raw=np.asarray(emg,dtype=np.float64); neg=raw < 0
    return np.maximum(raw,0.0), {"negative_count":int(neg.sum()),"negative_fraction":float(neg.mean()),"input_min":float(raw.min()),"rectified":True}

def _source_carriers(sources: tuple[str,...], cache: Path|None) -> tuple[dict[str,np.ndarray],dict[str,np.ndarray]]:
    key=hashlib.sha256("|".join(sources).encode()).hexdigest(); path=cache/f"rsyn3_{key}.npz" if cache else None
    if path and path.is_file():
        with np.load(path,allow_pickle=False) as z:
            if str(z["sources"].item()) == "|".join(sources) and {"basis_dictionary","basis_scale","normalizer_mean","normalizer_scale"}.issubset(z.files): return ({s:np.ascontiguousarray(z[s]) for s in sources},{k:np.ascontiguousarray(z[k]) for k in ("basis_dictionary","basis_scale","normalizer_mean","normalizer_scale")})
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as raw_data, syn3
    loaded={s:raw_data.load_support_bins(nwb_path(s),emg_trial_stop=310,neural_trial_stop=10) for s in sources}; rectified={s:_rectify(loaded[s].emg) for s in sources}; basis=syn3.fit_source_nmf(np.concatenate([rectified[s][0] for s in sources])); raw={}
    for s in sources:
        support,_=_rectify(loaded[s].emg[loaded[s].emg_trial_ids < 10]); sc=syn3.project_basis(support,basis); w,i=syn3.fit_all_units(sc,loaded[s].rates); raw[s]=syn3.carrier_from_encoding(w,i)
    mean,scale=syn3.source_normalizer(list(raw.values())); out={s:np.ascontiguousarray(syn3.normalize_carriers(raw[s],mean,scale),dtype=np.float32) for s in sources}
    if path:
        path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(f".{os.getpid()}.tmp.npz"); np.savez_compressed(tmp,sources=np.array("|".join(sources)),basis_dictionary=basis.dictionary,basis_scale=basis.scale,normalizer_mean=mean,normalizer_scale=scale,**out); os.replace(tmp,path)
        info=path.with_suffix(".json"); t=info.with_suffix(f".{os.getpid()}.json.tmp"); t.write_text(json.dumps({"sources":list(sources),"fit":"source EMG [0,310), source M10 only","rectification":{s:rectified[s][1] for s in sources}},sort_keys=True)); os.replace(t,info)
    return out,{"basis_dictionary":np.ascontiguousarray(basis.dictionary),"basis_scale":np.ascontiguousarray(basis.scale),"normalizer_mean":np.ascontiguousarray(mean),"normalizer_scale":np.ascontiguousarray(scale)}

def materialize_sources(target: str, *, cache: Path|None=None) -> dict[str,Any]:
    """Training reader: never opens target data, target calibration, or target labels."""
    sources=source_for(target); records=_records(sources); support={s:_slice_trials(records[s],*SUPPORT) for s in sources}; train_raw={s:_slice_trials(records[s],*TRAIN) for s in sources}; val_raw={s:_slice_trials(records[s],*VAL) for s in sources}; train,val=_dataset(train_raw,support),_dataset(val_raw,support); _assert_reset(train,TRAIN); _assert_reset(val,VAL)
    carriers,rsyn3=_source_carriers(sources,cache); return {"sources":sources,"train":train,"val":val,"banks":{s:_bank(s,carriers[s]) for s in sources},"calib":{s:np.ascontiguousarray(train.calib_trialized_neural_features[s][:10],dtype=np.float32) for s in sources},"rsyn3":rsyn3,"basis_scope":"source EMG [0,310), source carrier M10, source-only normalizer"}

def materialize_target(target: str, source: Mapping[str,Any], arm: str) -> dict[str,Any]:
    """Score-only target reader, called only after source-only selection."""
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as raw_data, syn3
    record=_records((target,))[target]
    if arm == "Z_NONE":
        query=_slice_trials(record,*QUERY)
        class QueryOnly:
            window_size=100
            def __init__(self, q):
                self.neural_data={target:np.pad(q["neural"],((99,0),(0,0)),constant_values=0.)}; self.covariate_data={target:np.pad(q["covariates"],((99,0),(0,0)),constant_values=0.)}; self.eval_mask={target:np.pad(q["eval_mask"],(99,0),constant_values=False)}; self.window_indices=[(target,i) for i in range(len(self.neural_data[target])-99) if self.eval_mask[target][i+99]]; self.trial_start_indices={target:np.array([99])}
            def __len__(self): return len(self.window_indices)
            def __getitem__(self,i):
                _,st=self.window_indices[i]; return self.neural_data[target][st:st+100],self.covariate_data[target][st:st+100],None,target
        return {"target":QueryOnly(query)}
    ds=_dataset({target:_slice_trials(record,*QUERY)},{target:_slice_trials(record,*SUPPORT)}); _assert_reset(ds,QUERY)
    if arm == "B_ACTIVITY_ONLY": return {"target":ds,"target_bank":_bank(target,np.zeros((64,4),dtype=np.float32)),"target_calib":np.ascontiguousarray(ds.calib_trialized_neural_features[target][:10],dtype=np.float32)}
    sources=tuple(source["sources"]); loaded={s:raw_data.load_support_bins(nwb_path(s),emg_trial_stop=310,neural_trial_stop=10) for s in sources}; basis=syn3.fit_source_nmf(np.concatenate([_rectify(loaded[s].emg)[0] for s in sources])); raw=[]
    for s in sources:
        sc=syn3.project_basis(_rectify(loaded[s].emg[loaded[s].emg_trial_ids<10])[0],basis); w,i=syn3.fit_all_units(sc,loaded[s].rates); raw.append(syn3.carrier_from_encoding(w,i))
    mean,scale=syn3.source_normalizer(raw); z=raw_data.load_support_bins(nwb_path(target),emg_trial_stop=10,neural_trial_stop=10); sc=syn3.project_basis(_rectify(z.emg)[0],basis); w,i=syn3.fit_all_units(sc,z.rates); carrier=np.ascontiguousarray(syn3.normalize_carriers(syn3.carrier_from_encoding(w,i),mean,scale),dtype=np.float32)
    return {"target":ds,"target_bank":_bank(target,carrier),"target_calib":np.ascontiguousarray(ds.calib_trialized_neural_features[target][:10],dtype=np.float32)}
