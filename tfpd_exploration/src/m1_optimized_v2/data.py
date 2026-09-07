"""New strict source-only M1 dataloader; it never asks legacy setup to find target files."""
from __future__ import annotations
import sys
from collections import OrderedDict
from typing import Any
import numpy as np
import torch
from . import bank, plan

def _streaming_path():
    p=str(plan.REPO_ROOT / "streaming_calibration_exp")
    if p not in sys.path: sys.path.insert(0,p)

class SourceCarrierDataset:
    def __init__(self,base,loaded): self.base,self.loaded=base,loaded
    def __len__(self): return len(self.base)
    def __getitem__(self,index):
        neural,target,calib,session=self.base[index][:4]; name=session.decode() if isinstance(session,bytes) else str(session)
        if name not in plan.SOURCE_SESSIONS: raise RuntimeError("non-source record presented to source-only dataset")
        return neural,target,calib,session,np.ascontiguousarray(self.loaded["normalized"][name],dtype=np.float32)
    def __getattr__(self,name): return getattr(self.base,name)

def build_source_only_datamodule(loaded=None):
    _streaming_path()
    from src.data.m1_version_b_source_loso_datamodule import M1VersionBSourceOnlyFitDataModule
    loaded=loaded or bank.load()
    # Class itself constructs individual source filenames, never globbing the outer file.
    dm=M1VersionBSourceOnlyFitDataModule(task="m1",data_dir=str(plan.DATA_DIR),source_session_names=list(plan.SOURCE_SESSIONS),heldin_session_names=list(plan.SOURCE_SESSIONS),batch_size=32,window_size=plan.WINDOW,calibration_n_trials=10,random_calibration=False,smooth_calibration=False,max_trial_length=1024,standardize_covariates=False,use_intertrials=True,use_calib_intertrials=False,trial_feature_type="raw",interpolate_trials=True,interpolate_trials_kind="cubic",pad_value=-1.,validation_protocol="loso",loso_fold=0,include_heldout_in_fit=False,include_heldout_in_test=False,query_start_trial=0,heldin_query_start_trial=10,heldin_query_end_trial=210,allow_empty_heldout_query=False,num_workers=0,pin_memory=False,sampler_seed=plan.SEED,balance_session_batches=False,reshuffle_train_sampler_each_epoch=False,afc4_arm="none")
    dm.setup("fit")
    if getattr(dm,"target_path",None) is not None or dm.val_heldin_dataset is not None or dm.val_heldout_dataset is not None: raise RuntimeError("source-only adapter unexpectedly materialized a target/query")
    if tuple(dm.train_session_names) != plan.SOURCE_SESSIONS: raise RuntimeError("source roster drift")
    dm.train_dataset=SourceCarrierDataset(dm.train_dataset,loaded)
    # Replace inherited fold-1/epoch narrative with a literal, truthful receipt.
    def manifest():
        paths=getattr(dm,"source_paths",OrderedDict())
        return {"schema":"m1_optimized_v2_source_only_fit_v1","decoder_revision":plan.DECODER_REVISION,"carrier_revision":plan.CARRIER_REVISION,"outer_fold":0,"outer_session_name_only":plan.OUTER_SESSION,"source_sessions":list(plan.SOURCE_SESSIONS),"validation_sessions":[],"target_path_resolved_during_fit":False,"target_file_opened":False,"target_query_values_read":False,"target_backpropagation":False,"selection":"source-development-only; fixed candidate budget; no outer terminal eval","checkpoint_selection":"fixed epoch, never target query","formal":False,"evalai":False,"source_files":{n:str(p) for n,p in paths.items()}}
    dm.get_split_manifest=manifest
    return dm

def materialize_source_banks():
    from .calibration import load_materializer
    dm=build_source_only_datamodule(); mat=load_materializer(); out={}
    for name in plan.SOURCE_SESSIONS:
        calib=dm.train_dataset.base.calib_trialized_neural_features[name][:10]
        out[name]=mat.materialize(torch.as_tensor(calib[None,...],dtype=torch.float32),name)
    return out
