"""Falcon runtime for the sealed M1 R100 ACTIVITY_ONLY B checkpoint.

The payload contains static EMA-B3S E0 computed with literal zero side columns,
and literal zero direct T.  It is a CPU cached RIFT concat runtime.
"""
from __future__ import annotations
import argparse, io, json, pickle
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from falcon_challenge.interface import BCIDecoder
from btransform_unified_v1.bank import TaskBank
from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime

PAYLOAD_SCHEMA = "m1_rift_activity_only_r100_cached_falcon_payload_v1"
ARM = "B_ACTIVITY_ONLY"
CHANNELS, CONTEXT_BINS, EXPECTED_SESSION_COUNT, OFFICIAL_BATCH = 64, 100, 7, 8
EXPECTED_TAGS = {"20120924", "20120926", "20120927", "20120928", "20121004", "20121017", "20121024"}

class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"): module = module.replace("numpy._core", "numpy.core", 1)
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
        return super().find_class(module, name)

def load_payload(path: str | Path) -> dict:
    with open(path, "rb") as handle: return CPUUnpickler(handle).load()

def _need(ok: bool, message: str) -> None:
    if not ok: raise ValueError(message)

def _array_sha(value: np.ndarray) -> str:
    return __import__("hashlib").sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def _task_bank(tag: str, row: dict, *, session_id: str | None = None) -> TaskBank:
    raw_e0, raw_t, raw_mask = np.asarray(row["E0"]), np.asarray(row["T"]), np.asarray(row["unit_mask"])
    _need(raw_e0.dtype == np.float32 and raw_t.dtype == np.float32 and raw_mask.dtype == np.bool_, f"{tag}: payload dtype drift")
    e0=np.ascontiguousarray(raw_e0); carrier=np.ascontiguousarray(raw_t); mask=np.ascontiguousarray(raw_mask)
    _need(e0.shape == (64,100) and bool(np.isfinite(e0).all()), f"{tag}: E0 geometry/nonfinite")
    _need(carrier.shape == (64,4) and bool(np.isfinite(carrier).all()) and not bool(np.any(carrier)), f"{tag}: ACTIVITY_ONLY requires literal zero direct T")
    _need(mask.shape == (64,) and bool(mask.any()), f"{tag}: invalid unit mask")
    _need(row.get("e0_sha256") == _array_sha(e0) and row.get("t_sha256") == _array_sha(carrier), f"{tag}: payload array SHA drift")
    return TaskBank(session_id or str(row.get("session") or tag), e0, carrier, mask, np.zeros((0,100,64),np.float32), np.zeros((0,16),np.float32), np.zeros(0,np.int64),
                    {"shape":tuple(e0.shape),"trial_count":10,"budget":10,"estimator":"B EMA-B3S E0 with literal zero side/direct T","array_sha256":str(row["e0_sha256"])})

def validate_payload(payload: dict) -> None:
    _need(payload.get("schema") == PAYLOAD_SCHEMA, "unsupported payload schema")
    _need(payload.get("arm") == ARM, "payload arm must be B_ACTIVITY_ONLY")
    _need(payload.get("identity_interface") == "ema_b3s_zero_side_static_e0_concat", "payload identity route mismatch")
    _need(int(payload.get("context_bins",-1)) == CONTEXT_BINS and payload.get("bias_mode") == "recency", "payload R100/recency mismatch")
    route=payload.get("information_route",{})
    _need(route.get("encoder_side") == "literal_zero" and route.get("direct_t") == "literal_zero" and route.get("e0") == "ema_b3s(activity, literal_zero_side)", "payload does not seal B information routing")
    banks=payload.get("bank_by_dataset_tag",{})
    _need(isinstance(banks,dict) and set(banks) == EXPECTED_TAGS, "payload must have the exact seven official tags")
    for tag,row in banks.items():
        _need(isinstance(row, dict), f"{tag}: payload row must be an object")
        _task_bank(str(tag),row)

class M1RiftActivityOnlyCachedFalconDecoder(BCIDecoder):
    def __init__(self, task_config, model_path: str, batch_size: int=1):
        super().__init__(task_config=task_config,batch_size=batch_size)
        _need(1 <= int(batch_size) <= OFFICIAL_BATCH, "invalid batch size")
        self.task_config,self.batch_size=task_config,int(batch_size); payload=load_payload(model_path); validate_payload(payload)
        self.bank_by_dataset_tag=payload["bank_by_dataset_tag"]
        self.decoder=RiftConcatDecoder("m1",context_bins=100,bias_mode="recency",seed=42)
        missing,unexpected=self.decoder.load_state_dict({k:torch.as_tensor(v,dtype=torch.float32) for k,v in payload["ema_state_dict"].items()},strict=True)
        _need(not missing and not unexpected,"EMA state mismatch")
        self.decoder.eval(); [p.requires_grad_(False) for p in self.decoder.parameters()]
        self.decoder=self.decoder.to("cpu"); self._runtime: CpuRiftRuntime|None=None; self.kind="rift_cached_activity_only"; self.window_size=100
    @staticmethod
    def _stem(value) -> str: return Path(value).stem
    def reset(self,dataset_tags: Iterable[Path]=(Path(""),)):
        tags=[self.task_config.hash_dataset(self._stem(value)) for value in dataset_tags]
        _need(len(tags)<=OFFICIAL_BATCH and all(tag in self.bank_by_dataset_tag for tag in tags),"unknown or excessive dataset tags")
        self._runtime=CpuRiftRuntime(self.decoder,[_task_bank(tag,self.bank_by_dataset_tag[tag]) for tag in tags],[str(tag) for tag in tags],temporal_backend="cached")
    def on_done(self,dones: np.ndarray): return None
    def set_batch_size(self,batch_size:int):
        _need(1<=int(batch_size)<=OFFICIAL_BATCH,"invalid batch size"); self.batch_size=int(batch_size); self._runtime=None
    def observe(self,neural_observations:np.ndarray): return None
    def predict(self,neural_observations:np.ndarray)->np.ndarray:
        if self._runtime is None: raise RuntimeError("reset(dataset_tags) must be called before predict")
        value=np.asarray(neural_observations,np.float32); _need(value.ndim==2 and value.shape[1]==self.task_config.n_channels,"expected [B,64] observations")
        active=len(value); _need(active<=len(self._runtime.ids),"evaluator batch exceeds registered roster")
        if active<len(self._runtime.ids): value=np.pad(value,((0,len(self._runtime.ids)-active),(0,0)))
        valid=torch.zeros(len(self._runtime.ids),dtype=torch.bool); valid[:active]=True
        with torch.inference_mode(): out=self._runtime.advance(torch.from_numpy(np.ascontiguousarray(value)),valid_mask=valid)
        result=out[:active].numpy(); _need(bool(np.isfinite(result).all()),"nonfinite M1 result"); return result.astype(np.float32,copy=False)

def smoke(payload_path:str,window_path:str)->None:
    from falcon_challenge.config import FalconConfig,FalconTask
    bundle=np.load(window_path); decoder=M1RiftActivityOnlyCachedFalconDecoder(FalconConfig(task=FalconTask.m1),payload_path,1); decoder.reset([str(bundle["tag_stem"])])
    got=None
    for row in np.asarray(bundle["window"],np.float32): got=decoder.predict(row[None,:])
    expected=np.asarray(bundle["expected"],np.float32)
    if got is None or not np.allclose(got,expected,atol=1e-5,rtol=1e-5): raise RuntimeError("container smoke mismatch")
    print(json.dumps({"status":"CONTAINER_SMOKE_PASS","kind":decoder.kind},sort_keys=True))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--smoke-payload",required=True);p.add_argument("--smoke-window",required=True);a=p.parse_args();smoke(a.smoke_payload,a.smoke_window)
