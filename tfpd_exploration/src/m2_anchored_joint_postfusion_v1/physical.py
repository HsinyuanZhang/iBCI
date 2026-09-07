"""AJPF post-attempt physical coordinator with no import-time runtime access."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import controller, plan


class PhysicalError(RuntimeError): pass
def _need(ok: bool, message: str) -> None:
    if not ok: raise PhysicalError(message)

@dataclass(frozen=True)
class LiveSourceStream:
    """Live-materialized source result; historical values are comparisons only."""
    batches: tuple[tuple[controller.M30Coordinate,...],...]
    coordinate_count: int
    coordinate_sha256: str
    batch_sha256: str

def recompute_live_stream(*, coordinates: Sequence[controller.M30Coordinate]) -> LiveSourceStream:
    """Recompute stream from coordinates then compare it to held APFG literals."""
    batches=controller.canonical_batches(coordinates)
    _need(sum(len(batch) for batch in batches)==len(coordinates),"AJPF live coordinate materialization drift")
    # This is the reviewed APFG source_authority_summary framing, recomputed
    # from live coordinates rather than accepted as a caller assertion.
    coordinate_sha=hashlib.sha256(json.dumps([
        (item.session,item.query_trial,item.window_start,item.state_digest)
        for batch in batches for item in batch
    ],separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
    batch_sha=hashlib.sha256(json.dumps([
        (batch[0].session,[item.state_digest for item in batch],[item.window_start for item in batch])
        for batch in batches
    ],separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
    _need(len(coordinates)==plan.SOURCE_COORDINATES and len(batches)==plan.SOURCE_GROUPS_PER_EPOCH,
          "AJPF live 91717/3455 count drift")
    _need(coordinate_sha==plan.SOURCE_COORDINATE_SHA256 and batch_sha==plan.SOURCE_BATCH_SHA256,
          "AJPF live APFG digest mismatch")
    return LiveSourceStream(batches=batches,coordinate_count=len(coordinates),coordinate_sha256=coordinate_sha,batch_sha256=batch_sha)

def _rng_state(torch: Any) -> tuple[object, tuple[Any,...], Any, Any]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state(0)

def _set_rng(torch: Any, state: tuple[object,tuple[Any,...],Any,Any]) -> None:
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2]); torch.cuda.set_rng_state(state[3],0)

def paired_three_arm_step(*, torch: Any, arms: Mapping[str, Any], batch: Any, mask: Any) -> dict[str, Any]:
    """One true paired group: one shared mask and restored post-mask RNG per arm.

    Arm objects must expose ``step(batch, mask)`` and return receipt-safe
    scalar evidence.  The future route supplies internal model adapters, not
    caller callbacks/public API.
    """
    _need(tuple(arms)==plan.ARM_ORDER and mask.ndim==2,"AJPF paired arm/mask topology drift")
    state=_rng_state(torch); evidence={}
    for arm in plan.ARM_ORDER:
        _set_rng(torch,state)
        row=dict(arms[arm].step(batch=batch,mask=mask))
        _need(row.get("forward_calls")==1 and row.get("backward_calls")==1 and row.get("adam_steps")==1
              and row.get("teacher_forward_calls")==0 and row.get("internal_unit_dropout_draws")==0,
              "AJPF paired step evidence drift")
        evidence[arm]=row
    final=_rng_state(torch)
    _need(final==state,"AJPF paired end RNG drift")
    return evidence

def smoke_and_epochs(*, torch: Any, arms: Mapping[str,Any], batches: Sequence[Any], masks: Sequence[Any]) -> tuple[dict[str,Any], tuple[dict[str,Any],...]]:
    """Same iterator: first 12 groups count in epoch1, then exact 12 epochs."""
    _need(len(batches)==plan.SOURCE_GROUPS_PER_EPOCH and len(masks)==len(batches),"AJPF live shared group count/mask drift")
    smoke=[]; epochs=[]
    for epoch in range(1,plan.EPOCHS+1):
        rows=[]
        for ordinal,(batch,mask) in enumerate(zip(batches,masks)):
            row=paired_three_arm_step(torch=torch,arms=arms,batch=batch,mask=mask)
            rows.append(row)
            if epoch==1 and ordinal<12: smoke.append(row)
        _need(len(rows)==plan.SOURCE_GROUPS_PER_EPOCH,"AJPF epoch group shortage")
        epochs.append({"epoch":epoch,"shared_groups":len(rows),"sparse_first":rows[0],"sparse_last":rows[-1]})
    _need(len(smoke)==12,"AJPF same-iterator smoke drift")
    return {"groups":12,"counted_in_epoch1":True,"arms":list(plan.ARM_ORDER)},tuple(epochs)
