"""Prospective-only H1 selected-EMA cold-history 2x2 phase skeleton.

No execution is authorized by this file: `run` additionally needs an external
hash-bound authorization and a completed 32-update resource smoke receipt.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from .cold_history import apply_cold_history

ARMS=("FLAT","ROUTE"); CELLS=tuple(f"{a}_{t}" for a in ARMS for t in ("CONTROL","PREFIX"))
EPOCHS, UPDATES, MICRO, EFFECTIVE, SEED = 2, 731, 8, 32, 42
PROTOCOL={"schema":"h1_selected_ema_cold_history_2x2_prospective_v1","cells":list(CELLS),"epochs":EPOCHS,"updates_per_epoch":UPDATES,"microbatch":MICRO,"effective_batch":EFFECTIVE,"lr":1e-5,"wd":.01,"clip_norm":1.,"ema":"fresh .9995; first successful update copies RAW","initial":"actual frozen selected epoch12 plain EMA, exact within-arm clones","history":"CONTROL p0; PREFIX p.5 per row, observed length uniform 1..699 selected else700; zero left history only/current bin and targets unchanged","sampler":"frozen all23212 source windows; original epochs1,2 identity bytes exactly; p.1 whole-unit dropout shared all four cells","reporting":"e1 source208 RAW+EMA only; e2 RAW+EMA ALL20325; no selection/promotion","primary":"fixed e2 EMA complete20325 pooled R2 PREFIX minus matched CONTROL","segments":"fixed cold8702/full11623 IDs hash-bound","forbidden":["minival selection","official","C2","new split","automatic launch"],"hard_budget_seconds":14400}

def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clone_cells(models: dict[str, torch.nn.Module]) -> dict[str, torch.nn.Module]:
    """Strict paired copies from one frozen selected model per arm."""
    if set(models) != set(ARMS): raise ValueError("requires FLAT/ROUTE bases")
    cells={}
    for arm, model in models.items():
        cells[f"{arm}_CONTROL"]=copy.deepcopy(model); cells[f"{arm}_PREFIX"]=copy.deepcopy(model)
        if state_digest(cells[f"{arm}_CONTROL"].state_dict()) != state_digest(cells[f"{arm}_PREFIX"].state_dict()): raise RuntimeError("within-arm initial clone drift")
    return cells

def state_digest(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes())
    return h.hexdigest()

def require_authorization(authorization: Path, *, code: dict, bindings: dict, output: Path, smoke: Path) -> dict:
    if output.exists(): raise FileExistsError(output)
    auth=json.loads(authorization.read_text()); receipt=json.loads(smoke.read_text())
    if (auth.get("protocol") != PROTOCOL or auth.get("protocol_sha256") != digest(PROTOCOL) or auth.get("code") != code or auth.get("bindings") != bindings or auth.get("output") != str(output) or receipt.get("status") != "PASS_32_UPDATE_RESOURCE_SMOKE" or receipt.get("bindings") != bindings): raise RuntimeError("external prospective authorization/resource smoke mismatch")
    return auth

def train_step(cells, optimizers, emas, x, target, bank, keep, *, epoch, batch_id):
    """One matched effective batch. Optimizer grouping/EMA are supplied by runner."""
    full,_=apply_cold_history(x,seed=SEED,epoch=epoch,batch_id=batch_id,probability=0.)
    prefix,lengths=apply_cold_history(x,seed=SEED,epoch=epoch,batch_id=batch_id,probability=.5)
    if not torch.equal(prefix[:,-1],x[:,-1]) or not torch.equal(full,x): raise RuntimeError("history treatment contract")
    losses={}
    for key in CELLS:
        model=cells[key]; opt=optimizers[key]; model.train(); opt.zero_grad(set_to_none=True)
        value=prefix if key.endswith("PREFIX") else full
        pieces=[]
        for offset in range(0,len(value),MICRO):
            width=len(value[offset:offset+MICRO])
            loss=torch.nn.functional.mse_loss(model.forward_last(value[offset:offset+MICRO],bank,dropout_keep=keep[offset:offset+MICRO]).float(),target[offset:offset+MICRO].float())
            if not bool(torch.isfinite(loss)): raise RuntimeError("nonfinite cold-history loss")
            (loss*(width/len(value))).backward(); pieces.append(float(loss.detach())*(width/len(value)))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); opt.step(); emas[key].update_after_step(model); losses[key]=float(sum(pieces))
    return {"rows":len(x),"cold_rows":int((lengths<700).sum()),"loss":losses}

def run(*args, **kwargs):
    """Intentionally gated placeholder for reviewed orchestration, never auto-launches."""
    if os.environ.get("H1_COLD_HISTORY_PROSPECTIVE_GO") != "1": raise RuntimeError("explicit reviewed GO required; no actual phase launched")
    raise RuntimeError("runner wiring awaits root review; helper/protocol only")

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--authorization",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--smoke",type=Path,required=True);p.parse_args(argv);return run()
if __name__=="__main__": main()
