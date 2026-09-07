"""Frozen matched H1 B(full-window)/T(current-query) source-only experiment."""
from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .cache import build_or_load, authority, validate_authority, ROOT as CACHE_ROOT
from .model import make_matched_pair
from .score import evaluate
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

ROOT=Path(__file__).resolve().parents[2]/'results/decoder_validation_v2/20260905_190000/h1/paired_source_v1'
SEED=42; W=700; MICRO=8; EFFECTIVE=32; EPOCHS=12; LR=1e-4; WD=1e-2; EMA=.9995
def _rng(seed, *parts): return np.random.default_rng(int.from_bytes(hashlib.sha256('|'.join(map(str,(seed,*parts))).encode()).digest()[:8],'little'))
def batches(cache, epoch):
    out=[]
    for name,row in cache['train'].items():
        order=_rng(SEED,'sampler',epoch,name).permutation(row['query_starts'])
        out.extend((name,order[i:i+EFFECTIVE]) for i in range(0,len(order),EFFECTIVE))
    order=_rng(SEED,'batch-order',epoch).permutation(len(out)); return [out[i] for i in order]
def collate(row, starts, device):
    x=np.stack([row['neural'][int(s):int(s)+W] for s in starts]).astype(np.float32,copy=False)
    native=np.stack([row['velocity'][int(s)+W-1] for s in starts]).astype(np.float32,copy=False)
    return torch.as_tensor(x,device=device),torch.as_tensor(native*20.,device=device)
def keep_mask(n, epoch, batch, device):
    g=torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(f'{SEED}|dropout|{epoch}|{batch}'.encode()).digest()[:8],'little'))
    return (torch.rand((n,176),generator=g)>=.1).to(device)
def groups(model):
    decay=[]; no=[]
    for n,p in model.named_parameters():
        (no if p.ndim==1 or n.endswith('.bias') or 'norm' in n.lower() else decay).append(p)
    return [{'params':decay,'weight_decay':WD},{'params':no,'weight_decay':0.}]
def state(model,opt,ema,epoch,step):
    return {'model':model.state_dict(),'optimizer':opt.state_dict(),'ema':ema.checkpoint_state(),'epoch':epoch,'step':step,
     'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),'cuda':torch.cuda.get_rng_state_all()},
     'resume_contract':'end-of-epoch only; no mid-epoch resume is implemented or claimed',
     'recipe':{'seed':SEED,'target':'20x native velocity','activity_scale':32,'batch':32,'microbatch':8,'dropout':.1,'lr':LR,'warmup_epochs':1,'ema':EMA}}
def restore(payload,model,opt,ema):
    model.load_state_dict(payload['model']);opt.load_state_dict(payload['optimizer']);ema.load_checkpoint_state(payload['ema']); r=payload['rng'];torch.set_rng_state(r['torch']);np.random.set_state(r['numpy']);random.setstate(r['python']);torch.cuda.set_rng_state_all(r['cuda']);return int(payload['epoch']),int(payload['step'])
def set_lr(opt, step, updates):
    lr=LR*min(1.,step/max(updates,1))
    for g in opt.param_groups:g['lr']=lr
def main(max_epochs=EPOCHS, profile_updates=0, resume=False, attempt='paired_source_v1'):
    global ROOT
    ROOT = ROOT.parent / attempt
    if ROOT.exists() and any(ROOT.iterdir()) and not resume:
      raise FileExistsError(f"refusing to overwrite an existing attempt: {ROOT}")
    ROOT.mkdir(parents=True,exist_ok=True); cache=build_or_load(); recorded=json.loads((CACHE_ROOT/'source_cache_authority.json').read_text());validate_authority(cache,recorded)
    (ROOT/'input_authority.json').write_text(json.dumps({'cache_authority':recorded,'attempt_code_authority':authority(cache)['code_sha256']},indent=2,sort_keys=True)+'\n')
    dev=torch.device('cuda:0');torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    full,t=make_matched_pair(activity_scale=32.);arms={'full':full.to(dev),'t':t.to(dev)};opts={k:torch.optim.AdamW(groups(m),lr=LR) for k,m in arms.items()};emas={k:DecoderEMA(m,decay=EMA) for k,m in arms.items()}
    begin=1;steps={k:0 for k in arms}
    if resume:
      for k in arms:
        p=ROOT/f'{k}_latest.pt'; e,s=restore(torch.load(p,map_location='cpu',weights_only=False),arms[k],opts[k],emas[k]);begin=max(begin,e+1);steps[k]=s
    updates=len(batches(cache,1)); progress_path=ROOT/'progress.json'
    report=json.loads(progress_path.read_text()) if resume and progress_path.is_file() else {'schema':'h1_paired_source_v1','selection':'frozen minival W700 stride4 endpoints; governing metric is EMA concatenated R2; equal-session mean and worst session are report-only','terminal':'complete all-mask stream scored separately','epochs':[], 'resume_contract':'end-of-epoch checkpoints only'}
    start=time.time()
    for epoch in range(begin,max_epochs+1):
      bs=batches(cache,epoch)
      if profile_updates: bs=bs[:profile_updates]
      for b,(name,starts) in enumerate(bs):
        row=cache['train'][name]; x,y=collate(row,starts,dev);bank=H1Bank(row['bank']['E0'].to(dev),row['bank']['T'].to(dev),row['bank']['unit_mask'].to(dev));keep=keep_mask(len(starts),epoch,b,dev)
        for arm in ('full','t'):
          m=arms[arm];m.train();opts[arm].zero_grad(set_to_none=True);steps[arm]+=1;set_lr(opts[arm],steps[arm],updates)
          for off in range(0,len(starts),MICRO):
            pred=m.forward_last(x[off:off+MICRO],bank,dropout_keep=keep[off:off+MICRO]);(F.mse_loss(pred,y[off:off+MICRO])*(len(x[off:off+MICRO])/len(x))).backward()
          torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opts[arm].step();emas[arm].update_after_step(m)
      row={'epoch':epoch,'elapsed_s':time.time()-start,'arms':{}}
      for arm in ('full','t'):
        payload=state(arms[arm],opts[arm],emas[arm],epoch,steps[arm])
        torch.save(payload,ROOT/f'{arm}_epoch_{epoch:03d}.pt')
        torch.save(payload,ROOT/f'{arm}_latest.pt')
        raw=evaluate(arms[arm],device=dev,mode='selection')
        ema_view=emas[arm].score_with_ema(arms[arm],lambda m:evaluate(m,device=dev,mode='selection'))
        row['arms'][arm]={'raw_selection':raw,'ema_selection':ema_view}
      report['epochs'].append(row);progress_path.write_text(json.dumps(report,indent=2)+'\n')
      if profile_updates: break
    if not profile_updates:
      selected={}
      for arm in ('full','t'):
        # max preserves the first occurrence, enforcing the predeclared early tie-break.
        winner=max(report['epochs'],key=lambda r:r['arms'][arm]['ema_selection']['r2_concat'])
        selected[arm]={'epoch':winner['epoch'],'governing_metric':'ema_selection.r2_concat','score':winner['arms'][arm]['ema_selection']['r2_concat'],
          'checkpoint':str(ROOT/f"{arm}_epoch_{winner['epoch']:03d}.pt"),'tie_break':'earliest epoch'}
      freeze={'schema':'h1_paired_source_v1_selection_freeze','selection_surface':'frozen 2908 minival endpoints','selected':selected,
        'terminal_rule':'terminal complete 20325-bin metrics are evaluated only from these frozen selections'}
      (ROOT/'selection_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
      report['terminal_complete_stream']={}
      for arm in ('full','t'):
        payload=torch.load(selected[arm]['checkpoint'],map_location='cpu',weights_only=False)
        arms[arm].load_state_dict(payload['model']); chosen=DecoderEMA(arms[arm],decay=EMA);chosen.load_checkpoint_state(payload['ema'])
        report['terminal_complete_stream'][arm]=chosen.score_with_ema(arms[arm],lambda m:evaluate(m,device=dev,mode='complete'))
      report['endpoint_epoch12_complete_stream']={}
      for arm in ('full','t'):
        payload=torch.load(ROOT/f'{arm}_epoch_{max_epochs:03d}.pt',map_location='cpu',weights_only=False)
        arms[arm].load_state_dict(payload['model']); endpoint=DecoderEMA(arms[arm],decay=EMA);endpoint.load_checkpoint_state(payload['ema'])
        report['endpoint_epoch12_complete_stream'][arm]=endpoint.score_with_ema(arms[arm],lambda m:evaluate(m,device=dev,mode='complete'))
      report['selection_freeze']=freeze
      (ROOT/'final.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--max-epochs',type=int,default=EPOCHS);p.add_argument('--profile-updates',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--attempt',default='paired_source_v1');a=p.parse_args();main(a.max_epochs,a.profile_updates,a.resume,a.attempt)
