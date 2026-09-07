"""Frozen fresh-init V4 matched FULL/T formal source run; no resume path."""
from __future__ import annotations
import hashlib,json,random,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.h1_optimized_v2.score import evaluate
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .model import make_matched_pair
OUT=ROOT/'paired_v4_signed_12ep_v1';SEED=42;W=700;EPOCHS=12;LR=1e-4;MICRO=8;EFFECTIVE=32;EMA=.9995
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_sha(m):
 h=hashlib.sha256()
 for n,t in sorted(m.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def rng(*x):return np.random.default_rng(int.from_bytes(hashlib.sha256('|'.join(map(str,(SEED,*x))).encode()).digest()[:8],'little'))
def batches(cache,epoch):
 o=[]
 for n,row in sorted(cache['train'].items()):
  s=rng('sampler',epoch,n).permutation(row['query_starts']);o.extend((n,s[i:i+EFFECTIVE]) for i in range(0,len(s),EFFECTIVE))
 q=rng('order',epoch).permutation(len(o));return [o[i] for i in q]
def collate(row,s,d):return torch.as_tensor(np.stack([row['neural'][int(x):int(x)+W] for x in s]).astype('float32'),device=d),torch.as_tensor(np.stack([row['velocity'][int(x)+W-1] for x in s]).astype('float32')*20,device=d)
def keep(n,e,b,d):
 g=torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(f'{SEED}|keep|{e}|{b}'.encode()).digest()[:8],'little')); k=torch.rand((n,176),generator=g)>=.1
 # whole-unit dropout cannot yield empty rows.
 k[k.sum(-1)==0,0]=True;return k.to(d)
def ckpt(m,o,ema,e,step):return {'model':m.state_dict(),'optimizer':o.state_dict(),'ema':ema.checkpoint_state(),'epoch':e,'step':step,'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),'cuda':torch.cuda.get_rng_state_all()},'frontend_contract_version':4,'recipe':'paired_v4_signed_12ep_v1'}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth)
 recipe={'seed':42,'init':'fresh make_matched_pair V4; never probe warmstart','frontend_contract_version':4,'activity_scale':1,'window':700,'units':176,'out_dim':7,'target':'20x native velocity, predict divide20','epochs':12,'lr':LR,'weight_decay':.01,'optimizer':'AdamW decay except bias/norm/1d zero','schedule':'linear warmup epoch1 then constant','clip_norm':1,'dropout':'deterministic per-example whole-unit p=.10, same keep FULL/T','effective_batch':32,'microbatch':8,'ema':EMA,'selection':'frozen minival 2908 endpoints, EMA pooled R2, earliest epoch tie','complete':'all 20325 minival eval-mask bins post-selection'}
 OUT.mkdir(parents=True);code={x:sha(Path(__file__).with_name(x)) for x in ('model.py','paired_train.py')};(OUT/'input_authority.json').write_text(json.dumps({'source_cache_authority':auth,'code_sha256':code,'recipe':recipe},indent=2,sort_keys=True)+'\n')
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);d=torch.device('cuda:0');full,t=make_matched_pair();arms={'full':full.to(d),'t':t.to(d)}
 # Exact matched shared components at fresh initialization.
 for key in ('frontend','final_norm','readout'):
  a,b=getattr(arms['full'],key).state_dict(),getattr(arms['t'],key).state_dict()
  if any(not torch.equal(a[n],b[n]) for n in a):raise RuntimeError('matched init drift '+key)
 initial={k:state_sha(v) for k,v in arms.items()};opt={k:torch.optim.AdamW(groups(v),lr=LR) for k,v in arms.items()};emas={k:DecoderEMA(v,decay=EMA) for k,v in arms.items()};steps={k:0 for k in arms};report={'schema':'h1_v4_signed_matched_formal_v1','status':'RUNNING','recipe':recipe,'initial_state_sha256':initial,'epochs':[]};start=time.time();updates=len(batches(cache,1))
 for epoch in range(1,EPOCHS+1):
  for bi,(n,s) in enumerate(batches(cache,epoch)):
   row=cache['train'][n];x,y=collate(row,s,d);b=H1Bank(*[row['bank'][q].to(d) for q in ('E0','T','unit_mask')]);k=keep(len(s),epoch,bi,d)
   for arm,m in arms.items():
    m.train();opt[arm].zero_grad(set_to_none=True);steps[arm]+=1;lr=LR*min(1.,steps[arm]/updates)
    for pg in opt[arm].param_groups:pg['lr']=lr
    for i in range(0,len(s),MICRO):(F.mse_loss(m.forward_last(x[i:i+MICRO],b,k[i:i+MICRO]),y[i:i+MICRO])*(len(x[i:i+MICRO])/len(x))).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt[arm].step();emas[arm].update_after_step(m)
  er={'epoch':epoch,'elapsed_s':time.time()-start,'arms':{}}
  for arm,m in arms.items():
   p=ckpt(m,opt[arm],emas[arm],epoch,steps[arm]);torch.save(p,OUT/f'{arm}_epoch_{epoch:03d}.pt');torch.save(p,OUT/f'{arm}_latest.pt')
   raw=evaluate(m,device=d,mode='selection');ema=emas[arm].score_with_ema(m,lambda z:evaluate(z,device=d,mode='selection'));er['arms'][arm]={'raw_selection_diagnostic':raw,'ema_selection_governing':ema}
  report['epochs'].append(er);(OUT/'progress.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 selected={}
 for arm in arms:
  win=max(report['epochs'],key=lambda x:x['arms'][arm]['ema_selection_governing']['r2_concat']);selected[arm]={'epoch':win['epoch'],'ema_r2_concat':win['arms'][arm]['ema_selection_governing']['r2_concat'],'checkpoint':str(OUT/f'{arm}_epoch_{win["epoch"]:03d}.pt'),'tie_break':'earliest'}
 freeze={'schema':'h1_v4_selection_freeze_v1','selection_surface':'frozen 2908 minival endpoints','selected':selected};(OUT/'selection_freeze.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n');report['selection_freeze']=freeze;report['complete_selected']={};report['complete_epoch12']={}
 for arm,m in arms.items():
  for label,path in [('complete_selected',Path(selected[arm]['checkpoint'])),('complete_epoch12',OUT/f'{arm}_epoch_012.pt')]:
   p=torch.load(path,map_location=d,weights_only=False);m.load_state_dict(p['model'],strict=True);e=DecoderEMA(m,decay=EMA);e.load_checkpoint_state(p['ema']);report[label][arm]=e.score_with_ema(m,lambda z:evaluate(z,device=d,mode='complete'))
 report['status']='COMPLETE';report['trained_state_sha256']={k:state_sha(v) for k,v in arms.items()};(OUT/'final.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
