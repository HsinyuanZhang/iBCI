"""EvalAI entry for the sealed M1 RIFT ACTIVITY_ONLY CPU payload."""
import argparse, os, sys
for k,v in (("OMP_NUM_THREADS","2"),("MKL_NUM_THREADS","2"),("OPENBLAS_NUM_THREADS","2"),("NUMEXPR_NUM_THREADS","2")): os.environ.setdefault(k,v)
sys.path.insert(0,os.environ.get("RIFT_PKG","/pkg"))
import torch
from falcon_challenge.config import FalconConfig,FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from m1_rift_falcon_decoder import M1RiftActivityOnlyCachedFalconDecoder
def main():
 p=argparse.ArgumentParser();p.add_argument("--evaluation",choices=("local","remote"),required=True);p.add_argument("--model-path",default="/data/decoder.pkl");p.add_argument("--split",choices=("m1",),default="m1");p.add_argument("--phase",choices=("minival","test"),default="test");p.add_argument("--batch-size",type=int,default=4);a=p.parse_args();torch.set_num_threads(2)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 evaluator=FalconEvaluator(eval_remote=a.evaluation=="remote",split="m1",dataloader_workers=0)
 evaluator.evaluate(M1RiftActivityOnlyCachedFalconDecoder(FalconConfig(task=FalconTask.m1),a.model_path,a.batch_size),phase=a.phase)
if __name__=="__main__":main()
