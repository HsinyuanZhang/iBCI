"""EvalAI entry for the M1 current-query Transformer runtime."""
import argparse, os
for _k,_v in (("OMP_NUM_THREADS","2"),("MKL_NUM_THREADS","2"),("OPENBLAS_NUM_THREADS","2"),("NUMEXPR_NUM_THREADS","2")):
    os.environ.setdefault(_k,_v)
import torch
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from trf_falcon_decoder import TrfFalconDecoder
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--evaluation", choices=("local","remote"), required=True)
    p.add_argument("--model-path", default="/data/decoder.pkl")
    p.add_argument("--split", default="m1")
    p.add_argument("--phase", default="test")
    p.add_argument("--batch-size", type=int, default=4)
    a=p.parse_args()
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    cfg=FalconConfig(task=FalconTask.m1)
    dec=TrfFalconDecoder(task_config=cfg, model_path=a.model_path, batch_size=a.batch_size)
    ev=FalconEvaluator(eval_remote=a.evaluation=="remote", split="m1", dataloader_workers=0)
    ev.evaluate(dec, phase=a.phase)
if __name__=="__main__":
    main()
