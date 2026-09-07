"""Archive-only descriptive diagnostic for the sealed e8/SPINT minival replay."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RECEIPT=ROOT/'tfpd_exploration/results/m2/family_v1/source_minival_e8_spint_m30_replay_v1.json'
OUT=ROOT/'tfpd_exploration/results/m2/family_v1/e8_source_minival_residual_diagnostic_v1.json'

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def stat(y:np.ndarray,p:np.ndarray)->dict[str,object]:
    err=p-y; centered=y-y.mean(axis=0,keepdims=True)
    sse=(err*err).sum(axis=0); sst=(centered*centered).sum(axis=0)
    # MSE = squared mean error (bias) + variance of the residual, per output.
    bias=err.mean(axis=0); resid_var=((err-bias)**2).mean(axis=0)
    corr=[]
    for j in range(y.shape[1]):
        corr.append(float(np.corrcoef(y[:,j],p[:,j])[0,1]) if y.shape[0]>1 and y[:,j].std()>0 and p[:,j].std()>0 else None)
    return {'target_mean':y.mean(0).tolist(),'target_std':y.std(0).tolist(),'prediction_mean':p.mean(0).tolist(),'prediction_std':p.std(0).tolist(),'prediction_minus_target_bias':bias.tolist(),'residual_variance':resid_var.tolist(),'mse':(err*err).mean(0).tolist(),'bias_squared':(bias*bias).tolist(),'r2_per_output':(1-sse/sst).tolist(),'pearson_correlation_per_output':corr,'target_variance':y.var(0).tolist(),'sse':sse.tolist(),'sst':sst.tolist()}
def run()->dict[str,object]:
    r=json.loads(RECEIPT.read_text()); required={'stream_npz_sha256','rows','e8_equal_session_r2','spint_equal_session_r2'}
    if not required <= set(r): raise RuntimeError('replay receipt schema drift')
    archive=RECEIPT.with_suffix('.npz')
    if sha(archive)!=r['stream_npz_sha256']: raise RuntimeError('sealed replay archive hash drift')
    with np.load(archive,allow_pickle=False) as z:
        e=np.asarray(z['e8_prediction'],dtype=np.float64); s=np.asarray(z['spint_prediction'],dtype=np.float64); y=np.asarray(z['target'],dtype=np.float64); sessions=np.asarray(z['session'])
    if e.shape!=s.shape or e.shape!=y.shape or e.shape!=(1011,2) or sessions.shape!=(1011,): raise RuntimeError('sealed native archive geometry drift')
    rows=[]
    for expected in r['rows']:
        name=expected['session']; ix=sessions==name
        if int(ix.sum())!=expected['window_count']: raise RuntimeError('receipt/session count drift')
        rows.append({'session':name,'window_count':int(ix.sum()),'e8':stat(y[ix],e[ix]),'spint':stat(y[ix],s[ix]),'spint_minus_e8':{'r2_per_output_delta':(np.asarray(stat(y[ix],s[ix])['r2_per_output'])-np.asarray(stat(y[ix],e[ix])['r2_per_output'])).tolist(),'mse_per_output_delta':(np.asarray(stat(y[ix],s[ix])['mse'])-np.asarray(stat(y[ix],e[ix])['mse'])).tolist()}})
    return {'schema':'m2_family_v1_source_minival_residual_diagnostic_v1','status':'DESCRIPTIVE_ARCHIVE_ONLY_NOT_SELECTION','input_receipt_sha256':sha(RECEIPT),'input_npz_sha256':sha(archive),'native_units':True,'no_new_model_forwards':True,'no_fitting_or_selection':True,'interpretation':'Per-output descriptive decomposition only: MSE equals squared bias plus residual variance; it does not establish causal mechanisms or generalization.','overall':{'e8':stat(y,e),'spint':stat(y,s),'recorded_equal_session_r2':{'e8':r['e8_equal_session_r2'],'spint':r['spint_equal_session_r2']}},'rows':rows}
if __name__=='__main__':
    value=run(); OUT.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); print(json.dumps({'out':str(OUT),'sha256':sha(OUT)},sort_keys=True))
