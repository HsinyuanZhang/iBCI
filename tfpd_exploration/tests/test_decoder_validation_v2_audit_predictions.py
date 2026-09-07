import numpy as np
import pytest

from tfpd_exploration.src.decoder_validation_v2 import audit_predictions as audit


def test_float64_r2_and_metrics_are_not_float32_accidentally():
    y=np.array([[1e8,1.],[1e8+2.,3.]],dtype=np.float64)
    p=y.copy(); p[0,0]+=0.25
    assert audit._r2(p,y) < 1.0
    assert set(audit._metrics(p,y,np.zeros((1,2))))=={"model","zero","train_only_mean"}


def test_m1_audit_checks_padded_endpoint_and_native_targets(tmp_path,monkeypatch):
    starts=np.array([10,20,30],dtype=np.int64); sessions=np.array(["ses-20120926","ses-20120927","ses-20120928"])
    target=np.arange(48,dtype=np.float32).reshape(3,16); truth={(s,int(i)):v for s,i,v in zip(sessions,starts,target)}
    monkeypatch.setattr(audit,"_m1_source_truth",lambda:(truth,set(truth),"split",target.mean(0,keepdims=True),"train"))
    monkeypatch.setattr(audit,"_m1_paths",lambda:{s:tmp_path/s for s in audit.M1_SESSIONS})
    path=tmp_path/"x.npz"
    np.savez(path,prediction=target,target=target,session=sessions,window_start=starts,window_end_exclusive=starts+100,bin_timestep=starts+99,train_target_mean=target.mean(0),split_train_sha256=np.array(["x"]),split_dev_sha256=np.array(["split"]),checkpoint_sha256=np.array(["x"]),carrier_npz_sha256=np.array(["x"]))
    # Count is intentionally strict: this tiny fixture reaches the count gate.
    with pytest.raises(ValueError,match="31252"):
        audit.audit_m1(path)


def test_m1_audit_valid_export_traverses_full_success_path(tmp_path,monkeypatch):
    n=31252; starts=np.arange(n,dtype=np.int64); sessions=np.asarray([audit.M1_SESSIONS[i%3] for i in range(n)])
    target=np.broadcast_to(np.arange(16,dtype=np.float32),(n,16)).copy()+starts[:,None].astype(np.float32)*1e-3
    truth={(str(s),int(w)):target[i] for i,(s,w) in enumerate(zip(sessions,starts,strict=True))}
    monkeypatch.setattr(audit,"_m1_source_truth",lambda:(truth,set(truth),"dev",target.mean(0,keepdims=True),"train"))
    monkeypatch.setattr(audit,"_m1_paths",lambda:{s:tmp_path/s for s in audit.M1_SESSIONS})
    path=tmp_path/"valid.npz"
    import json
    carrier=json.loads((audit.ROOT/"m1/rSyn3-refit-v1.source-only.receipt.json").read_text())["digests"]["npz"]
    np.savez(path,prediction=target,target=target,session=sessions,window_start=starts,window_end_exclusive=starts+100,bin_timestep=starts+99,train_target_mean=target.mean(0),split_train_sha256=np.array(["train"]),split_dev_sha256=np.array(["dev"]),checkpoint_sha256=np.array(["a"*64]),carrier_npz_sha256=np.array([carrier]))
    out=audit.audit_m1(path)
    assert out["n"]==31252 and out["pooled"]["model"]==1.0


def test_m1_audit_rejects_supplied_checkpoint_with_wrong_bytes(tmp_path,monkeypatch):
    n=31252; starts=np.arange(n,dtype=np.int64); sessions=np.asarray([audit.M1_SESSIONS[i%3] for i in range(n)])
    target=np.broadcast_to(np.arange(16,dtype=np.float32),(n,16)).copy()+starts[:,None].astype(np.float32)
    truth={(str(s),int(w)):target[i] for i,(s,w) in enumerate(zip(sessions,starts,strict=True))}
    monkeypatch.setattr(audit,"_m1_source_truth",lambda:(truth,set(truth),"dev",target.mean(0,keepdims=True),"train"))
    monkeypatch.setattr(audit,"_m1_paths",lambda:{s:tmp_path/s for s in audit.M1_SESSIONS})
    import json
    carrier=json.loads((audit.ROOT/"m1/rSyn3-refit-v1.source-only.receipt.json").read_text())["digests"]["npz"]
    export=tmp_path/"x.npz"; checkpoint=tmp_path/"bad.pt"; checkpoint.write_bytes(b"not-the-exported-checkpoint")
    np.savez(export,prediction=target,target=target,session=sessions,window_start=starts,window_end_exclusive=starts+100,bin_timestep=starts+99,train_target_mean=target.mean(0),split_train_sha256=np.array(["train"]),split_dev_sha256=np.array(["dev"]),checkpoint_sha256=np.array(["a"*64]),carrier_npz_sha256=np.array([carrier]))
    with pytest.raises(ValueError,match="checkpoint bytes"):
        audit.audit_m1(export,checkpoint=checkpoint)
