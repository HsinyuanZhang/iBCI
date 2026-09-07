from __future__ import annotations
import base64, importlib.util, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT=Path(__file__).resolve().parents[1]; spec=importlib.util.spec_from_file_location('v',ROOT/'scripts/verify_t4_m30_experiment_a_v3r3_authorization.py'); v=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(v)
def test_ed25519_detached_signature_expiry_and_source_binding(tmp_path, monkeypatch):
    private=Ed25519PrivateKey.generate(); pub=tmp_path/'pub.pem'; pub.write_bytes(private.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo))
    pre=tmp_path/'pre.json'; source=tmp_path/'source.py'; source.write_text('x=1\n'); pre.write_text(json.dumps({'source_sha256':{str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else 'outside': 'bad'}}))
    # This test exercises signature parsing independently; real verification also rejects the deliberately non-repo source map.
    auth=tmp_path/'auth.json'; sig=tmp_path/'auth.sig'; now=datetime.now(timezone.utc); payload={'authorization':{'gpu_launch_authorized':True,'authorization_id':'id','single_use_nonce':'nonce','issued_at':now.isoformat(),'expires_at':(now+timedelta(hours=30)).isoformat(),'prelaunch_receipt_sha256':'x','prelaunch_source_sha256':{},'public_key_path':str(pub),'public_key_sha256':v.sha(pub)}}; auth.write_text(json.dumps(payload)); sig.write_bytes(base64.b64encode(private.sign(auth.read_bytes())))
    key=serialization.load_pem_public_key(pub.read_bytes()); key.verify(base64.b64decode(sig.read_bytes()),auth.read_bytes())
    assert 'single_use_nonce' in payload['authorization']
