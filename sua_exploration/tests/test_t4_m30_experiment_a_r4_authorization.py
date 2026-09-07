from __future__ import annotations
import base64, importlib.util, json
from datetime import datetime,timedelta,timezone
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT=Path(__file__).resolve().parents[1]; spec=importlib.util.spec_from_file_location('r4',ROOT/'scripts/t4_m30_experiment_a_r4_authorization.py'); r4=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(r4)
def install(tmp_path, *, issued=None, expires=None):
 tmp_path.mkdir(parents=True,exist_ok=True)
 key=Ed25519PrivateKey.generate(); pub=tmp_path/'pub.pem'; pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)); source=tmp_path/'x.py';source.write_text('ok\n'); pre=tmp_path/'pre.json'; pre.write_text(json.dumps({'source_sha256':{'x.py':r4.sha(source)}})); now=datetime.now(timezone.utc); issued=issued or now-timedelta(minutes=1); expires=expires or now+timedelta(hours=1)
 auth=tmp_path/'auth.json'; sig=tmp_path/'auth.sig'; payload={'authorization':{'gpu_launch_authorized':True,'authorization_id':'a','single_use_nonce':'n','issued_at':issued.isoformat(),'expires_at':expires.isoformat(),'prelaunch_receipt_sha256':r4.sha(pre),'prelaunch_source_sha256':json.loads(pre.read_text())['source_sha256'],'public_key_path':str(pub),'public_key_sha256':r4.sha(pub)}};auth.write_text(json.dumps(payload));sig.write_bytes(base64.b64encode(key.sign(auth.read_bytes())))
 r4.ROOT=tmp_path;r4.PUBLIC_KEY=pub;r4.RECEIPT=pre;r4.AUTH=auth;r4.SIG=sig;r4.CLAIM=tmp_path/'claim.json';return source,auth,sig
def test_atomic_nonce_claim_and_replay_signature_and_expiry(tmp_path):
 source,auth,sig=install(tmp_path); assert r4.claim()['authorization_id']=='a'
 try:r4.claim();assert False
 except PermissionError:pass
 auth.write_text('{}')
 try:r4.require_claim();assert False
 except PermissionError:pass
def test_future_expired_and_long_authorization_fail(tmp_path):
 now=datetime.now(timezone.utc)
 for issued,expires in ((now+timedelta(hours=1),now+timedelta(hours=2)),(now-timedelta(hours=2),now-timedelta(hours=1)),(now-timedelta(minutes=1),now+timedelta(hours=31))):
  install(tmp_path/str(issued.timestamp()),issued=issued,expires=expires)
  try:r4.claim();assert False
  except PermissionError:pass
