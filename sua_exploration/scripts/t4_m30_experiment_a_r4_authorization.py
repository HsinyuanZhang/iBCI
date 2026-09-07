"""Fixed-path Ed25519 authorization and atomic single-use nonce claim for r4."""
from __future__ import annotations
import base64, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT=Path(__file__).resolve().parents[2]
PUBLIC_KEY=ROOT/'sua_exploration/configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem'
RECEIPT=ROOT/'sua_exploration/results/t4_m30_experiment_a_descriptor_prelaunch_v3_r4_20260802/receipt.json'
AUTH=ROOT/'sua_exploration/results/t4_m30_experiment_a_descriptor_prelaunch_v3_r4_20260802/root_gpu_authorization.json'
SIG=ROOT/'sua_exploration/results/t4_m30_experiment_a_descriptor_prelaunch_v3_r4_20260802/root_gpu_authorization.sig'
CLAIM=ROOT/'sua_exploration/results/sua_t4_m30_component_attribution_v4/authorization_nonce_claim.json'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _load(now:datetime|None=None)->dict:
 if not all(p.is_file() for p in (PUBLIC_KEY,RECEIPT,AUTH,SIG)): raise PermissionError('fixed r4 trust-anchor input missing')
 key=serialization.load_pem_public_key(PUBLIC_KEY.read_bytes())
 if not isinstance(key,Ed25519PublicKey):raise PermissionError('non-Ed25519 public key')
 try:key.verify(base64.b64decode(SIG.read_bytes(),validate=True),AUTH.read_bytes())
 except (InvalidSignature,ValueError) as e:raise PermissionError('bad detached signature') from e
 pre=json.loads(RECEIPT.read_text()); a=json.loads(AUTH.read_text())['authorization']; t=now or datetime.now(timezone.utc)
 issued=datetime.fromisoformat(a['issued_at'].replace('Z','+00:00')); expires=datetime.fromisoformat(a['expires_at'].replace('Z','+00:00'))
 if not(issued<=t<=expires and (expires-issued).total_seconds()<=30*3600):raise PermissionError('invalid authorization window')
 if a.get('gpu_launch_authorized') is not True or not a.get('authorization_id') or not a.get('single_use_nonce'):raise PermissionError('invalid grant/id/nonce')
 if a.get('prelaunch_receipt_sha256')!=sha(RECEIPT) or a.get('prelaunch_source_sha256')!=pre.get('source_sha256'):raise PermissionError('prelaunch binding mismatch')
 if a.get('public_key_path')!=str(PUBLIC_KEY) or a.get('public_key_sha256')!=sha(PUBLIC_KEY):raise PermissionError('public-key binding mismatch')
 for rel, expected in pre['source_sha256'].items():
  if not (ROOT/rel).is_file() or sha(ROOT/rel)!=expected:raise PermissionError(f'source/dependency drift: {rel}')
 return a
def claim() -> dict:
 a=_load(); CLAIM.parent.mkdir(parents=True,exist_ok=True)
 payload=json.dumps({'authorization_id':a['authorization_id'],'single_use_nonce':a['single_use_nonce'],'authorization_sha256':sha(AUTH)},sort_keys=True).encode()+b'\n'
 try:fd=os.open(CLAIM,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 except FileExistsError as e:raise PermissionError('authorization nonce already claimed') from e
 with os.fdopen(fd,'wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
 return a
def require_claim()->dict:
 a=_load()
 if not CLAIM.is_file():raise PermissionError('scheduler nonce claim missing')
 c=json.loads(CLAIM.read_text())
 if (c.get('authorization_id'),c.get('single_use_nonce'),c.get('authorization_sha256')) != (a['authorization_id'],a['single_use_nonce'],sha(AUTH)):raise PermissionError('scheduler nonce claim mismatch')
 return a
