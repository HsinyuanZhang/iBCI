#!/usr/bin/env python3
"""Ed25519 trust-anchor verifier for the fixed Experiment-A r3 authorization paths.

The detached signature covers the exact authorization JSON bytes.  The private
key is deliberately never read, generated, or logged by this repository.
"""
from __future__ import annotations
import base64, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT=Path(__file__).resolve().parents[2]
PUBLIC_KEY=ROOT/'sua_exploration/configs/t4_m30_experiment_a_v3r3_root_ed25519_public.pem'
AUTHORIZATION=ROOT/'sua_exploration/results/t4_m30_experiment_a_descriptor_prelaunch_v3_r3_20260802/root_gpu_authorization.json'
SIGNATURE=ROOT/'sua_exploration/results/t4_m30_experiment_a_descriptor_prelaunch_v3_r3_20260802/root_gpu_authorization.sig'
def sha(path: Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(*, prelaunch: Path, expected_prelaunch_sha: str, now: datetime|None=None) -> dict:
    if not (PUBLIC_KEY.is_file() and AUTHORIZATION.is_file() and SIGNATURE.is_file()): raise PermissionError('fixed trust-anchor public key, authorization, or signature is absent')
    if sha(prelaunch)!=expected_prelaunch_sha: raise PermissionError('prelaunch receipt SHA mismatch')
    key=serialization.load_pem_public_key(PUBLIC_KEY.read_bytes())
    if not isinstance(key,Ed25519PublicKey): raise TypeError('trust anchor is not Ed25519')
    try: key.verify(base64.b64decode(SIGNATURE.read_bytes(),validate=True),AUTHORIZATION.read_bytes())
    except (InvalidSignature, ValueError) as exc: raise PermissionError('invalid detached Ed25519 authorization signature') from exc
    auth=json.loads(AUTHORIZATION.read_text())['authorization']; current=now or datetime.now(timezone.utc)
    issued=datetime.fromisoformat(auth['issued_at'].replace('Z','+00:00')); expires=datetime.fromisoformat(auth['expires_at'].replace('Z','+00:00'))
    if not (issued <= current <= expires and (expires-issued).total_seconds() <= 30*3600): raise PermissionError('authorization expired, not yet valid, or exceeds 30h maximum')
    pre=json.loads(prelaunch.read_text())
    if auth.get('gpu_launch_authorized') is not True or auth.get('authorization_id') in (None,'') or auth.get('single_use_nonce') in (None,''): raise PermissionError('missing authorization grant/id/nonce')
    if auth.get('prelaunch_receipt_sha256')!=expected_prelaunch_sha or auth.get('prelaunch_source_sha256')!=pre.get('source_sha256'): raise PermissionError('authorization prelaunch binding mismatch')
    if auth.get('public_key_path')!=str(PUBLIC_KEY) or auth.get('public_key_sha256')!=sha(PUBLIC_KEY): raise PermissionError('authorization trust-anchor binding mismatch')
    for rel, expected in pre['source_sha256'].items():
        path=ROOT/rel
        if not path.is_file() or sha(path)!=expected: raise PermissionError(f'v3r3 source drift: {rel}')
    return auth
