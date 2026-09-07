#!/usr/bin/env python3
"""Validate root authorization against the immutable v3 prelaunch receipt."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--authorization',type=Path,required=True); p.add_argument('--authorization-sha256',required=True); p.add_argument('--prelaunch',type=Path,required=True); p.add_argument('--prelaunch-sha256',required=True); a=p.parse_args()
 if sha(a.authorization)!=a.authorization_sha256: raise ValueError('authorization receipt hash mismatch')
 if sha(a.prelaunch)!=a.prelaunch_sha256: raise ValueError('prelaunch receipt hash mismatch')
 auth=json.loads(a.authorization.read_text()).get('authorization',{}); pre=json.loads(a.prelaunch.read_text())
 if auth.get('gpu_launch_authorized') is not True: raise ValueError('root GPU authorization absent')
 if auth.get('prelaunch_receipt_sha256') != a.prelaunch_sha256: raise ValueError('authorization does not bind v3 receipt SHA')
 if auth.get('prelaunch_source_sha256') != pre.get('source_sha256'): raise ValueError('authorization does not bind exact v3 source hashes')
 if pre.get('status') != 'PASS' or pre.get('authorization',{}).get('gpu_launch_authorized') is not False: raise ValueError('invalid v3 prelaunch status')
 root=Path(__file__).resolve().parents[2]
 for rel, expected in pre.get('source_sha256',{}).items():
  path=root/rel
  if not path.is_file() or sha(path) != expected: raise ValueError(f'v3 source drift: {rel}')
if __name__=='__main__': main()
