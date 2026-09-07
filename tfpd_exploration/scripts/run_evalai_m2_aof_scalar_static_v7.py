"""Inert AOF-S V7 entrypoint; no capability can be issued here."""
from __future__ import annotations
import argparse
def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");p.parse_args();print("READY_REQUIRES_OPAQUE_CAPABILITY; no capability is minted and no Docker/data/result action is performed.");return 0
if __name__=="__main__":raise SystemExit(main())
