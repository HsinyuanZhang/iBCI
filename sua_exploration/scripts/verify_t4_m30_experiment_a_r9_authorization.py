#!/usr/bin/env python3
from t4_m30_experiment_a_r9_authorization import claim,require_claim
import argparse
p=argparse.ArgumentParser();p.add_argument('--claim',action='store_true');p.add_argument('--require-claim',action='store_true');a=p.parse_args()
if a.claim==a.require_claim:raise SystemExit(2)
(claim if a.claim else require_claim)()
