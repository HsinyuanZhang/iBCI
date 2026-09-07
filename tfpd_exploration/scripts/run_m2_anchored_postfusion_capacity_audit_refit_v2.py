#!/usr/bin/env python3
from __future__ import annotations
import json
from tfpd_exploration.src.m2_anchored_postfusion_capacity_audit_refit_v2 import plan
print(json.dumps({'schema':plan.SCHEMA,'status':'INERT__V1_TERMINAL_LITERALS_REQUIRED','v1_literals_bound':plan.V1_BODIES is not None},sort_keys=True))
