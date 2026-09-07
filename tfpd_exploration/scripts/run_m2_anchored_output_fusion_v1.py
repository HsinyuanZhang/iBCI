#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from tfpd_exploration.src.m2_anchored_output_fusion_v1 import plan
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--repo-root',default='/home/xinyuan/Work_host/SPINT');a=p.parse_args()
if not a.execute:print(json.dumps({'schema':plan.SCHEMA,'status':'INERT__GPU0_SOURCE_ONLY_EXECUTE_REQUIRED'}))
else:
 from tfpd_exploration.src.m2_anchored_output_fusion_v1.driver import execute
 print(execute(Path(a.repo_root)))
