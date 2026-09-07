#!/usr/bin/env python3
"""APFC V1 entrypoint. `--execute` is root-authorized source-only GPU0 work."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from tfpd_exploration.src.m2_anchored_postfusion_capacity_screen_v1 import plan
def main():
 p=argparse.ArgumentParser(); p.add_argument('--execute',action='store_true'); p.add_argument('--repo-root',default='/home/xinyuan/Work_host/SPINT'); a=p.parse_args()
 if not a.execute:
  print(json.dumps({'schema':plan.SCHEMA,'status':'INERT__PASS_EXECUTE_FOR_AUTHORIZED_SOURCE_ONLY_GPU0_SCREEN','result_root_relative':plan.RESULT_ROOT_RELATIVE},sort_keys=True)); return
 from tfpd_exploration.src.m2_anchored_postfusion_capacity_screen_v1.driver import execute_source_screen
 print(execute_source_screen(Path(a.repo_root)))
if __name__=='__main__':main()
