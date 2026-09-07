#!/bin/bash
# EP-FILM H1 submission: push + register (user-run; proxy comes from .bashrc)
set -e
cd /home/xinyuan/Work_host/SPINT

/home/xinyuan/miniconda3/envs/spint/bin/python \
  tfpd_exploration/h1_series_20260830/scripts/push_h1_epfilm_evalai_v1.py \
  --execute \
  --confirm-image-id sha256:3180ac0b2ac117f77dda789245ce63c3ce9aab09509b81dc969856a650d97c7b

echo "=== push done, registering submission ==="
/home/xinyuan/miniconda3/envs/spint/bin/python \
  tfpd_exploration/h1_series_20260830/scripts/record_h1_epfilm_submission_v1.py --execute

echo "=== ALL DONE ==="
