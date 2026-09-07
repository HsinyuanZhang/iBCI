#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9
cd /home/xinyuan/Work_host/SPINT
exec /home/xinyuan/miniconda3/envs/spint/bin/python \
  -m tfpd_exploration.src.m1_temporal_decoder_quick_product_v1.train \
  --arm flat --max-epochs 1
