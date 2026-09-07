"""Frozen APFC V1 source-only contract; intentionally Torch-free."""
from __future__ import annotations

import hashlib, json
from pathlib import Path

SCHEMA='m2_anchored_postfusion_capacity_screen_v1'
CELL='M2_ANCHORED_POSTFUSION_CAPACITY_SCREEN_V1'
RESULT_ROOT_RELATIVE='tfpd_exploration/results/m2_anchored_postfusion_capacity_screen_v1/screen'
DESIGN_RELATIVE='tfpd_exploration/docs/DESIGN_ANCHORED_POSTFUSION_CAPACITY_SCREEN_V1_20260902.md'
DESIGN_SHA256='ba1bf5b6257dac34d2ef7fb0ff07f3d1e295ff8e0ec84ba74e18a21261be958b'
WORKORDER_RELATIVE='tfpd_exploration/docs/WORKORDER_ANCHORED_POSTFUSION_CAPACITY_SCREEN_V1_20260902.md'
WORKORDER_SHA256='1846b7456908b5ed63a8e832fe8d0e1f9376e96baf8bf8507b390354fb195d44'
GPU_UUID='GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9'
LIVE_ENV={'CUDA_VISIBLE_DEVICES':'0','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8',
          'PYTHONHASHSEED':'0','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
          'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
          'PYTHONPATH':'/home/xinyuan/Work_host/SPINT'}
SEED=42; EPOCHS=12; BATCH_SIZE=32; WINDOW_BINS=50; TRIAL_BINS=100; SIDE_DIM=4
ARMS=('A-S1','A-TB4','A-DC2'); POOL_CYCLE=(4,10,30); SUPPORT_COUNT=4
ADAM_LR=1e-4; ADAM_BETAS=(0.9,0.999); ADAM_EPS=1e-8; ADAM_WEIGHT_DECAY=0.; ADAM_AMSGRAD=False
SOURCE_SESSIONS=7; FIT_SESSIONS=5; VALIDATION_SESSIONS=2
SELECTED_CHECKPOINT_SHA256='25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e'
SELECTED_STUDENT_STATE_SHA256='2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20'
ACTIVITY_AUTHORITY='pooled_g00m_linear'
V1_SCALAR_DELTA=0.0022026004
STATIC_CLOSURE_RELATIVES=(DESIGN_RELATIVE,WORKORDER_RELATIVE,
 'tfpd_exploration/src/m2_anchored_postfusion_capacity_screen_v1/__init__.py',
 'tfpd_exploration/src/m2_anchored_postfusion_capacity_screen_v1/plan.py',
 'tfpd_exploration/src/m2_anchored_postfusion_capacity_screen_v1/gates.py',
 'tfpd_exploration/src/m2_anchored_postfusion_capacity_screen_v1/runner.py',
 'tfpd_exploration/src/m2_anchored_postfusion_capacity_screen_v1/driver.py',
 'tfpd_exploration/src/m2_anchored_postfusion_gate_v1/adapter.py',
 'tfpd_exploration/src/m2_anchored_postfusion_gate_v1/plan.py',
 'tfpd_exploration/src/m2_anchored_postfusion_gate_v1/runtime.py',
 'tfpd_exploration/src/m2_anchored_postfusion_gate_v1/source_replay.py',
 'tfpd_exploration/src/m2_anchored_postfusion_gate_v1/pools.py',
 'tfpd_exploration/src/m2_anchored_postfusion_gate_v1/selection.py',
 'tfpd_exploration/src/m2_anchored_postfusion_gate_v1/training.py',
 'tfpd_exploration/src/pit_m2_v1/trainer.py',
 'tfpd_exploration/src/m2_postfusion_checkpoint_score_v1/physical.py',
 'tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/physical.py',
 'tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py',
 'tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py',
 'streaming_calibration_exp/src/models/components/streaming_spint.py',
 'streaming_calibration_exp/src/models/components/streaming_encoders.py',
 'streaming_calibration_exp/src/models/streaming_calibration_module.py',
 'streaming_calibration_exp/src/data/falcon_datamodule.py',
 'tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py',
 'tfpd_exploration/scripts/run_m2_anchored_postfusion_capacity_screen_v1.py')
def canonical_json(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
def sha256_bytes(x): return hashlib.sha256(x).hexdigest()
def sha256_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def closure(root):
    root=Path(root); out={}
    for rel in STATIC_CLOSURE_RELATIVES:
        p=root/rel
        if not p.is_file(): raise ValueError(f'APFC closure leaf missing: {rel}')
        out[rel]=sha256_file(p)
    return out
