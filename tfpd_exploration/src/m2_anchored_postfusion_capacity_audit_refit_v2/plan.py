"""Torch-free APFC V2 audit contract; V1 graph literals remain deferred."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
SCHEMA='m2_anchored_postfusion_capacity_audit_refit_v2'
ROOT_RELATIVE='tfpd_exploration/results/m2_anchored_postfusion_capacity_audit_refit_v2'
WORKORDER_RELATIVE='tfpd_exploration/docs/WORKORDER_ANCHORED_POSTFUSION_CAPACITY_AUDIT_REFIT_V2_20260902.md'
WORKORDER_SHA256='ecbb9528922db1f535968a67f2badbe2812bafa0a0af0604d19dc3aa64817cd2'
V1_ROOT_RELATIVE='tfpd_exploration/results/m2_anchored_postfusion_capacity_screen_v1/screen'
V1_BODIES={
 'attempt.json':'3b076d78381eef5c2513481e98aed258dc660641c2dc99c6cac7fa2c0c1146e4',
 'launch.json':'b0e3c4270f9d3533c02a3d7ececfc46f0a2b484d15cae1254d2d33ac2ba3d379',
 'source_authority.json':'471166d635b1711ff06b2e187a55c4a33d3bf98236a71c4edacd34b1c7331625',
 'screen.json':'8cba5b6dc95222cc935242cb3be1fc6edad17d42cc6fcf5c28c726027032e84c',
 'terminal.json':'a63fe5755238aafdcf117f7b70c74858d7288ccaf875aec2cdad299f0474b16a',
}
ARMS=('A-S1','A-TB4','A-DC2'); EPOCHS=12; V1_SCALAR_GAIN=.0022026004
PREDICTION_TOLERANCE=2e-6; R2_TOLERANCE=2e-7
CAPACITY_GATE={'mean_vs_scalar':.003,'positive_sessions':2,'worst_vs_scalar':-.002,'mean_vs_zero':.005}
def canonical_json(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha256_bytes(x):return hashlib.sha256(x).hexdigest()
def sha256_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
