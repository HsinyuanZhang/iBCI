"""Frozen pre-run settings for the authorized-to-be-started-later paired run."""
from pathlib import Path

SCHEMA = "m2_family_v1_paired_source_only_v1"
TRAIN_ENV_FLAG = "M2_FAMILY_V1_TRAIN"
SEED = 42
EPOCHS = 24
UPDATES_PER_EPOCH = 3165
SMOKE_MAX_STEPS = 100
TRAIN_DTYPE = "float32"  # Explicitly frozen; this runner has no AMP/BF16 autocast.
LOG_EVERY_STEPS = 20
PAIR_WALLCLOCK_BUDGET_SECONDS = 6 * 3600
EMA_DECAY = 0.9995
LR_MAX, LR_MIN = 3e-4, 3e-5
WEIGHT_DECAY, GRAD_CLIP = 1e-2, 1.0
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/results/m2/family_v1/paired_source_only_v1"
SMOKE_ROOT = REPO_ROOT / "tfpd_exploration/results/m2/family_v1/source_only_smoke_v1"
SMOKE_V2_ROOT = REPO_ROOT / "tfpd_exploration/results/m2/family_v1/source_only_smoke_v2"
MEMBERS = ("FLAT", "ROUTE")
