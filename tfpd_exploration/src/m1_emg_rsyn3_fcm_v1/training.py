"""Stage-1 training contracts. GPU launch is refused without a capability."""
from tfpd_exploration.src.m1_emg_syn3_fcm_v1.training import (  # noqa: F401
    TrainingError,
    refuse_gpu_without_capability,
    select_fixed_last,
)
