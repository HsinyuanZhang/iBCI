"""Re-export parent injection; successor does not change P."""
from tfpd_exploration.src.m1_emg_syn3_fcm_v1.model import (  # noqa: F401
    InjectionError,
    TorchCarrierProjection,
    apply_injection,
    torch_apply,
    zero_linear,
)
